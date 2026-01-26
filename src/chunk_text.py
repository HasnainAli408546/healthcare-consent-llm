"""
Section-Aware Chunking Pipeline for Healthcare Consent Documents
Splits documents by legal/clinical sections while preserving structure and traceability.
"""

import json
import re
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConsentDocumentChunker:
    """Chunks healthcare consent and intake documents by section-like headings."""

    # Domain-specific section title patterns (case-insensitive)
    # Tuned to your dental consent / intake / HIPAA / financial policy samples.
    SECTION_PATTERNS = [
        r"\bINFORMED CONSENT\b",
        r"\bCONSENT FOR\b",
        r"\bCONSENT TO\b",
        r"\bCONSENT TO PROCEED\b",
        r"\bGENERAL CONSENT\b",
        r"\bRECOMMENDED TREATMENT\b",
        r"\bTREATMENT ALTERNATIVES\b",
        r"\bRISKS? AND COMPLICATIONS?\b",
        r"\bRISKS? / COMPLICATIONS?\b",
        r"\bMANAGING PROFESSIONAL RISKS\b",
        r"\bNOTICE OF PRIVACY PRACTICES\b",
        r"\bPRIVACY PRACTICES\b",
        r"\bHIPAA\b",
        r"\bFINANCIAL POLICY\b",
        r"\bFINANCIAL RESPONSIBILITY\b",
        r"\bPATIENT INFORMATION\b",
        r"\bPATIENT DENTAL INTAKE FORM\b",
        r"\bNEW PATIENT INTAKE FORM\b",
        r"\bMEDICAL HISTORY\b",
        r"\bDENTAL HISTORY\b",
        r"\bDENTAL INSURANCE\b",
        r"\bINSURANCE INFORMATION\b",
        r"\bEMERGENCY CONTACT\b",
        r"\bACKNOWLEDGEMENT OF RECEIPT\b",
        r"\bACKNOWLEDGEMENT OF RECEIPT OF\b",
        r"\bACKNOWLEDGEMENT\b",
        r"\bFINANCIAL POLICY\b",
        r"\bVIRTUAL TELEDENTISTRY\b",
        r"\bINFORMED CONSENT FORM\b",
        r"\bPERIODONTAL TREATMENT REFUSAL\b",
        r"\bTREATMENT REFUSAL\b",
        r"\bGENERAL DENTAL CONSENT\b",
        r"\bCONSENT FORM\b",
        r"\bPOST[- ]OPERATIVE\b",
    ]

    # Things that typically indicate signature blocks / admin tails
    SIGNATURE_PATTERNS = [
        r"\bSIGNATURE\b",
        r"\bPATIENT/ PARENT/ GUARDIAN\b",
        r"\bPATIENT/ PARENT / GUARDIAN\b",
        r"\bPATIENT/ PARENT\b",
        r"\bPATIENT OR PARENT\b",
        r"\bPATIENT NAME\b",
        r"\bPATIENT FULL NAME\b",
        r"\bWITNESS\b",
        r"\bDATE\b",
        r"\bRELATIONSHIP\b",
        r"\bACKNOWLEDGEMENT OF RECEIPT\b",
    ]

    def __init__(
        self,
        text_dir: str = "data/processed/text",
        output_dir: str = "data/processed/chunks",
    ):
        self.text_dir = Path(text_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compiled_section_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SECTION_PATTERNS
        ]
        self.compiled_signature_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SIGNATURE_PATTERNS
        ]

        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
            "files_with_sections": 0,
            "files_without_sections": 0,
            "total_sections": 0,
        }

    # ---------- Heuristics ----------

    def _looks_like_header_line(self, line: str) -> bool:
        """
        Heuristic for 'header-ish' line:
        - Non-empty.
        - Not too short and not extremely long.
        - Either mostly uppercase, or matches one of the known section phrases
          anywhere in the line (not just at the start).
        """
        stripped = line.strip()
        if not stripped:
            return False

        # Too short or too long lines are unlikely to be section headings
        if len(stripped) < 5:
            return False
        if len(stripped) > 140:
            return False

        # Strip leading non-letters (e.g., bullets, "Date______")
        core = re.sub(r"^[^A-Za-z]+", "", stripped)

        if not core:
            return False

        # Check if "core" is mostly uppercase letters (ignoring punctuation/digits)
        letters = [ch for ch in core if ch.isalpha()]
        if letters:
            upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        else:
            upper_ratio = 0.0

        if upper_ratio >= 0.6:
            return True

        # If not mostly uppercase, still allow if it matches a known section phrase
        for pattern in self.compiled_section_patterns:
            if pattern.search(core):
                return True

        return False

    def is_section_header(self, line: str) -> bool:
        """
        Final decision for section header.
        """
        return self._looks_like_header_line(line)

    def is_signature_section(self, title: str) -> bool:
        for p in self.compiled_signature_patterns:
            if p.search(title):
                return True
        # Also treat any line that heavily mentions "signature" or "witness" as signature
        if re.search(r"signature|witness|guardian", title, re.IGNORECASE):
            return True
        return False

    def infer_document_type(self, filename: str) -> str:
        name = filename.lower()
        if "intake" in name:
            return "intake"
        if "consent" in name:
            return "consent"
        if "privacy" in name or "hipaa" in name:
            return "privacy_notice"
        if "financial" in name:
            return "financial_policy"
        return "unknown"

    # ---------- Core logic ----------

    def detect_sections(self, lines: List[str]) -> List[Dict]:
        """
        Scan line-by-line and identify section header lines.
        Each header is recorded with its line index and cleaned title.
        """
        sections = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            if self.is_section_header(stripped):
                title = stripped.rstrip(":").strip()
                sections.append(
                    {
                        "title": title,
                        "line_index": idx,
                    }
                )

        return sections

    def chunk_by_sections(self, text: str, source_file: str) -> Dict:
        lines = text.splitlines()
        section_headers = self.detect_sections(lines)
        document_type = self.infer_document_type(source_file)

        # No detected sections: fallback to a single full-text chunk
        if not section_headers:
            self.stats["files_without_sections"] += 1
            return {
                "source_file": source_file,
                "document_type": document_type,
                "has_sections": False,
                "full_text": text.strip(),
                "char_count": len(text),
                "sections": [],
            }

        self.stats["files_with_sections"] += 1
        sections = []

        for i, header in enumerate(section_headers):
            start = header["line_index"] + 1
            end = (
                section_headers[i + 1]["line_index"]
                if i + 1 < len(section_headers)
                else len(lines)
            )

            content_lines = lines[start:end]
            content = "\n".join(content_lines).strip()
            if not content:
                continue

            section_title = header["title"]
            section = {
                "section_index": i + 1,
                "section_title": section_title,
                "content": content,
                "char_count": len(content),
                "line_count": content.count("\n") + 1,
                "is_signature_section": self.is_signature_section(section_title),
            }

            sections.append(section)
            self.stats["total_sections"] += 1

        return {
            "source_file": source_file,
            "document_type": document_type,
            "has_sections": True,
            "section_count": len(sections),
            "total_char_count": len(text),
            "sections": sections,
        }

    def process_file(self, text_file: Path) -> Dict:
        logger.info(f"Processing: {text_file.name}")

        try:
            with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            result = self.chunk_by_sections(text, text_file.name)

            output_file = self.output_dir / f"{text_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            self.stats["files_processed"] += 1

            logger.info(
                f"✓ {result.get('section_count', 0)} sections written → {output_file.name}"
            )

            return {
                "source_file": str(text_file),
                "output_file": str(output_file),
                "status": "success",
            }

        except Exception as e:
            self.stats["files_failed"] += 1
            logger.error(f"✗ Failed: {text_file.name} → {str(e)}")

            return {
                "source_file": str(text_file),
                "output_file": None,
                "status": "failed",
                "error": str(e),
            }

    def chunk_all(self) -> Dict:
        logger.info("=" * 60)
        logger.info("Starting Section-Aware Chunking Pipeline")
        logger.info("=" * 60)

        text_files = list(self.text_dir.glob("*.txt"))
        results = []

        for text_file in text_files:
            results.append(self.process_file(text_file))

        logger.info("=" * 60)
        logger.info("Chunking Complete")
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files failed: {self.stats['files_failed']}")
        logger.info(f"Files with sections: {self.stats['files_with_sections']}")
        logger.info(f"Files without sections: {self.stats['files_without_sections']}")
        logger.info(f"Total sections: {self.stats['total_sections']}")
        logger.info("=" * 60)

        return {"results": results, "stats": self.stats}


def main():
    chunker = ConsentDocumentChunker()
    chunker.chunk_all()


if __name__ == "__main__":
    main()
