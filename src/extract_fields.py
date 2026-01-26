"""
Field Extraction Pipeline for Healthcare Consent Documents
Maps sectioned consent documents into a normalized JSON schema.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConsentFieldExtractor:
    """
    Extracts structured fields from chunked consent documents
    according to schema.json.
    """

    def __init__(
        self,
        chunks_dir: str = "data/processed/chunks",
        schema_path: str = "schema.json",
    ):
        self.chunks_dir = Path(chunks_dir)
        self.schema_path = Path(schema_path)

        # Where to save extracted field JSONs
        self.output_dir = self.chunks_dir / "fields"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load schema meta (optional, currently just for version / type)
        self.schema_meta = self._load_schema()

        # Stats
        self.stats = {
            "files_processed": 0,
            "files_failed": 0,
        }

    # ---------- IO helpers ----------

    def _load_schema(self) -> Dict[str, Any]:
        if not self.schema_path.exists():
            logger.warning(
                f"schema.json not found at {self.schema_path}, proceeding without validation"
            )
            return {}

        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load schema.json: {e}")
            return {}

    def _load_chunk_file(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_extracted_fields(
        self, source_chunk_file: Path, fields: Dict[str, Any]
    ) -> Path:
        out_path = self.output_dir / source_chunk_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fields, f, indent=2, ensure_ascii=False)
        return out_path

    # ---------- Core helpers ----------

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _section_title_contains(title: str, *keywords: str) -> bool:
        t = title.lower()
        return any(k.lower() in t for k in keywords)

    @staticmethod
    def _text_contains(text: str, *keywords: str) -> bool:
        t = text.lower()
        return any(k.lower() in t for k in keywords)

    # ---------- Document kind ----------

    def infer_document_kind(self, chunk_data: Dict[str, Any]) -> str:
        """
        Rough document classifier:
        - intake
        - privacy_notice
        - financial_policy
        - consent
        - refusal
        - mixed
        - unknown
        """
        source = (chunk_data.get("source_file") or "").lower()
        titles = " ".join(
            s.get("section_title", "") for s in chunk_data.get("sections", [])
        ).lower()
        all_text = " ".join(
            (s.get("section_title", "") + " " + s.get("content", ""))
            for s in chunk_data.get("sections", [])
        ).lower()

        text = f"{source} {titles}"

        is_intake = "intake" in text or "new patient" in text
        is_privacy = (
            "privacy" in text or "notice of privacy" in text or "hipaa" in text
        )
        is_financial = "financial policy" in text or "financial" in text
        is_consent = "consent" in text
        is_refusal = "refusal" in text or "treatment refusal" in text

        # Explicit refusal trumps generic consent
        if is_refusal:
            return "refusal"

        # Avoid calling pure intake+financial "mixed" unless "consent" appears
        if is_intake and is_financial and not is_consent and not is_privacy:
            return "intake"

        flags = [is_intake, is_privacy, is_financial, is_consent]
        if sum(flags) > 1:
            return "mixed"
        if is_consent:
            return "consent"
        if is_intake:
            return "intake"
        if is_privacy:
            return "privacy_notice"
        if is_financial:
            return "financial_policy"

        # As a fallback, treat a document with strong refusal language as refusal
        if any(
            k in all_text
            for k in [
                "i am declining",
                "i decline",
                "refuse treatment",
                "refusal of treatment",
            ]
        ):
            return "refusal"

        return "unknown"

    # ----- Field extractors -----

    def extract_procedure(
        self, sections: List[Dict[str, Any]], source_file: str
    ) -> Dict[str, str]:
        """
        Extract procedure name + description from:
        - 'Recommended Treatment' sections
        - 'Informed Consent for ...' headings
        - 'CONSENT FOR ...' headings
        - or fall back to filename.
        """
        name = ""
        description_parts: List[str] = []

        # First pass: explicit "Recommended Treatment"
        for s in sections:
            title = s.get("section_title", "")
            content = s.get("content", "")

            if self._section_title_contains(title, "recommended treatment"):
                if not name:
                    m = re.search(r"for (.+)", title, re.IGNORECASE)
                    if m:
                        name = self._normalize_text(m.group(1))
                if not name:
                    name = self._infer_procedure_name_from_filename(source_file)

                if content:
                    description_parts.append(content)

        # Second pass: "Informed Consent for ..."
        if not name:
            for s in sections:
                title = s.get("section_title", "")
                if self._section_title_contains(title, "informed consent"):
                    m = re.search(r"for (.+)", title, re.IGNORECASE)
                    if m:
                        candidate = self._normalize_text(m.group(1))
                    else:
                        candidate = self._normalize_text(title)

                    # Avoid titles that are mostly underscores/placeholders
                    if candidate and not all(ch in "_ .-" for ch in candidate):
                        name = candidate
                        break

        # Third pass: "CONSENT FOR ..." (e.g., tooth extraction consent)
        if not name:
            for s in sections:
                title = s.get("section_title", "")
                if self._section_title_contains(title, "consent for"):
                    m = re.search(r"consent for (.+)", title, re.IGNORECASE)
                    if m:
                        candidate = self._normalize_text(m.group(1))
                        if candidate and not all(
                            ch in "_ .-" for ch in candidate
                        ):
                            name = candidate
                            break

        # Final fallback: filename-based inference
        if not name:
            name = self._infer_procedure_name_from_filename(source_file)

        description = (
            self._normalize_text(" ".join(description_parts))
            if description_parts
            else ""
        )

        return {
            "name": name,
            "description": description,
        }

    def _infer_procedure_name_from_filename(self, filename: str) -> str:
        stem = Path(filename).stem
        tokens = stem.replace("_", " ").replace("-", " ")
        tokens = re.sub(
            r"\b(consent|informed|form|dental|patient|intake|general|package)\b",
            "",
            tokens,
            flags=re.IGNORECASE,
        )
        return self._normalize_text(tokens)

    def extract_risks_and_complications(
        self, sections: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract risk/complication sentences from:
        - sections whose titles mention risk/complication
        - any line in any section that contains the words 'risk' or 'complication'
        - for classic risk lists, all numbered lines following a 'there are some risks' lead-in.
        """
        risks: List[str] = []

        for s in sections:
            title = s.get("section_title", "")
            content = s.get("content", "")

            lines = content.splitlines()

            # 1) sections titled with risk/complication
            if self._section_title_contains(title, "risk", "complication"):
                for line in lines:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("--- PAGE"):
                        continue
                    risks.append(self._normalize_text(stripped))

            # 2) any line mentioning 'risk' or 'complication'
            for line in lines:
                low = line.lower()
                if "risk" in low or "complication" in low:
                    stripped = line.strip()
                    if stripped:
                        risks.append(self._normalize_text(stripped))

            # 3) classic pattern: "there are some risks" followed by numbered items
            joined = self._normalize_text(content)
            if "there are some risks" in joined.lower():
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # accept numbered risk lines like "________ 1. ..." or "7. ..."
                    if re.match(r"[_\s]*\d+\.", stripped):
                        risks.append(self._normalize_text(stripped))

        # Deduplicate
        out: List[str] = []
        seen = set()
        for r in risks:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def extract_alternatives(self, sections: List[Dict[str, Any]]) -> List[str]:
        """
        Extract alternatives to treatment from:
        - sections with 'alternatives' in the title
        - content mentioning 'alternative methods of treatment' / 'other options'
        - additional patterns like 'alternatives to ... treatment are'
        """
        alts: List[str] = []

        # 1) sections explicitly about alternatives
        for s in sections:
            title = s.get("section_title", "")
            content = s.get("content", "")
            if self._section_title_contains(
                title, "treatment alternatives", "alternatives"
            ):
                for line in content.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    alts.append(self._normalize_text(stripped))

        # 2) content-driven alternatives
        if not alts:
            for s in sections:
                content = s.get("content", "")
                if self._text_contains(
                    content,
                    "alternative methods of treatment",
                    "other options",
                ):
                    alts.append(self._normalize_text(content))

        # 3) specific phrasing: "alternatives to ... treatment are ..."
        if not alts:
            alt_patterns = [
                r"alternatives to [^\.]* treatment are[^\.]*\.",
                r"alternatives to treatment are[^\.]*\.",
            ]
            for s in sections:
                content = s.get("content", "")
                lowered = content.lower()
                for pat in alt_patterns:
                    for m in re.finditer(pat, lowered):
                        span = content[m.start(): m.end()]
                        alts.append(self._normalize_text(span))

        return alts

    def extract_medications_and_sedation(
        self, sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        mentioned = False
        warnings: List[str] = []

        keywords = [
            "anesthesia",
            "sedation",
            "sedative",
            "nitrous oxide",
            "local anesthetic",
            "local anesthesia",
            "medication",
            "drug reactions",
        ]

        for s in sections:
            content = s.get("content", "")
            title = s.get("section_title", "")

            if self._section_title_contains(
                title, "medication", "drugs", "sedation", "anesthesia"
            ):
                mentioned = True
                for line in content.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    warnings.append(self._normalize_text(stripped))
            elif self._text_contains(content, *keywords):
                mentioned = True
                warnings.append(self._normalize_text(content))

        deduped: List[str] = []
        seen = set()
        for w in warnings:
            if w not in seen:
                seen.add(w)
                deduped.append(w)

        return {
            "mentioned": bool(mentioned),
            "warnings": deduped,
        }

    def extract_patient_responsibilities(
        self, sections: List[Dict[str, Any]]
    ) -> List[str]:
        items: List[str] = []

        for s in sections:
            title = s.get("section_title", "")
            content = s.get("content", "")

            if self._section_title_contains(
                title, "patient responsibilities", "responsibilities"
            ):
                for line in content.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    items.append(self._normalize_text(stripped))
            else:
                if self._section_title_contains(
                    title, "financial policy", "general consent", "consent to perform"
                ):
                    for line in content.splitlines():
                        if self._text_contains(
                            line,
                            "I understand that it is my responsibility",
                            "I understand that it is the patient's responsibility",
                            "responsible for",
                            "I agree to",
                        ):
                            items.append(self._normalize_text(line))

        return items

    def extract_right_to_refuse(
        self, sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        present = False
        text_fragments: List[str] = []

        for s in sections:
            title = s.get("section_title", "")
            content = s.get("content", "")

            if self._section_title_contains(title, "refusal", "treatment refusal"):
                present = True
                text_fragments.append(self._normalize_text(content))

        if not present:
            for s in sections:
                content = s.get("content", "")
                if self._text_contains(
                    content,
                    "I am declining",
                    "I decline",
                    "right to refuse",
                    "refuse treatment",
                    "refusal of treatment",
                ):
                    present = True
                    text_fragments.append(self._normalize_text(content))

        return {
            "present": present,
            "text": " ".join(text_fragments) if text_fragments else "",
        }

    def extract_authorization(
        self, sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Looks for authorization language: 'I authorize...', 'I give consent...', etc.
        Also considers strong permission/consent phrases commonly seen
        in dental/surgical consents.
        """
        present = False
        text_fragments: List[str] = []

        # Track consent title + signature for fallback
        has_consent_title = False
        has_signature_line = False

        # Phrases indicating explicit authorization/consent
        auth_phrases = [
            "i authorize",
            "i hereby authorize",
            "i authorize dr",
            "i give consent",
            "i hereby give consent",
            "i give my consent",
            "i give permission",
            "i consent to",
            "i have agreed to the treatment listed above",
            "i have agreed to the treatment",
            "informed consent to these terms",
            "this permission is for me (or my child/ward)",
            "i wish to proceed with the recommended treatment",
            "i also consent to the administration of local anesthesia",
        ]

        for s in sections:
            title = s.get("section_title", "")
            content = s.get("content", "")
            combined = f"{title}\n{content}"

            if self._section_title_contains(title, "consent for"):
                has_consent_title = True
            if any(
                k in combined.lower()
                for k in ["signature", "sign here", "sign below"]
            ):
                has_signature_line = True

            # Title-driven authorization sections
            if self._section_title_contains(
                title, "authorization", "consent to proceed"
            ):
                present = True
                text_fragments.append(self._normalize_text(content))
                continue

            # Content-driven authorization phrases
            if self._text_contains(combined, *auth_phrases):
                present = True
                text_fragments.append(self._normalize_text(content))

        # Fallback heuristic: consent title + signature lines but no explicit phrase
        if not present and has_consent_title and has_signature_line:
            present = True

        return {
            "present": present,
            "text": " ".join(text_fragments) if text_fragments else "",
        }

    def extract_signature(self, sections: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Signature metadata:
        - required: if any section mentions signature lines or is explicitly marked
          as a signature section in the chunking metadata.
        - guardian_allowed: if patient/parent/guardian or similar appears
          in a document that also requires a signature.
        """
        required = False
        guardian_signal = False

        for s in sections:
            title = s.get("section_title", "")
            content = s.get("content", "")
            combined = f"{title}\n{content}".lower()
            is_sig_section = bool(s.get("is_signature_section"))

            if is_sig_section or any(
                k in combined for k in ["signature", "sign here", "sign below"]
            ):
                required = True

            if any(
                k in combined
                for k in [
                    "patient/parent/guardian",
                    "patient or parent",
                    "parent or legal guardian",
                    "parent / guardian",
                    "guardian",
                    "if patient a minor",
                ]
            ):
                guardian_signal = True

        guardian_allowed = bool(required and guardian_signal)

        return {
            "required": required,
            "guardian_allowed": guardian_allowed,
        }

    # ---------- Main per-file pipeline ----------

    def extract_fields_from_document(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given the chunk JSON for a single document, return a dict
        matching the schema layout (but with concrete values).
        """
        source_file = chunk_data.get("source_file", "")
        sections: List[Dict[str, Any]] = chunk_data.get("sections", [])

        document_kind = self.infer_document_kind(chunk_data)

        if document_kind in ("intake", "financial_policy", "privacy_notice"):
            # Structural only, no consent extraction
            procedure = {"name": "", "description": ""}
            risks_and_complications: List[str] = []
            alternatives: List[str] = []
            meds_sedation = {"mentioned": False, "warnings": []}
            patient_resp = self.extract_patient_responsibilities(sections)
            right_to_refuse = {"present": False, "text": ""}
            authorization = {"present": False, "text": ""}
            signature = self.extract_signature(sections)
        else:
            # Full consent / mixed / refusal / unknown extraction
            procedure = self.extract_procedure(sections, source_file)
            risks_and_complications = self.extract_risks_and_complications(sections)
            alternatives = self.extract_alternatives(sections)
            meds_sedation = self.extract_medications_and_sedation(sections)
            patient_resp = self.extract_patient_responsibilities(sections)
            right_to_refuse = self.extract_right_to_refuse(sections)
            authorization = self.extract_authorization(sections)
            signature = self.extract_signature(sections)

        result = {
            "schema_version": self.schema_meta.get("schema_version", "1.0"),
            "document_type": document_kind,
            "source_file": source_file,
            "procedure": procedure,
            "risks_and_complications": risks_and_complications,
            "alternatives_to_treatment": alternatives,
            "medications_and_sedation": meds_sedation,
            "patient_responsibilities": patient_resp,
            "right_to_refuse": right_to_refuse,
            "authorization": authorization,
            "signature": signature,
        }

        return result

    def process_file(self, chunk_file: Path) -> Dict[str, Any]:
        logger.info(f"Extracting fields from: {chunk_file.name}")

        try:
            chunk_data = self._load_chunk_file(chunk_file)
            fields = self.extract_fields_from_document(chunk_data)
            out_path = self._save_extracted_fields(chunk_file, fields)

            self.stats["files_processed"] += 1
            logger.info(f"✓ Fields written → {out_path.name}")

            return {
                "source_file": str(chunk_file),
                "output_file": str(out_path),
                "status": "success",
            }

        except Exception as e:
            self.stats["files_failed"] += 1
            logger.error(f"✗ Failed extracting from {chunk_file.name}: {e}")

            return {
                "source_file": str(chunk_file),
                "output_file": None,
                "status": "failed",
                "error": str(e),
            }

    def extract_all(self) -> Dict[str, Any]:
        """
        Run extraction over all chunked JSON files in chunks_dir.
        """
        logger.info("=" * 60)
        logger.info("Starting Field Extraction Pipeline")
        logger.info("=" * 60)

        chunk_files = list(self.chunks_dir.glob("*.json"))
        results: List[Dict[str, Any]] = []

        for cf in chunk_files:
            if cf.parent == self.output_dir:
                continue
            results.append(self.process_file(cf))

        logger.info("=" * 60)
        logger.info("Field Extraction Complete")
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files failed: {self.stats['files_failed']}")
        logger.info("=" * 60)

        return {"results": results, "stats": self.stats}


def main():
    extractor = ConsentFieldExtractor()
    extractor.extract_all()


if __name__ == "__main__":
    main()
