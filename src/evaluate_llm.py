# src/evaluate_llm.py

"""
Evaluation for Groq + Llama extraction outputs.

Reads JSON files from data/processed/llm_fields and reports:
- Missing / empty core fields
- Consent-specific expectations (procedure, risks, authorization, signature)
"""

import json
from pathlib import Path
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Core fields to always track
FIELDS_TO_CHECK = [
    "procedure",
    "risks_and_complications",
    "alternatives_to_treatment",
    "authorization",
    "signature",
]


def is_empty_value(value: Any) -> bool:
    """Return True if a value should be treated as 'empty' for evaluation."""
    if value is None:
        return True
    if value == "":
        return True
    if value == []:
        return True
    if value == {}:
        return True
    return False


def evaluate_fields(
    fields_dir: str = "data/processed/llm_fields",
) -> Dict[str, Any]:
    fields_path = Path(fields_dir)
    files = list(fields_path.glob("*.json"))

    if not files:
        logger.info(f"No field files found in {fields_dir}")
        return {}

    stats: Dict[str, Any] = {
        "documents": 0,
        "missing_fields": {f: 0 for f in FIELDS_TO_CHECK},
        "empty_fields": {f: 0 for f in FIELDS_TO_CHECK},
        # consent-specific checks
        "consent_docs": 0,
        "consent_expectations": {
            "procedure.name_missing": 0,
            "risks_empty": 0,
            "authorization_not_present_true": 0,
            "signature_required_false": 0,
        },
        # refusal docs count (for visibility, but not used in expectations)
        "refusal_docs": 0,
    }

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        stats["documents"] += 1

        # 1) Generic missing/empty tracking (for all docs)
        for field in FIELDS_TO_CHECK:
            if field not in data:
                stats["missing_fields"][field] += 1
                continue

            value = data[field]
            if is_empty_value(value):
                stats["empty_fields"][field] += 1

        # 2) Consent-specific expectations only for consent/mixed docs
        doc_type = data.get("document_type", "consent")

        if doc_type == "refusal":
            # Count refusal docs just for transparency; no consent expectations apply
            stats["refusal_docs"] += 1
            continue

        if doc_type not in ("consent", "mixed"):
            # Intake, privacy_notice, financial_policy, unknown: skip consent expectations
            continue

        stats["consent_docs"] += 1

        # procedure.name should be non-empty
        proc = data.get("procedure") or {}
        if not proc.get("name"):
            stats["consent_expectations"]["procedure.name_missing"] += 1

        # risks_and_complications should be non-empty list
        risks = data.get("risks_and_complications")
        if not risks:
            stats["consent_expectations"]["risks_empty"] += 1

        # authorization.present should be True
        auth = data.get("authorization") or {}
        if auth.get("present") is not True:
            stats["consent_expectations"]["authorization_not_present_true"] += 1

        # signature.required should be True
        sig = data.get("signature") or {}
        if sig.get("required") is not True:
            stats["consent_expectations"]["signature_required_false"] += 1

    # --------- Logging report ---------
    docs = stats["documents"]
    logger.info("=" * 60)
    logger.info("Field Extraction Evaluation Results (Groq + Llama outputs)")
    logger.info("=" * 60)
    logger.info(f"Documents evaluated: {docs}\n")

    logger.info("Field coverage / emptiness (all documents):")
    for field in FIELDS_TO_CHECK:
        missing = stats["missing_fields"][field]
        empty = stats["empty_fields"][field]
        missing_pct = (missing / docs) * 100.0 if docs else 0.0
        empty_pct = (empty / docs) * 100.0 if docs else 0.0

        logger.info(
            f"- {field}: "
            f"missing={missing} ({missing_pct:.1f}%), "
            f"empty={empty} ({empty_pct:.1f}%)"
        )

    ce = stats["consent_expectations"]
    consent_docs = stats["consent_docs"] or 1  # avoid div-by-zero

    logger.info(
        f"\nConsent-specific expectations "
        f"(only consent/mixed docs, n={stats['consent_docs']}):"
    )
    logger.info(
        f"- procedure.name missing in {ce['procedure.name_missing']} / {consent_docs} "
        f"({ce['procedure.name_missing'] / consent_docs * 100.0:.1f}%)"
    )
    logger.info(
        f"- risks_and_complications empty in {ce['risks_empty']} / {consent_docs} "
        f"({ce['risks_empty'] / consent_docs * 100.0:.1f}%)"
    )
    logger.info(
        f"- authorization.present != True in "
        f"{ce['authorization_not_present_true']} / {consent_docs} "
        f"({ce['authorization_not_present_true'] / consent_docs * 100.0:.1f}%)"
    )
    logger.info(
        f"- signature.required != True in "
        f"{ce['signature_required_false']} / {consent_docs} "
        f"({ce['signature_required_false'] / consent_docs * 100.0:.1f}%)"
    )

    logger.info(f"\nRefusal documents (n={stats['refusal_docs']}):")
    logger.info("  - Refusal docs are excluded from consent expectations.")
    logger.info("=" * 60)

    return stats


if __name__ == "__main__":
    evaluate_fields()
