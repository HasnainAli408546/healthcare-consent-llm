# src/eval_gold_fields.py

import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import logging

from .gold_config import GOLD_DOCS, get_gold_path, get_pred_path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _normalize_str(s: str) -> str:
    """Lowercase + collapse whitespace for robust matching."""
    return " ".join((s or "").lower().split())


def _set_from_list(lst: List[str]) -> Set[str]:
    """Turn list of strings into a normalized set."""
    out: Set[str] = set()
    for x in lst or []:
        norm = _normalize_str(x)
        if norm:
            out.add(norm)
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_prf(gold_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0  # trivial perfect
    if not pred_set and gold_set:
        return 0.0, 0.0, 0.0
    if pred_set and not gold_set:
        # all predicted are false positives
        return 0.0, 0.0, 0.0

    inter = gold_set & pred_set
    p = len(inter) / len(pred_set) if pred_set else 0.0
    r = len(inter) / len(gold_set) if gold_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def eval_gold_fields() -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Evaluating extractor against GOLD STANDARD docs")
    logger.info("=" * 60)

    # Aggregates
    proc_total = 0
    proc_correct = 0

    risks_docs = 0
    risks_p_sum = risks_r_sum = risks_f1_sum = 0.0

    alts_docs = 0
    alts_p_sum = alts_r_sum = alts_f1_sum = 0.0

    auth_total = 0
    auth_correct = 0

    sig_total = 0
    sig_correct = 0

    per_doc_results: List[Dict[str, Any]] = []

    for name in GOLD_DOCS:
        gold_path = get_gold_path(name)
        pred_path = get_pred_path(name)

        if not gold_path.exists():
            logger.warning(f"[SKIP] Gold file missing: {gold_path}")
            continue
        if not pred_path.exists():
            logger.warning(f"[SKIP] Pred file missing: {pred_path}")
            continue

        gold = _load_json(gold_path)
        pred = _load_json(pred_path)

        logger.info(f"\n--- {name} ---")

        doc_result: Dict[str, Any] = {"name": name}

        # 1) Procedure name
        gold_proc_name = _normalize_str(
            (gold.get("procedure") or {}).get("name", "")
        )
        pred_proc_name = _normalize_str(
            (pred.get("procedure") or {}).get("name", "")
        )

        if gold_proc_name:
            proc_total += 1
            correct = int(gold_proc_name == pred_proc_name)
            proc_correct += correct
            doc_result["procedure_match"] = bool(correct)
            if not correct:
                logger.info(f"  procedure.name mismatch:")
                logger.info(f"    gold: {gold_proc_name}")
                logger.info(f"    pred: {pred_proc_name}")

        # 2) Risks and complications
        gold_risks = _set_from_list(gold.get("risks_and_complications", []))
        pred_risks = _set_from_list(pred.get("risks_and_complications", []))

        if gold_risks or pred_risks:
            p, r, f1 = compute_prf(gold_risks, pred_risks)
            risks_docs += 1
            risks_p_sum += p
            risks_r_sum += r
            risks_f1_sum += f1
            doc_result["risks_p"] = p
            doc_result["risks_r"] = r
            doc_result["risks_f1"] = f1

            if p < 1.0 or r < 1.0:
                only_gold = sorted(gold_risks - pred_risks)
                only_pred = sorted(pred_risks - gold_risks)
                if only_gold:
                    logger.info("  risks: missed (in gold, not in pred):")
                    for x in only_gold:
                        logger.info(f"    - {x}")
                if only_pred:
                    logger.info("  risks: extra (in pred, not in gold):")
                    for x in only_pred:
                        logger.info(f"    - {x}")

        # 3) Alternatives (if present in gold)
        gold_alts = _set_from_list(gold.get("alternatives_to_treatment", []))
        pred_alts = _set_from_list(pred.get("alternatives_to_treatment", []))

        if gold_alts or pred_alts:
            p, r, f1 = compute_prf(gold_alts, pred_alts)
            alts_docs += 1
            alts_p_sum += p
            alts_r_sum += r
            alts_f1_sum += f1
            doc_result["alts_p"] = p
            doc_result["alts_r"] = r
            doc_result["alts_f1"] = f1

            if p < 1.0 or r < 1.0:
                only_gold = sorted(gold_alts - pred_alts)
                only_pred = sorted(pred_alts - gold_alts)
                if only_gold:
                    logger.info("  alternatives: missed (in gold, not in pred):")
                    for x in only_gold:
                        logger.info(f"    - {x}")
                if only_pred:
                    logger.info("  alternatives: extra (in pred, not in gold):")
                    for x in only_pred:
                        logger.info(f"    - {x}")

        # 4) Authorization present (bool)
        # Map your gold structure into a boolean "present" when possible.
        # For general informed consent:
        gold_auth = gold.get("authorization") or {}
        gold_auth_present = gold_auth.get("present", None)

        # Some of your gold docs use "authorization_scope" instead:
        if gold_auth_present is None and "authorization_scope" in gold:
            gold_auth_present = (gold["authorization_scope"] or {}).get(
                "present", None
            )

        pred_auth_present = (pred.get("authorization") or {}).get(
            "present", None
        )

        if gold_auth_present is not None:
            auth_total += 1
            if gold_auth_present == pred_auth_present:
                auth_correct += 1
                doc_result["auth_match"] = True
            else:
                doc_result["auth_match"] = False
                logger.info(
                    f"  authorization.present mismatch: gold={gold_auth_present}, pred={pred_auth_present}"
                )

        # 5) Signature required (bool)
        gold_sig_required = (gold.get("signature") or {}).get("required", None)
        pred_sig_required = (pred.get("signature") or {}).get("required", None)

        if gold_sig_required is not None:
            sig_total += 1
            if gold_sig_required == pred_sig_required:
                sig_correct += 1
                doc_result["signature_match"] = True
            else:
                doc_result["signature_match"] = False
                logger.info(
                    f"  signature.required mismatch: gold={gold_sig_required}, pred={pred_sig_required}"
                )

        per_doc_results.append(doc_result)

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE GOLD vs EXTRACTOR METRICS")
    logger.info("=" * 60)

    if proc_total:
        logger.info(
            f"procedure.name accuracy: {proc_correct}/{proc_total} "
            f"({proc_correct / proc_total * 100.0:.1f}%)"
        )

    if risks_docs:
        logger.info(
            f"risks_and_complications (macro-avg over {risks_docs} docs): "
            f"P={risks_p_sum/risks_docs:.3f}, "
            f"R={risks_r_sum/risks_docs:.3f}, "
            f"F1={risks_f1_sum/risks_docs:.3f}"
        )

    if alts_docs:
        logger.info(
            f"alternatives_to_treatment (macro-avg over {alts_docs} docs): "
            f"P={alts_p_sum/alts_docs:.3f}, "
            f"R={alts_r_sum/alts_docs:.3f}, "
            f"F1={alts_f1_sum/alts_docs:.3f}"
        )

    if auth_total:
        logger.info(
            f"authorization.present accuracy: {auth_correct}/{auth_total} "
            f"({auth_correct / auth_total * 100.0:.1f}%)"
        )

    if sig_total:
        logger.info(
            f"signature.required accuracy: {sig_correct}/{sig_total} "
            f"({sig_correct / sig_total * 100.0:.1f}%)"
        )

    logger.info("=" * 60)

    return {
        "per_doc": per_doc_results,
        "aggregates": {
            "procedure": {"total": proc_total, "correct": proc_correct},
            "risks": {
                "docs": risks_docs,
                "P_macro": risks_p_sum / risks_docs if risks_docs else 0.0,
                "R_macro": risks_r_sum / risks_docs if risks_docs else 0.0,
                "F1_macro": risks_f1_sum / risks_docs if risks_docs else 0.0,
            },
            "alternatives": {
                "docs": alts_docs,
                "P_macro": alts_p_sum / alts_docs if alts_docs else 0.0,
                "R_macro": alts_r_sum / alts_docs if alts_docs else 0.0,
                "F1_macro": alts_f1_sum / alts_docs if alts_docs else 0.0,
            },
            "authorization": {
                "total": auth_total,
                "correct": auth_correct,
            },
            "signature": {
                "total": sig_total,
                "correct": sig_correct,
            },
        },
    }


if __name__ == "__main__":
    eval_gold_fields()
