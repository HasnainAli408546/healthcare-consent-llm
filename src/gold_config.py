# src/gold_config.py

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = PROJECT_ROOT / "data" / "gold_standard"
PRED_DIR = PROJECT_ROOT / "data" / "processed" / "chunks" / "fields"

# Gold-standard basenames (without .json)
GOLD_DOCS = [
    "general_dental_informed_consent",
    "new_patient_package_intake",
    "oral_surgery_consent",
    "tooth_extraction_consent",
    "dental_implants_informed_consent",
    "informed_consent_virtual_services",
    "pediatric_patient_intake_form",
]

def get_gold_path(name: str) -> Path:
    return GOLD_DIR / f"{name}.json"

def get_pred_path(name: str) -> Path:
    return PRED_DIR / f"{name}.json"
