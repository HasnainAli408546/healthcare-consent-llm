# src/llm_extract.py

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from groq import Groq

# Load .env from project root
load_dotenv()

# ============================================================
# Env & Groq client config
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in the environment.")

client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# Paths & Config
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROMPT_PATH = BASE_DIR / "src" / "prompts" / "extraction_prompt.txt"
TEXT_DIR = BASE_DIR / "data" / "processed" / "text"
OUT_DIR = BASE_DIR / "data" / "processed" / "llm_fields"

SLEEP_BETWEEN_CALLS = 1.0  # rate-limit safety

# ============================================================
# Loaders
# ============================================================

def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def load_document_text(doc_id: str) -> str:
    """
    Load text for a document.
    Supports:
      - single file: <doc_id>.txt
      - chunked files: <doc_id>_chunk_XXX.txt
    """
    single_file = TEXT_DIR / f"{doc_id}.txt"
    if single_file.exists():
        return single_file.read_text(encoding="utf-8")

    chunks = sorted(TEXT_DIR.glob(f"{doc_id}_chunk_*.txt"))
    if not chunks:
        raise FileNotFoundError(f"No text found for document: {doc_id}")

    return "\n\n".join(p.read_text(encoding="utf-8") for p in chunks)


def get_doc_ids() -> List[str]:
    doc_ids = set()

    for p in TEXT_DIR.glob("*.txt"):
        if "_chunk_" not in p.stem:
            doc_ids.add(p.stem)

    for p in TEXT_DIR.glob("*_chunk_*.txt"):
        doc_ids.add(p.stem.split("_chunk_")[0])

    return sorted(doc_ids)

# ============================================================
# LLM Call (Groq + Llama)
# ============================================================

def call_llm(prompt: str) -> Dict[str, Any]:
    """
    Call Groq-hosted Llama model using Chat Completions API.
    Expects the prompt to already include instructions + document text.
    """
    response = client.chat.completions.create(
        model=GROQ_MODEL,  # e.g. "llama-3.1-8b-instant"
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        max_tokens=700,
    )

    content = response.choices[0].message.content

    # Extract JSON substring
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Model returned no JSON object:\n{content}")

    json_str = content[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON:\n{json_str}") from e

# ============================================================
# Normalization (Critical for evaluation)
# ============================================================

def _ensure_list(x) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    return [str(x).strip()]


def _ensure_bool_or_null(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes"):
            return True
        if s in ("false", "no"):
            return False
    return None


def normalize_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw model output into the exact schema used for evaluation.
    Assumes the model may include 'document_type' plus the other fields.
    """
    procedure = obj.get("procedure", {}) or {}
    authorization = obj.get("authorization", {}) or {}
    signature = obj.get("signature", {}) or {}

    return {
        "document_type": obj.get("document_type"),  # keep classification
        "procedure": {
            "name": procedure.get("name"),
        },
        "risks_and_complications": _ensure_list(
            obj.get("risks_and_complications")
        ),
        "alternatives_to_treatment": _ensure_list(
            obj.get("alternatives_to_treatment")
        ),
        "authorization": {
            "present": _ensure_bool_or_null(authorization.get("present"))
        },
        "signature": {
            "required": _ensure_bool_or_null(signature.get("required"))
        },
    }

# ============================================================
# Save Output
# ============================================================

def save_output(doc_id: str, data: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ============================================================
# Main Runner
# ============================================================

def run():
    prompt_template = load_prompt_template()
    doc_ids = get_doc_ids()

    if not doc_ids:
        raise RuntimeError("No documents found to process")

    print("=" * 60)
    print("Running LLM-based healthcare document extraction (Groq + Llama)")
    print("=" * 60)

    for doc_id in doc_ids:
        print(f"\n--- {doc_id} ---")
        try:
            text = load_document_text(doc_id)
            prompt = prompt_template.replace("{{DOCUMENT_TEXT}}", text)

            raw_output = call_llm(prompt)
            normalized_output = normalize_output(raw_output)
            save_output(doc_id, normalized_output)

            print("✓ Extracted successfully")
            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"✗ Failed: {e}")


if __name__ == "__main__":
    run()
