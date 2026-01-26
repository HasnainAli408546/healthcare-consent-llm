# src/api.py

import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set")

client = Groq(api_key=GROQ_API_KEY)

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "prompts" / "extraction_prompt.txt"


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def call_llm(prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700,
    )
    content = resp.choices[0].message.content
    start = content.find("{")
    end = content.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Model returned no JSON object:\n{content}")
    import json
    return json.loads(content[start:end])


from src.llm_extract import normalize_output  # reuse your normalizer


class ExtractRequest(BaseModel):
    document_text: str


class ExtractResponse(BaseModel):
    document_type: str | None = None
    procedure: dict
    risks_and_complications: list[str]
    alternatives_to_treatment: list[str]
    authorization: dict
    signature: dict


app = FastAPI(title="Healthcare Consent Extractor")


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    prompt_template = load_prompt_template()
    prompt = prompt_template.replace("{{DOCUMENT_TEXT}}", req.document_text)
    raw = call_llm(prompt)
    normalized = normalize_output(raw)
    return normalized
