# src/api.py

import os
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# -----------------------------------------------------------------------------
# Env & LLM setup
# -----------------------------------------------------------------------------

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
    return json.loads(content[start:end])


from src.llm_extract import normalize_output  # reuse your normalizer

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    document_text: str


class ExtractResponse(BaseModel):
    document_type: str | None = None
    procedure: dict
    risks_and_complications: list[str]
    alternatives_to_treatment: list[str]
    authorization: dict
    signature: dict


# -----------------------------------------------------------------------------
# FastAPI app & CORS
# -----------------------------------------------------------------------------

app = FastAPI(title="Healthcare Consent Extractor")

# Allow your frontend to call this API (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["http://localhost:3000", "https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Existing JSON endpoint (text already extracted)
# -----------------------------------------------------------------------------

@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    """
    Accepts raw consent document text and returns structured extraction.
    """
    prompt_template = load_prompt_template()
    prompt = prompt_template.replace("{{DOCUMENT_TEXT}}", req.document_text)
    raw = call_llm(prompt)
    normalized = normalize_output(raw)
    return normalized


# -----------------------------------------------------------------------------
# New PDF upload endpoint for the frontend
# -----------------------------------------------------------------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Utility: extract text from a PDF in memory.

    You can swap implementation (PyPDF2, pypdf, pdfplumber, etc.)
    without touching the endpoint signature.
    """
    try:
        import io
        from pypdf import PdfReader  # pip install pypdf
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for PDF extraction. Add 'pypdf' to requirements.txt."
        ) from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n\n".join(parts).strip()


@app.post("/analyze-consent", response_model=ExtractResponse)
async def analyze_consent(file: UploadFile = File(...)):
    """
    Accepts a consent form PDF upload, extracts text,
    runs the same LLM pipeline as /extract, and returns structured data.
    """
    # Basic validation
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read entire PDF into memory
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # 1) Extract raw text from PDF
    try:
        document_text = extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from PDF: {exc}",
        )

    if not document_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the PDF",
        )

    # 2) Reuse your existing LLM extraction pipeline
    prompt_template = load_prompt_template()
    prompt = prompt_template.replace("{{DOCUMENT_TEXT}}", document_text)

    try:
        raw = call_llm(prompt)
        normalized = normalize_output(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM extraction failed: {exc}",
        )

    return normalized
