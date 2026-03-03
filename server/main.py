"""
GABRIEL OpenQDA API Server
--------------------------
A thin FastAPI layer that exposes GABRIEL's qualitative-analysis functions as
HTTP endpoints consumed by the OpenQDA Laravel back-end.

Running this server directly against the GABRIEL source tree (rather than
through a separate wrapper package) means every change to the prompts or
task logic in this repo is reflected immediately without any extra packaging
step.

Endpoints
---------
GET  /status     – health / configuration check
POST /codify     – highlight passages matching a qualitative code
POST /rate       – score passages on natural-language attributes (0–100)
POST /classify   – assign label(s) to each passage
POST /deidentify – replace PII with realistic stand-ins
POST /extract    – pull structured field values from each passage
POST /filter     – boolean screening: which passages satisfy a condition?
"""

import asyncio
import os
import uuid
import tempfile
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status as http_status
from pydantic import BaseModel

load_dotenv()

# ── GABRIEL import ────────────────────────────────────────────────────────────
try:
    import gabriel
    GABRIEL_AVAILABLE = True
except ImportError:
    GABRIEL_AVAILABLE = False

app = FastAPI(title="GABRIEL – OpenQDA Analyze Service")

# ── Gemini / provider routing ─────────────────────────────────────────────────
# Google exposes an OpenAI-compatible REST API.  When a gemini-* model is
# requested we temporarily redirect the OpenAI SDK to Google's endpoint.
# A lock serialises env-var swaps so concurrent Gemini calls don't clobber
# each other's OPENAI_API_KEY / OPENAI_BASE_URL values.

_GEMINI_PREFIX     = "gemini-"
_GOOGLE_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_gemini_lock: asyncio.Lock | None = None


def _get_gemini_lock() -> asyncio.Lock:
    global _gemini_lock
    if _gemini_lock is None:
        _gemini_lock = asyncio.Lock()
    return _gemini_lock


def _is_gemini(model: str) -> bool:
    return model.lower().startswith(_GEMINI_PREFIX)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tmp_dir() -> str:
    """Return a fresh unique temp directory for a single request."""
    path = os.path.join(tempfile.gettempdir(), "gabriel_runs", uuid.uuid4().hex)
    os.makedirs(path, exist_ok=True)
    return path


def _require_gabriel(model: str = "") -> None:
    if not GABRIEL_AVAILABLE:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The gabriel package could not be imported.",
        )
    if _is_gemini(model):
        if not os.getenv("GOOGLE_GEMINI_API_KEY"):
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GOOGLE_GEMINI_API_KEY environment variable is not set.",
            )
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OPENAI_API_KEY environment variable is not set.",
            )


@asynccontextmanager
async def _model_env(model: str):
    """
    For Gemini models: swap OPENAI_API_KEY / OPENAI_BASE_URL to Google's
    OpenAI-compatible endpoint for the duration of the block, then restore.
    A per-process lock prevents concurrent requests from interfering.
    """
    if not _is_gemini(model):
        yield
        return

    async with _get_gemini_lock():
        old_key  = os.environ.get("OPENAI_API_KEY")
        old_base = os.environ.get("OPENAI_BASE_URL")
        try:
            os.environ["OPENAI_API_KEY"]  = os.getenv("GOOGLE_GEMINI_API_KEY", "")
            os.environ["OPENAI_BASE_URL"] = _GOOGLE_OPENAI_URL
            yield
        finally:
            if old_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_key
            if old_base is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = old_base


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    status: str
    gabriel_available: bool
    api_key_configured: bool
    gemini_api_key_configured: bool


class TextItem(BaseModel):
    id: str
    text: str


class CodifyRequest(BaseModel):
    texts: list[TextItem]
    code_name: str
    code_description: str
    model: str = "gpt-4o-mini"
    additional_instructions: str | None = None


class CodifyResult(BaseModel):
    id: str
    passages: list[str]


class RateRequest(BaseModel):
    texts: list[TextItem]
    attributes: dict[str, str]
    model: str = "gpt-4o-mini"


class RateResult(BaseModel):
    id: str
    scores: dict[str, Any]


class ClassifyRequest(BaseModel):
    texts: list[TextItem]
    labels: dict[str, str]
    multi_label: bool = False
    model: str = "gpt-4o-mini"


class ClassifyResult(BaseModel):
    id: str
    labels: list[str]


class DeidentifyRequest(BaseModel):
    texts: list[TextItem]
    model: str = "gpt-4o-mini"


class DeidentifyResult(BaseModel):
    id: str
    anonymized_text: str


class ExtractRequest(BaseModel):
    texts: list[TextItem]
    fields: dict[str, str]
    model: str = "gpt-4o-mini"


class ExtractResult(BaseModel):
    id: str
    values: dict[str, Any]


class FilterRequest(BaseModel):
    texts: list[dict[str, str]]
    condition: str
    model: str = "gpt-4o-mini"


class FilterResult(BaseModel):
    id: str
    name: str | None = None
    meets_condition: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/status", response_model=ServiceStatus)
def get_status():
    return ServiceStatus(
        status="running",
        gabriel_available=GABRIEL_AVAILABLE,
        api_key_configured=bool(os.getenv("OPENAI_API_KEY")),
        gemini_api_key_configured=bool(os.getenv("GOOGLE_GEMINI_API_KEY")),
    )


@app.post("/codify", response_model=list[CodifyResult])
async def codify(request: CodifyRequest):
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])
    save_dir = _tmp_dir()

    kwargs: dict = dict(
        column_name="text",
        categories={request.code_name: request.code_description},
        save_dir=save_dir,
        model=request.model,
        reset_files=True,
    )
    if request.additional_instructions:
        kwargs["additional_instructions"] = request.additional_instructions

    async with _model_env(request.model):
        result_df = await gabriel.codify(df, **kwargs)

    output: list[CodifyResult] = []
    for _, row in result_df.iterrows():
        passages = row.get(request.code_name, []) or []
        if isinstance(passages, str):
            passages = [passages]
        output.append(CodifyResult(id=str(row["id"]), passages=list(passages)))
    return output


@app.post("/rate", response_model=list[RateResult])
async def rate(request: RateRequest):
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])
    save_dir = _tmp_dir()

    async with _model_env(request.model):
        result_df = await gabriel.rate(
            df,
            column_name="text",
            attributes=request.attributes,
            save_dir=save_dir,
            model=request.model,
            reset_files=True,
        )

    attr_cols = list(request.attributes.keys())
    return [
        RateResult(id=str(row["id"]), scores={a: row.get(a) for a in attr_cols})
        for _, row in result_df.iterrows()
    ]


@app.post("/classify", response_model=list[ClassifyResult])
async def classify(request: ClassifyRequest):
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])
    save_dir = _tmp_dir()

    async with _model_env(request.model):
        result_df = await gabriel.classify(
            df,
            column_name="text",
            labels=request.labels,
            save_dir=save_dir,
            model=request.model,
            reset_files=True,
        )

    label_cols = list(request.labels.keys())
    return [
        ClassifyResult(
            id=str(row["id"]),
            labels=[lbl for lbl in label_cols if row.get(lbl)],
        )
        for _, row in result_df.iterrows()
    ]


@app.post("/deidentify", response_model=list[DeidentifyResult])
async def deidentify(request: DeidentifyRequest):
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])
    save_dir = _tmp_dir()

    async with _model_env(request.model):
        result_df = await gabriel.deidentify(
            df,
            column_name="text",
            save_dir=save_dir,
            model=request.model,
            reset_files=True,
        )

    anon_col = "text_deidentified" if "text_deidentified" in result_df.columns else "text"
    return [
        DeidentifyResult(id=str(row["id"]), anonymized_text=str(row.get(anon_col, "")))
        for _, row in result_df.iterrows()
    ]


@app.post("/extract", response_model=list[ExtractResult])
async def extract(request: ExtractRequest):
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])
    save_dir = _tmp_dir()

    async with _model_env(request.model):
        result_df = await gabriel.extract(
            df,
            column_name="text",
            attributes=request.fields,
            save_dir=save_dir,
            model=request.model,
            reset_files=True,
        )

    field_cols = list(request.fields.keys())
    return [
        ExtractResult(id=str(row["id"]), values={f: row.get(f) for f in field_cols})
        for _, row in result_df.iterrows()
    ]


@app.post("/filter", response_model=list[FilterResult])
async def filter_sources(request: FilterRequest):
    _require_gabriel(request.model)
    df = pd.DataFrame([
        {"id": t["id"], "name": t.get("name", ""), "text": t["text"]}
        for t in request.texts
    ])
    save_dir = _tmp_dir()

    async with _model_env(request.model):
        result_df = await gabriel.filter(
            df,
            column_name="text",
            condition=request.condition,
            save_dir=save_dir,
            model=request.model,
            reset_files=True,
        )

    return [
        FilterResult(
            id=str(row["id"]),
            name=str(row.get("name", "")) or None,
            meets_condition=bool(row.get("meets_condition", False)),
        )
        for _, row in result_df.iterrows()
    ]
