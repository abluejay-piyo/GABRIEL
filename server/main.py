"""
GABRIEL plugin API server (FastAPI)

This service is intentionally plugin-owned. OpenQDA frontend calls /api/gabriel/*,
and Traefik routes those requests to this container without changing OpenQDA core.
"""

import asyncio
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status as http_status
from pydantic import BaseModel, ConfigDict

load_dotenv()

try:
    import gabriel

    GABRIEL_AVAILABLE = True
except ImportError:
    GABRIEL_AVAILABLE = False

app = FastAPI(title="GABRIEL Plugin API")

DEFAULT_N_PARALLELS = max(1, int(os.getenv("GABRIEL_API_N_PARALLELS", "24")))
DEFAULT_TIMEOUT_FACTOR = float(os.getenv("GABRIEL_API_TIMEOUT_FACTOR", "3.0"))
DEFAULT_MAX_TIMEOUT = float(os.getenv("GABRIEL_API_MAX_TIMEOUT", "180"))

_GEMINI_PREFIX = "gemini-"
_GOOGLE_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_gemini_lock: asyncio.Lock | None = None


def _get_gemini_lock() -> asyncio.Lock:
    global _gemini_lock
    if _gemini_lock is None:
        _gemini_lock = asyncio.Lock()
    return _gemini_lock


def _is_gemini(model: str) -> bool:
    return model.lower().startswith(_GEMINI_PREFIX)


def _tmp_dir() -> str:
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
    elif not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY environment variable is not set.",
        )


@asynccontextmanager
async def _model_env(model: str):
    if not _is_gemini(model):
        yield
        return

    async with _get_gemini_lock():
        old_key = os.environ.get("OPENAI_API_KEY")
        old_base = os.environ.get("OPENAI_BASE_URL")
        try:
            os.environ["OPENAI_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY", "")
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


class ServiceStatus(BaseModel):
    status: str
    gabriel_available: bool
    api_key_configured: bool
    gemini_api_key_configured: bool


class TextItem(BaseModel):
    id: str
    text: str
    name: str | None = None


class RuntimeTuningRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class CodifyRequest(RuntimeTuningRequest):
    texts: list[TextItem]
    code_name: str
    code_description: str | None = None
    model: str = "gpt-5.4-mini"
    instructions: str | None = None
    additional_instructions: str | None = None
    n_rounds: int | None = None


class CodifyResult(BaseModel):
    id: str
    passages: list[str]


class RateRequest(RuntimeTuningRequest):
    texts: list[TextItem]
    attributes: dict[str, str]
    model: str = "gpt-5.4-mini"
    n_runs: int | None = None


class RateResult(BaseModel):
    id: str
    scores: dict[str, Any]


class ClassifyRequest(RuntimeTuningRequest):
    texts: list[TextItem]
    labels: dict[str, str]
    multi_label: bool = False
    model: str = "gpt-5.4-mini"
    n_runs: int | None = None


class ClassifyResult(BaseModel):
    id: str
    labels: list[str]


class DeidentifyRequest(RuntimeTuningRequest):
    texts: list[TextItem]
    model: str = "gpt-5.4-mini"


class DeidentifyResult(BaseModel):
    id: str
    anonymized_text: str


class ExtractRequest(RuntimeTuningRequest):
    texts: list[TextItem]
    fields: dict[str, str]
    model: str = "gpt-5.4-mini"
    n_runs: int | None = None


class ExtractResult(BaseModel):
    id: str
    values: dict[str, Any]


class FilterRequest(RuntimeTuningRequest):
    texts: list[TextItem]
    condition: str
    model: str = "gpt-5.4-nano"
    n_runs: int | None = None


class FilterResult(BaseModel):
    id: str
    name: str | None = None
    meets_condition: bool


def _runtime_overrides(payload: RuntimeTuningRequest, reserved_keys: set[str]) -> dict[str, Any]:
    values = payload.model_dump(exclude_none=True)
    return {
        key: value
        for key, value in values.items()
        if key not in reserved_keys
    }


def _apply_safe_runtime_defaults(kwargs: dict[str, Any]) -> None:
    kwargs.setdefault("n_parallels", DEFAULT_N_PARALLELS)
    kwargs.setdefault("timeout_factor", DEFAULT_TIMEOUT_FACTOR)
    kwargs.setdefault("max_timeout", DEFAULT_MAX_TIMEOUT)


@app.get("/status", response_model=ServiceStatus)
def get_status() -> ServiceStatus:
    return ServiceStatus(
        status="running",
        gabriel_available=GABRIEL_AVAILABLE,
        api_key_configured=bool(os.getenv("OPENAI_API_KEY")),
        gemini_api_key_configured=bool(os.getenv("GOOGLE_GEMINI_API_KEY")),
    )


@app.post("/codify", response_model=list[CodifyResult])
async def codify(request: CodifyRequest) -> list[CodifyResult]:
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])

    kwargs: dict[str, Any] = dict(
        column_name="text",
        categories={request.code_name: request.code_description or ""},
        save_dir=_tmp_dir(),
        model=request.model,
        reset_files=True,
    )
    instructions = request.instructions or request.additional_instructions
    if instructions:
        kwargs["additional_instructions"] = instructions
    if request.n_rounds is not None:
        kwargs["n_rounds"] = request.n_rounds
    kwargs.update(
        _runtime_overrides(
            request,
            {
                "texts",
                "code_name",
                "code_description",
                "model",
                "instructions",
                "additional_instructions",
                "n_rounds",
            },
        )
    )
    _apply_safe_runtime_defaults(kwargs)

    async with _model_env(request.model):
        result_df = await gabriel.codify(df, **kwargs)

    out: list[CodifyResult] = []
    for _, row in result_df.iterrows():
        passages = row.get(request.code_name, []) or []
        if isinstance(passages, str):
            passages = [passages]
        out.append(CodifyResult(id=str(row.get("id")), passages=list(passages)))
    return out


@app.post("/rate", response_model=list[RateResult])
async def rate(request: RateRequest) -> list[RateResult]:
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])

    kwargs: dict[str, Any] = dict(
        column_name="text",
        attributes=request.attributes,
        save_dir=_tmp_dir(),
        model=request.model,
        reset_files=True,
    )
    if request.n_runs is not None:
        kwargs["n_runs"] = request.n_runs
    kwargs.update(
        _runtime_overrides(
            request,
            {"texts", "attributes", "model", "n_runs"},
        )
    )
    _apply_safe_runtime_defaults(kwargs)

    async with _model_env(request.model):
        result_df = await gabriel.rate(df, **kwargs)

    out: list[RateResult] = []
    for _, row in result_df.iterrows():
        rid = str(row.get("id"))
        scores = {k: row.get(k) for k in request.attributes.keys()}
        out.append(RateResult(id=rid, scores=scores))
    return out


@app.post("/classify", response_model=list[ClassifyResult])
async def classify(request: ClassifyRequest) -> list[ClassifyResult]:
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])

    kwargs: dict[str, Any] = dict(
        column_name="text",
        labels=request.labels,
        save_dir=_tmp_dir(),
        model=request.model,
        reset_files=True,
    )
    if request.n_runs is not None:
        kwargs["n_runs"] = request.n_runs
    kwargs.update(
        _runtime_overrides(
            request,
            {"texts", "labels", "multi_label", "model", "n_runs"},
        )
    )
    _apply_safe_runtime_defaults(kwargs)

    async with _model_env(request.model):
        result_df = await gabriel.classify(df, **kwargs)

    label_names = list(request.labels.keys())
    out: list[ClassifyResult] = []
    for _, row in result_df.iterrows():
        rid = str(row.get("id"))
        labels: list[str] = []
        for label in label_names:
            val = row.get(label)
            if isinstance(val, (bool, int, float)) and bool(val):
                labels.append(label)
        if not request.multi_label and len(labels) > 1:
            labels = labels[:1]
        out.append(ClassifyResult(id=rid, labels=labels))
    return out


@app.post("/deidentify", response_model=list[DeidentifyResult])
async def deidentify(request: DeidentifyRequest) -> list[DeidentifyResult]:
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])

    kwargs: dict[str, Any] = dict(
        column_name="text",
        save_dir=_tmp_dir(),
        model=request.model,
        reset_files=True,
    )
    kwargs.update(
        _runtime_overrides(
            request,
            {"texts", "model"},
        )
    )
    _apply_safe_runtime_defaults(kwargs)

    async with _model_env(request.model):
        result_df = await gabriel.deidentify(df, **kwargs)

    out: list[DeidentifyResult] = []
    for _, row in result_df.iterrows():
        rid = str(row.get("id"))
        out.append(DeidentifyResult(id=rid, anonymized_text=str(row.get("deidentified_text", ""))))
    return out


@app.post("/extract", response_model=list[ExtractResult])
async def extract(request: ExtractRequest) -> list[ExtractResult]:
    _require_gabriel(request.model)
    df = pd.DataFrame([{"id": t.id, "text": t.text} for t in request.texts])

    kwargs: dict[str, Any] = dict(
        column_name="text",
        attributes=request.fields,
        save_dir=_tmp_dir(),
        model=request.model,
        reset_files=True,
    )
    if request.n_runs is not None:
        kwargs["n_runs"] = request.n_runs
    kwargs.update(
        _runtime_overrides(
            request,
            {"texts", "fields", "model", "n_runs"},
        )
    )
    _apply_safe_runtime_defaults(kwargs)

    async with _model_env(request.model):
        result_df = await gabriel.extract(df, **kwargs)

    out: list[ExtractResult] = []
    for _, row in result_df.iterrows():
        rid = str(row.get("id"))
        values = {k: row.get(k) for k in request.fields.keys()}
        out.append(ExtractResult(id=rid, values=values))
    return out


@app.post("/filter", response_model=list[FilterResult])
async def filter_items(request: FilterRequest) -> list[FilterResult]:
    _require_gabriel(request.model)
    df = pd.DataFrame([
        {"id": t.id, "text": t.text, "name": t.name if t.name is not None else t.text}
        for t in request.texts
    ])

    kwargs: dict[str, Any] = dict(
        column_name="text",
        condition=request.condition,
        save_dir=_tmp_dir(),
        model=request.model,
        reset_files=True,
        shuffle=False,
    )
    if request.n_runs is not None:
        kwargs["n_runs"] = request.n_runs
    kwargs.update(
        _runtime_overrides(
            request,
            {"texts", "condition", "model", "n_runs"},
        )
    )
    _apply_safe_runtime_defaults(kwargs)

    async with _model_env(request.model):
        result_df = await gabriel.filter(df, **kwargs)

    out: list[FilterResult] = []
    for _, row in result_df.iterrows():
        out.append(
            FilterResult(
                id=str(row.get("id")),
                name=row.get("name"),
                meets_condition=bool(row.get("meets_condition", False)),
            )
        )
    return out
