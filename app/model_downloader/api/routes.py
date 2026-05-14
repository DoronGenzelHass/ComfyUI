"""Aiohttp routes for the server-side model download subsystem.

Endpoint surface (all POST, all under ``/api/``, all kebab-case):

  - ``/api/models-availability-status``     — bulk status query.
  - ``/api/missing-models-metadata``        — bulk file-size / gated probe.
  - ``/api/download-models``                — start a batch of downloads.
  - ``/api/cancel-model-download-session``  — cancel a single in-flight one.

The contract is intentionally narrow: only model_ids of the form
``<directory>/<filename>`` (validated via ``app.model_downloader.paths``)
are accepted, and only URLs on the same allowlist the frontend already
uses (HuggingFace, Civitai, localhost) can be fetched. Both are required
to keep the server out of the SSRF business for this feature.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from aiohttp import web
from pydantic import BaseModel, ValidationError

from app.model_downloader.allowlist import is_url_allowed
from app.model_downloader.download_server import (
    DOWNLOAD_SERVER,
    DownloadSession,
)
from app.model_downloader.downloader import schedule_batch
from app.model_downloader.gated_detection import probe_url
from app.model_downloader.paths import (
    InvalidModelId,
    parse_model_id,
    resolve_existing,
)
from app.model_downloader.api import schemas_in, schemas_out

ROUTES = web.RouteTableDef()


def register_routes(app: web.Application) -> None:
    """Wire the model-downloader routes into the running aiohttp app.

    Called once from ``server.py`` during ``PromptServer`` startup.
    """
    app.add_routes(ROUTES)


# ----- response helpers (same envelope as app/assets/api/routes.py) -----


def _error(status: int, code: str, message: str, details: dict | None = None) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message, "details": details or {}}},
        status=status,
    )


def _validation_error(code: str, ve: ValidationError) -> web.Response:
    return _error(400, code, "Validation failed.", {"errors": json.loads(ve.json())})


def _ok(payload: BaseModel, status: int = 200) -> web.Response:
    return web.json_response(
        payload.model_dump(mode="json", exclude_none=False),
        status=status,
    )


async def _parse_body(request: web.Request, model: type[BaseModel]) -> Any:
    """Parse a JSON body into a pydantic model or raise a 400 response."""
    try:
        raw = await request.json()
    except json.JSONDecodeError:
        return _error(400, "INVALID_JSON", "Request body must be valid JSON.")
    try:
        return model.model_validate(raw)
    except ValidationError as ve:
        return _validation_error("INVALID_BODY", ve)


# ----- 1. availability status -----


@ROUTES.post("/api/models-availability-status")
async def models_availability_status(request: web.Request) -> web.Response:
    parsed = await _parse_body(request, schemas_in.AvailabilityStatusRequest)
    if isinstance(parsed, web.Response):
        return parsed

    available: list[str] = []
    missing: list[str] = []
    downloading: list[schemas_out.DownloadingEntry] = []

    for model_id in parsed.model_ids:
        try:
            parse_model_id(model_id)
        except InvalidModelId:
            # Ill-formed identifier: treat as "not on disk, not downloading"
            # so the frontend can surface it as missing without us 400-ing
            # the whole batch.
            missing.append(model_id)
            continue

        active = DOWNLOAD_SERVER.get(model_id)
        if active is not None:
            downloading.append(schemas_out.DownloadingEntry(
                model_id=model_id,
                progress=active.progress,
                bytes_downloaded=active.bytes_downloaded,
                total_bytes=active.total_bytes,
            ))
            continue

        if resolve_existing(model_id) is not None:
            available.append(model_id)
        else:
            missing.append(model_id)

    return _ok(schemas_out.AvailabilityStatusResponse(
        available=available,
        missing=missing,
        downloading=downloading,
    ))


# ----- 2. missing-models metadata -----


@ROUTES.post("/api/missing-models-metadata")
async def missing_models_metadata(request: web.Request) -> web.Response:
    parsed = await _parse_body(request, schemas_in.MissingModelsMetadataRequest)
    if isinstance(parsed, web.Response):
        return parsed

    # Probe each URL concurrently. Each probe handles its own errors and
    # returns a result-shaped object, so gather() can use return_exceptions=False
    # without us having to write per-task try/except wrappers.
    items = list(parsed.models.items())
    results = await asyncio.gather(*(probe_url(url) for _, url in items))

    metadata: dict[str, schemas_out.MissingModelMetadataEntry] = {}
    for (model_id, _url), probe in zip(items, results):
        metadata[model_id] = schemas_out.MissingModelMetadataEntry(
            file_size=probe.file_size,
            is_gated=probe.is_gated,
        )

    return _ok(schemas_out.MissingModelsMetadataResponse(models=metadata))


# ----- 3. start downloads -----


@ROUTES.post("/api/download-models")
async def download_models(request: web.Request) -> web.Response:
    parsed = await _parse_body(request, schemas_in.DownloadModelsRequest)
    if isinstance(parsed, web.Response):
        return parsed

    if not parsed.models:
        return _error(400, "EMPTY_REQUEST", "No models supplied.")

    # ----- precondition pass: validate everything BEFORE registering anything -----
    # Atomic semantics: if any model fails any precondition (invalid id,
    # not allow-listed URL, already on disk, already downloading, or gated),
    # the entire request fails and no state is changed.
    requested = list(parsed.models.items())

    for model_id, url in requested:
        try:
            parse_model_id(model_id)
        except InvalidModelId as e:
            return _error(400, "INVALID_MODEL_ID", str(e),
                          {"model_id": model_id})

        if not is_url_allowed(url):
            return _error(
                400, "URL_NOT_ALLOWED",
                "Server-side downloads only accept HuggingFace, Civitai, "
                "or localhost URLs ending in a known model extension.",
                {"model_id": model_id, "url": url},
            )

        if resolve_existing(model_id) is not None:
            return _error(409, "ALREADY_AVAILABLE",
                          f"Model already exists on disk: {model_id}",
                          {"model_id": model_id})

        if DOWNLOAD_SERVER.is_downloading(model_id):
            return _error(409, "ALREADY_DOWNLOADING",
                          f"A download for {model_id} is already in progress.",
                          {"model_id": model_id})

    # Gated check happens last because it's the only one that talks to
    # the network. Concurrent HEADs.
    probes = await asyncio.gather(*(probe_url(url) for _, url in requested))
    for (model_id, url), probe in zip(requested, probes):
        if probe.is_gated:
            return _error(
                400, "MODEL_GATED",
                f"Model {model_id} is gated by its host; please acquire it manually.",
                {"model_id": model_id, "url": url},
            )

    # ----- registration pass: try_register is atomic per model_id -----
    # Defensive: another request might have raced past our pre-check
    # between the loop above and here. try_register handles that.
    sessions: list[DownloadSession] = []
    for model_id, url in requested:
        session = DOWNLOAD_SERVER.try_register(model_id, url)
        if session is None:
            # Race: someone else got in. Roll back what we registered.
            for s in sessions:
                DOWNLOAD_SERVER.cancel(s.model_id)
            return _error(409, "ALREADY_DOWNLOADING",
                          f"A download for {model_id} is already in progress (race).",
                          {"model_id": model_id})
        sessions.append(session)

    schedule_batch(sessions)
    logging.info(
        "[model_downloader] scheduled %d downloads: %s",
        len(sessions), [s.model_id for s in sessions],
    )

    return _ok(schemas_out.DownloadModelsResponse(
        accepted=True,
        scheduled=[s.model_id for s in sessions],
    ), status=202)


# ----- 4. cancel a session -----


@ROUTES.post("/api/cancel-model-download-session")
async def cancel_model_download_session(request: web.Request) -> web.Response:
    parsed = await _parse_body(request, schemas_in.CancelDownloadSessionRequest)
    if isinstance(parsed, web.Response):
        return parsed

    cancelled = DOWNLOAD_SERVER.cancel(parsed.model_id)
    if not cancelled:
        return _error(404, "NOT_DOWNLOADING",
                      f"No active download for {parsed.model_id}.",
                      {"model_id": parsed.model_id})

    return _ok(schemas_out.CancelDownloadSessionResponse(cancelled=True))
