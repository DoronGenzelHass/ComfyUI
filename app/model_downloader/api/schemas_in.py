"""Request schemas for the model-downloader API.

Each endpoint accepts a small JSON body. Pydantic enforces the shape at
the boundary; route handlers operate only on validated values past that.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AvailabilityStatusRequest(BaseModel):
    """``POST /api/models-availability-status``.

    Sent by the frontend on each poll. Lists the model_ids it cares
    about (i.e. the ones declared by the loaded workflow). The server's
    response only mentions those ids — other in-flight downloads on
    the server are not leaked.
    """
    model_ids: list[str] = Field(default_factory=list)


class MissingModelsMetadataRequest(BaseModel):
    """``POST /api/missing-models-metadata``.

    Maps model_id → URL for each model the frontend wants metadata on.
    The URL is the one declared in ``properties.models[i].url`` in the
    workflow JSON.
    """
    models: dict[str, str] = Field(default_factory=dict)


class DownloadModelsRequest(BaseModel):
    """``POST /api/download-models``.

    Same shape as the metadata request — the URL for each model_id.
    Returns immediately after validation and scheduling.
    """
    models: dict[str, str] = Field(default_factory=dict)


class CancelDownloadSessionRequest(BaseModel):
    """``POST /api/cancel-model-download-session``."""
    model_id: str


__all__ = [
    "AvailabilityStatusRequest",
    "MissingModelsMetadataRequest",
    "DownloadModelsRequest",
    "CancelDownloadSessionRequest",
]
