"""Response schemas for the model-downloader API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DownloadingEntry(BaseModel):
    """One in-flight download in an availability response."""
    model_id: str
    progress: Optional[float] = None  # None until Content-Length is known
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None


class AvailabilityStatusResponse(BaseModel):
    available: list[str]
    missing: list[str]
    downloading: list[DownloadingEntry]


class MissingModelMetadataEntry(BaseModel):
    file_size: Optional[int] = None
    is_gated: bool = False


class MissingModelsMetadataResponse(BaseModel):
    # Map mirrors the request shape: model_id → metadata.
    models: dict[str, MissingModelMetadataEntry]


class DownloadModelsResponse(BaseModel):
    accepted: bool
    scheduled: list[str]


class CancelDownloadSessionResponse(BaseModel):
    cancelled: bool


__all__ = [
    "DownloadingEntry",
    "AvailabilityStatusResponse",
    "MissingModelMetadataEntry",
    "MissingModelsMetadataResponse",
    "DownloadModelsResponse",
    "CancelDownloadSessionResponse",
]
