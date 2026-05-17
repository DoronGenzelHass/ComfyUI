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


class HfAuthStatus(BaseModel):
    """Folded into the availability response so the frontend can poll one
    endpoint and learn whether the HF login state changed (which would
    affect ``is_hf_downloadable`` on subsequent metadata probes)."""
    token_available: bool
    eligible: bool


class AvailabilityStatusResponse(BaseModel):
    available: list[str]
    missing: list[str]
    downloading: list[DownloadingEntry]
    hf_auth: HfAuthStatus


class MissingModelMetadataEntry(BaseModel):
    file_size: Optional[int] = None
    # HF-only: whether the server can fetch the URL right now (true =
    # public or token has access; false = gated, no access; null =
    # non-HF URL, or probe failed). Frontend renders a "gated" UI
    # iff this is false.
    is_hf_downloadable: Optional[bool] = None


class MissingModelsMetadataResponse(BaseModel):
    # Map mirrors the request shape: model_id → metadata.
    models: dict[str, MissingModelMetadataEntry]


class DownloadModelsResponse(BaseModel):
    accepted: bool
    scheduled: list[str]


class CancelDownloadSessionResponse(BaseModel):
    cancelled: bool


class HfAuthTokenStatusResponse(BaseModel):
    token_available: bool
    # Convenience for the settings UI — populated when the token is
    # present + still works against ``HfApi.whoami``. ``None`` otherwise.
    username: Optional[str] = None


class HfAuthLoginStartResponse(BaseModel):
    authorize_url: str


class HfAuthLogoutResponse(BaseModel):
    logged_out: bool


__all__ = [
    "DownloadingEntry",
    "HfAuthStatus",
    "AvailabilityStatusResponse",
    "MissingModelMetadataEntry",
    "MissingModelsMetadataResponse",
    "DownloadModelsResponse",
    "CancelDownloadSessionResponse",
    "HfAuthTokenStatusResponse",
    "HfAuthLoginStartResponse",
    "HfAuthLogoutResponse",
]
