"""Metadata probe: file size + HF reachability.

Two independent questions per URL:

  - ``file_size`` — the byte length of the file, when the server is
    willing to disclose it via a HEAD's ``Content-Length``. Always
    optional; we don't fail the request when it's missing.

  - ``is_hf_downloadable`` — *for HuggingFace URLs only*, can the
    server fetch this with the currently-stored auth token? Computed
    via ``HfApi.auth_check`` (same path the LTX downloader uses).
    ``None`` for non-HF URLs (the concept doesn't apply) and for HF
    URLs where the probe failed entirely (network blip).

Both checks are best-effort: any exception ends up as ``None``/``None``
so the route can return a clean response.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from app.model_downloader.hf_auth.auth_store import HF_AUTH_STORE
from app.model_downloader.hf_url import is_hf_url, repo_id_from_url
from app.model_downloader.http_client import get_session


_HEAD_TIMEOUT = aiohttp.ClientTimeout(total=15)


@dataclass
class MetadataProbeResult:
    file_size: Optional[int]
    is_hf_downloadable: Optional[bool]


async def probe_url(url: str) -> MetadataProbeResult:
    """HEAD for size + (HF-only) auth_check for downloadability.

    Both probes run concurrently when the URL is on HF; for non-HF
    URLs we just HEAD for size.
    """
    if is_hf_url(url):
        size_task = asyncio.create_task(_probe_size(url))
        hf_task = asyncio.create_task(_probe_hf_downloadable(url))
        size, hf_downloadable = await asyncio.gather(size_task, hf_task)
        return MetadataProbeResult(file_size=size, is_hf_downloadable=hf_downloadable)

    size = await _probe_size(url)
    return MetadataProbeResult(file_size=size, is_hf_downloadable=None)


# --- size probe ---------------------------------------------------------- #


async def _probe_size(url: str) -> Optional[int]:
    """HEAD the URL and return ``Content-Length`` if available."""
    try:
        session = await get_session()
        async with session.head(url, allow_redirects=True, timeout=_HEAD_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return _parse_content_length(resp.headers.get("Content-Length"))
    except (aiohttp.ClientError, TimeoutError, OSError):
        return None


def _parse_content_length(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        n = int(value)
    except ValueError:
        return None
    return n if n >= 0 else None


# --- HF auth_check probe ------------------------------------------------- #


async def _probe_hf_downloadable(url: str) -> Optional[bool]:
    """Return whether the server can fetch this HF URL with current auth.

    ``True``  → repo is public, or gated and current token has access
    ``False`` → repo is gated and current token (if any) lacks access
    ``None``  → couldn't determine (network error, malformed URL, etc.)
    """
    repo_id = repo_id_from_url(url)
    if repo_id is None:
        return None

    token = HF_AUTH_STORE.get_token_sync()
    token_str: Optional[str] = token.access_token if token else None

    # auth_check is synchronous + blocks on a network call; run it in
    # a worker thread so we don't stall the event loop.
    try:
        await asyncio.to_thread(_auth_check_sync, repo_id, token_str)
        return True
    except GatedRepoError:
        return False
    except RepositoryNotFoundError:
        # The repo doesn't exist or the user can't even see it as private.
        # Either way, the server can't download it — treat the same as
        # "no access".
        return False
    except HfHubHTTPError as e:
        logging.debug("[hf_auth] auth_check transient failure for %s: %s", repo_id, e)
        return None
    except Exception as e:
        logging.warning("[hf_auth] unexpected auth_check error for %s: %s", repo_id, e)
        return None


def _auth_check_sync(repo_id: str, token: Optional[str]) -> None:
    """Thin sync wrapper around ``HfApi.auth_check`` for ``asyncio.to_thread``."""
    api = HfApi()
    api.auth_check(repo_id, token=token)
