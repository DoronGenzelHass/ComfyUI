"""HEAD-based gated-repo / file-size detection.

Mirrors the heuristic the frontend ships in
``missingModelDownload-*.js``: a HEAD against the model URL whose result
distinguishes three cases:

  - 200 OK         → file is publicly downloadable; surface
                     ``Content-Length`` as ``file_size``.
  - 401/403/451 on a ``huggingface.co`` URL → gated repo (license to
                     accept); ``is_gated = True``.
  - anything else  → unknown; both fields ``None``/``False``.

This phase intentionally does not authenticate to HuggingFace; gated
detection lets the frontend show the right message and the user
acquires the file out-of-band. Adding real ``hf_hub`` auth is a future
phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import aiohttp


# Same {401, 403, 451} set the frontend checks. Worth keeping as one
# constant so the two sides cannot drift independently.
_GATED_STATUS_CODES = frozenset({401, 403, 451})

_HF_HOST_MARKER = "huggingface.co"

_HEAD_TIMEOUT = aiohttp.ClientTimeout(total=15)


@dataclass
class MetadataProbeResult:
    file_size: Optional[int]
    is_gated: bool


async def probe_url(url: str) -> MetadataProbeResult:
    """HEAD the URL and classify the response.

    Returns a result that's always safe to serialize even on network
    errors — we never raise out of here.
    """
    try:
        async with aiohttp.ClientSession(timeout=_HEAD_TIMEOUT) as session:
            async with session.head(url, allow_redirects=True) as resp:
                if resp.status == 200:
                    return MetadataProbeResult(
                        file_size=_parse_content_length(resp.headers.get("Content-Length")),
                        is_gated=False,
                    )
                if resp.status in _GATED_STATUS_CODES and _HF_HOST_MARKER in url:
                    return MetadataProbeResult(file_size=None, is_gated=True)
                return MetadataProbeResult(file_size=None, is_gated=False)
    except (aiohttp.ClientError, TimeoutError, OSError):
        # Network blip / DNS failure / TLS issue: don't fail the whole
        # request, just report "unknown".
        return MetadataProbeResult(file_size=None, is_gated=False)


def _parse_content_length(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        n = int(value)
    except ValueError:
        return None
    return n if n >= 0 else None
