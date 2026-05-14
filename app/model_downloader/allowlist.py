"""URL allowlist for server-side model fetches.

Mirrors the frontend's ``isModelDownloadable`` allowlist so the two flows
agree on which URLs are eligible for download. Server-side allowlisting is
the primary SSRF defense for this subsystem — workflow JSON is untrusted
input (anyone can hand-craft one), so we never let the server fetch URLs
outside this list.
"""

from urllib.parse import urlparse

# Frontend parity: ``missingModelDownload-*.js`` exports the same triple as
# ``i = [...]`` (Civitai / HuggingFace / localhost).
_ALLOWED_URL_PREFIXES = (
    "https://huggingface.co/",
    "https://civitai.com/",
    "http://localhost:",
    "http://127.0.0.1:",
)

# Frontend parity: same set as ``a = [...]`` in the bundle.
_ALLOWED_MODEL_EXTENSIONS = (
    ".safetensors",
    ".sft",
    ".ckpt",
    ".pth",
    ".pt",
)


def is_url_allowed(url: str) -> bool:
    """Check whether ``url`` is permitted as a server-side download source.

    Returns True only when both:
      - the URL starts with one of the allowed prefixes, AND
      - the URL's final path segment ends with a known model extension.

    Both checks are required to keep arbitrary HTML / API endpoints on
    allowlisted hosts (e.g. ``https://huggingface.co/api/...``) off the table.
    """
    if not isinstance(url, str) or not url:
        return False
    if not any(url.startswith(p) for p in _ALLOWED_URL_PREFIXES):
        return False
    path = urlparse(url).path
    return any(path.endswith(ext) for ext in _ALLOWED_MODEL_EXTENSIONS)
