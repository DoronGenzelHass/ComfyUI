"""Path resolution for model downloads.

Model identifiers used across the download API are *relative destination
paths* of the form ``<directory>/<filename>`` (e.g. ``loras/my_lora.safetensors``).
This module turns one of those identifiers into an absolute on-disk path
under one of ComfyUI's registered model folders, while rejecting unknown
folders, path traversal, and other ill-formed inputs.
"""

import os
import re
from typing import Optional, Tuple

import folder_paths


# Constrain components so a model_id can never escape its target directory.
# - directory: a single path segment of safe chars
# - filename: a single path segment of safe chars, must end with a model extension
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class InvalidModelId(ValueError):
    """Raised when a model_id is syntactically invalid or refers to an
    unknown model folder."""


def parse_model_id(model_id: str) -> Tuple[str, str]:
    """Split ``<directory>/<filename>`` and validate both components.

    Returns ``(directory, filename)``. Raises ``InvalidModelId`` on
    malformed input. Does NOT touch the filesystem.
    """
    if not isinstance(model_id, str) or "/" not in model_id:
        raise InvalidModelId(f"model_id must be '<directory>/<filename>', got {model_id!r}")
    directory, _, filename = model_id.partition("/")
    if "/" in filename or not directory or not filename:
        raise InvalidModelId(f"model_id must be exactly one '/' separator, got {model_id!r}")
    if not _SEGMENT_RE.match(directory):
        raise InvalidModelId(f"invalid directory segment {directory!r}")
    if not _SEGMENT_RE.match(filename):
        raise InvalidModelId(f"invalid filename segment {filename!r}")
    if directory not in folder_paths.folder_names_and_paths:
        raise InvalidModelId(f"unknown model folder {directory!r}")
    return directory, filename


def resolve_existing(model_id: str) -> Optional[str]:
    """Return the absolute path of an installed model, or None if missing.

    Honours ``extra_model_paths.yaml`` transparently via
    ``folder_paths.get_full_path``.
    """
    directory, filename = parse_model_id(model_id)
    return folder_paths.get_full_path(directory, filename)


def resolve_destination(model_id: str) -> Tuple[str, str]:
    """Return ``(final_path, tmp_path)`` for a download.

    Downloads land at the first registered path for the model's directory
    (the "primary" location). The ``.tmp`` sibling is used as the write
    target and atomically renamed on success.
    """
    directory, filename = parse_model_id(model_id)
    roots = folder_paths.get_folder_paths(directory)
    if not roots:
        raise InvalidModelId(f"no on-disk path registered for folder {directory!r}")
    root = roots[0]
    final_path = os.path.join(root, filename)
    tmp_path = final_path + ".tmp"
    return final_path, tmp_path


def iter_all_tmp_paths():
    """Yield every ``*.tmp`` file under every registered model folder.

    Used at startup to sweep orphans left by crashed/restarted downloads.
    """
    seen_roots: set[str] = set()
    for directory in folder_paths.folder_names_and_paths.keys():
        for root in folder_paths.get_folder_paths(directory):
            if root in seen_roots or not os.path.isdir(root):
                continue
            seen_roots.add(root)
            try:
                for entry in os.scandir(root):
                    if entry.is_file() and entry.name.endswith(".tmp"):
                        yield entry.path
            except OSError:
                # Folder might be unreadable / missing on certain mounts —
                # not fatal, just skip it.
                continue
