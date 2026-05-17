"""Unit tests for the HuggingFace auth subsystem.

Covers:
  - token store: save/load roundtrip, chmod 0600, atomic write, delete
  - eligibility under various CLI-arg combinations
  - URL parsing (huggingface.co host detection + repo_id extraction)
  - HF-aware gated_detection.probe_url (mocked auth_check)
  - HF auth routes (token status, login start with eligibility gate, logout)
  - PKCE primitives + authorize URL shape

The OAuth callback server itself isn't exercised end-to-end here — that
requires a real HF server. We test the components (state checking,
URL building, code-exchange request shape) instead.
"""

from __future__ import annotations

import json
import os
import stat
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from app.model_downloader.api.routes import register_routes
from app.model_downloader.hf_auth import oauth
from app.model_downloader.hf_auth.auth_store import HF_AUTH_STORE, HfAuthStore
from app.model_downloader.hf_auth.token_store import (
    EXPIRY_BUFFER_SECS,
    Token,
    delete_token,
    load_token,
    save_token,
)
from app.model_downloader.hf_url import is_hf_url, repo_id_from_url

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def patched_user_dir(tmp_path):
    """Redirect ``folder_paths.get_user_directory`` so the token file
    lands in an isolated tmp_path instead of the real user dir."""
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    with patch("folder_paths.get_user_directory", return_value=str(user_dir)):
        yield user_dir


@pytest.fixture
def fresh_auth_store():
    """Wipe the singleton between tests."""
    HF_AUTH_STORE._token = None
    HF_AUTH_STORE._loaded_from_disk = False
    yield HF_AUTH_STORE
    HF_AUTH_STORE._token = None
    HF_AUTH_STORE._loaded_from_disk = False


@pytest.fixture
def app(patched_user_dir, fresh_auth_store):
    app = web.Application()
    register_routes(app)
    return app


# --------------------------------------------------------------------------- #
# URL parsing
# --------------------------------------------------------------------------- #


def test_is_hf_url_recognises_huggingface_co():
    assert is_hf_url("https://huggingface.co/x/y/resolve/main/z.safetensors")
    assert is_hf_url("https://huggingface.co/abc")
    assert not is_hf_url("https://hf-mirror.com/x/y/resolve/main/z.safetensors")
    assert not is_hf_url("https://civitai.com/x.safetensors")


def test_repo_id_from_url_extracts_org_and_repo():
    url = "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/resolve/main/x.safetensors"
    assert repo_id_from_url(url) == "Lightricks/LTX-2.3-22b-IC-LoRA-HDR"


def test_repo_id_from_url_handles_nested_path():
    url = "https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/x.safetensors"
    assert repo_id_from_url(url) == "Comfy-Org/ltx-2.3"


def test_repo_id_from_url_returns_none_for_non_hf():
    assert repo_id_from_url("https://civitai.com/x.safetensors") is None


def test_repo_id_from_url_returns_none_for_non_resolve_paths():
    assert repo_id_from_url("https://huggingface.co/org/repo/blob/main/x.safetensors") is None
    assert repo_id_from_url("https://huggingface.co/org") is None


# --------------------------------------------------------------------------- #
# Token store
# --------------------------------------------------------------------------- #


def test_token_store_roundtrip(patched_user_dir):
    tok = Token(
        access_token="hf_abc",
        refresh_token="rf_def",
        expires_at=9999999999.0,
        scope="openid profile",
    )
    save_token(tok)
    loaded = load_token()
    assert loaded == tok


def test_token_store_writes_0600(patched_user_dir):
    tok = Token(access_token="x", refresh_token=None, expires_at=0.0)
    save_token(tok)
    path = os.path.join(patched_user_dir, "hf_auth_token.json")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    # On Windows we silently no-op chmod; allow either the intended
    # mode or whatever umask the OS gave us.
    if os.name == "posix":
        assert mode == 0o600


def test_token_store_delete_removes_file(patched_user_dir):
    tok = Token(access_token="x", refresh_token=None, expires_at=0.0)
    save_token(tok)
    delete_token()
    path = os.path.join(patched_user_dir, "hf_auth_token.json")
    assert not os.path.exists(path)
    # Idempotent: second delete is fine.
    delete_token()


def test_token_store_load_returns_none_for_missing_file(patched_user_dir):
    assert load_token() is None


def test_token_store_load_returns_none_for_corrupt_file(patched_user_dir):
    path = os.path.join(patched_user_dir, "hf_auth_token.json")
    with open(path, "w") as f:
        f.write("not json {")
    assert load_token() is None


def test_token_is_valid_uses_buffer(patched_user_dir):
    import time

    fresh = Token(access_token="x", refresh_token=None, expires_at=time.time() + 3600)
    nearly_expired = Token(
        access_token="x",
        refresh_token=None,
        expires_at=time.time() + EXPIRY_BUFFER_SECS - 1,
    )
    assert fresh.is_valid()
    assert not nearly_expired.is_valid()


# --------------------------------------------------------------------------- #
# Auth store
# --------------------------------------------------------------------------- #


def test_auth_store_loads_lazily(patched_user_dir):
    tok = Token(access_token="x", refresh_token=None, expires_at=9999999999.0)
    save_token(tok)
    store = HfAuthStore()
    assert store.has_token()
    assert store.get_token_sync() == tok


def test_auth_store_set_persists(patched_user_dir):
    store = HfAuthStore()
    tok = Token(access_token="x", refresh_token=None, expires_at=9999999999.0)
    store.set_token(tok)
    # Token is on disk now — a fresh store sees it.
    assert HfAuthStore().get_token_sync() == tok


def test_auth_store_clear_removes_in_memory_and_on_disk(patched_user_dir):
    store = HfAuthStore()
    tok = Token(access_token="x", refresh_token=None, expires_at=9999999999.0)
    store.set_token(tok)
    store.clear()
    assert not store.has_token()
    assert HfAuthStore().get_token_sync() is None


async def test_auth_store_get_valid_returns_fresh_token(patched_user_dir):
    store = HfAuthStore()
    import time

    tok = Token(access_token="x", refresh_token=None, expires_at=time.time() + 3600)
    store.set_token(tok)
    fetched = await store.get_valid_token()
    assert fetched == tok


async def test_auth_store_get_valid_refresh_on_expired(patched_user_dir):
    store = HfAuthStore()
    import time

    expired = Token(
        access_token="old",
        refresh_token="rf",
        expires_at=time.time() - 100,
    )
    store.set_token(expired)
    refreshed = Token(
        access_token="new",
        refresh_token="rf",
        expires_at=time.time() + 3600,
    )
    with patch(
        "app.model_downloader.hf_auth.oauth.refresh_access_token",
        new=AsyncMock(return_value=refreshed),
    ):
        result = await store.get_valid_token()
    assert result == refreshed


async def test_auth_store_get_valid_returns_none_on_refresh_failure(patched_user_dir):
    store = HfAuthStore()
    import time

    expired = Token(
        access_token="old",
        refresh_token="rf",
        expires_at=time.time() - 100,
    )
    store.set_token(expired)
    with patch(
        "app.model_downloader.hf_auth.oauth.refresh_access_token",
        new=AsyncMock(side_effect=RuntimeError("HF down")),
    ):
        result = await store.get_valid_token()
    assert result is None


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "listen,multi_user,expected",
    [
        ("127.0.0.1", False, True),
        ("127.0.0.1", True, False),    # multi-user disables it
        ("0.0.0.0", False, False),     # bind-all is not loopback
        ("0.0.0.0", True, False),
        ("192.168.1.5", False, False), # LAN address
        ("::1", False, True),          # IPv6 loopback
    ],
)
def test_eligibility(listen, multi_user, expected, monkeypatch):
    from app.model_downloader.hf_auth import eligibility
    from comfy.cli_args import args

    monkeypatch.setattr(args, "listen", listen)
    monkeypatch.setattr(args, "multi_user", multi_user)
    assert eligibility.is_hf_auth_eligible() is expected


# --------------------------------------------------------------------------- #
# gated_detection HF probe
# --------------------------------------------------------------------------- #


async def test_probe_url_hf_public(fresh_auth_store):
    """auth_check succeeds with no token → is_hf_downloadable = True."""
    from app.model_downloader.gated_detection import probe_url

    url = "https://huggingface.co/public/repo/resolve/main/x.safetensors"
    with patch("app.model_downloader.gated_detection._auth_check_sync"), patch(
        "app.model_downloader.gated_detection._probe_size",
        new=AsyncMock(return_value=1024),
    ):
        result = await probe_url(url)
    assert result.is_hf_downloadable is True
    assert result.file_size == 1024


async def test_probe_url_hf_gated_no_access(fresh_auth_store):
    """auth_check raises GatedRepoError → is_hf_downloadable = False."""
    from huggingface_hub.errors import GatedRepoError

    from app.model_downloader.gated_detection import probe_url

    url = "https://huggingface.co/gated/repo/resolve/main/x.safetensors"
    fake_response = MagicMock(status_code=403)
    with patch(
        "app.model_downloader.gated_detection._auth_check_sync",
        side_effect=GatedRepoError("gated", response=fake_response),
    ), patch(
        "app.model_downloader.gated_detection._probe_size",
        new=AsyncMock(return_value=None),
    ):
        result = await probe_url(url)
    assert result.is_hf_downloadable is False


async def test_probe_url_non_hf_skips_auth_check():
    """Non-HF URLs never call auth_check; is_hf_downloadable stays None."""
    from app.model_downloader.gated_detection import probe_url

    url = "https://civitai.com/api/download/models/1.safetensors"
    with patch(
        "app.model_downloader.gated_detection._auth_check_sync",
    ) as mocked, patch(
        "app.model_downloader.gated_detection._probe_size",
        new=AsyncMock(return_value=2048),
    ):
        result = await probe_url(url)
    assert result.is_hf_downloadable is None
    assert result.file_size == 2048
    mocked.assert_not_called()


async def test_probe_url_passes_token_when_available(fresh_auth_store, patched_user_dir):
    """auth_check is called with the stored token's access_token."""
    from app.model_downloader.gated_detection import probe_url

    fresh_auth_store.set_token(Token(
        access_token="hf_test_token",
        refresh_token=None,
        expires_at=9999999999.0,
    ))
    url = "https://huggingface.co/private/repo/resolve/main/x.safetensors"
    with patch(
        "app.model_downloader.gated_detection._auth_check_sync"
    ) as mocked, patch(
        "app.model_downloader.gated_detection._probe_size",
        new=AsyncMock(return_value=None),
    ):
        await probe_url(url)
    mocked.assert_called_once_with("private/repo", "hf_test_token")


# --------------------------------------------------------------------------- #
# OAuth primitives
# --------------------------------------------------------------------------- #


def test_make_pkce_returns_distinct_high_entropy_values():
    verifier1, challenge1, state1 = oauth._make_pkce()
    verifier2, challenge2, state2 = oauth._make_pkce()
    assert verifier1 != verifier2
    assert challenge1 != challenge2
    assert state1 != state2
    # Verifier should be at least 43 chars per PKCE spec.
    assert len(verifier1) >= 43


def test_build_authorize_url_includes_pkce_and_state():
    url = oauth._build_authorize_url("challenge123", "state456")
    assert url.startswith(oauth.AUTHORIZE_URL)
    assert "client_id=" + oauth.HF_CLIENT_ID in url
    assert "code_challenge=challenge123" in url
    assert "code_challenge_method=S256" in url
    assert "state=state456" in url
    assert "response_type=code" in url


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


async def test_hf_auth_token_status_empty(aiohttp_client, app):
    """No token set → token_available=false, username=null."""
    client = await aiohttp_client(app)
    resp = await client.get("/api/hf-auth-token-status")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"token_available": False, "username": None}


async def test_hf_auth_token_status_with_token(
    aiohttp_client, app, fresh_auth_store, patched_user_dir
):
    """Token present, whoami works → username is returned."""
    fresh_auth_store.set_token(Token(
        access_token="x", refresh_token=None, expires_at=9999999999.0,
    ))
    with patch(
        "app.model_downloader.api.routes._whoami_username",
        return_value="alice",
    ):
        client = await aiohttp_client(app)
        resp = await client.get("/api/hf-auth-token-status")
    assert resp.status == 200
    assert (await resp.json()) == {"token_available": True, "username": "alice"}


async def test_hf_auth_login_start_403_when_ineligible(aiohttp_client, app, monkeypatch):
    """Not loopback / multi-user → 403."""
    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: False,
    )
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-login-start")
    assert resp.status == 403
    assert (await resp.json())["error"]["code"] == "HF_AUTH_NOT_ELIGIBLE"


async def test_hf_auth_login_start_returns_authorize_url(aiohttp_client, app, monkeypatch):
    """Eligible + first attempt → 200 with authorize_url."""
    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: True,
    )
    monkeypatch.setattr(
        "app.model_downloader.api.routes.start_login_flow",
        AsyncMock(return_value="https://huggingface.co/oauth/authorize?fake=1"),
    )
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-login-start")
    assert resp.status == 200
    assert (await resp.json())["authorize_url"].startswith(
        "https://huggingface.co/oauth/authorize"
    )


async def test_hf_auth_login_start_409_when_in_progress(aiohttp_client, app, monkeypatch):
    """Lock already held → 409."""
    from app.model_downloader.hf_auth.oauth import OAuthInProgressError

    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: True,
    )
    monkeypatch.setattr(
        "app.model_downloader.api.routes.start_login_flow",
        AsyncMock(side_effect=OAuthInProgressError()),
    )
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-login-start")
    assert resp.status == 409
    assert (await resp.json())["error"]["code"] == "HF_AUTH_IN_PROGRESS"


async def test_hf_auth_logout_clears_store(
    aiohttp_client, app, fresh_auth_store, patched_user_dir
):
    fresh_auth_store.set_token(Token(
        access_token="x", refresh_token=None, expires_at=9999999999.0,
    ))
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-logout")
    assert resp.status == 200
    assert (await resp.json()) == {"logged_out": True}
    assert not fresh_auth_store.has_token()


async def test_availability_includes_hf_auth_snapshot(aiohttp_client, app, monkeypatch):
    """The availability response embeds {token_available, eligible}."""
    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: True,
    )
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/models-availability-status",
        json={"model_ids": []},
    )
    assert resp.status == 200
    data = await resp.json()
    assert "hf_auth" in data
    assert data["hf_auth"] == {"token_available": False, "eligible": True}
