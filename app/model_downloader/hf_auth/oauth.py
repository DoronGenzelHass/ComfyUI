"""OAuth 2.0 PKCE flow against HuggingFace's authorization server.

Wired so that ``POST /api/hf-auth-login-start`` can:
  1. Generate state + PKCE verifier/challenge in this process.
  2. Spin up a short-lived loopback HTTP server at port 41954 to
     receive the redirect callback from HF.
  3. Return the ``authorize_url`` for the frontend to open in a new tab.

After the user grants consent on huggingface.co, HF redirects to the
local callback URL with ``code`` and ``state``. The callback server
validates ``state`` (CSRF), exchanges the code for tokens via PKCE,
hands the resulting Token to ``HF_AUTH_STORE.set_token``, and shuts
itself down.

# TODO: register a dedicated ComfyUI-Org OAuth app on huggingface.co
# and replace ``HF_CLIENT_ID`` + ``REDIRECT_URI`` below. For the POC we
# reuse the LTX model-downloader node's registered OAuth app.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import threading
import time

import aiohttp
from aiohttp import web

from app.model_downloader.hf_auth.auth_store import HF_AUTH_STORE
from app.model_downloader.hf_auth.token_store import Token


# --- HF OAuth app registration -------------------------------------------- #
# Reusing the LTX model-downloader node's client_id for the POC. See module
# docstring.
HF_CLIENT_ID = "a8189e14-9246-4f19-bd6a-a307bdcb9276"

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 41954
CALLBACK_PATH = "/api/auth/huggingface/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

AUTHORIZE_URL = "https://huggingface.co/oauth/authorize"
TOKEN_URL = "https://huggingface.co/oauth/token"
SCOPE = "openid profile read-repos"

# Maximum time the callback server stays up waiting for the user to
# complete consent on huggingface.co. Past this, the port closes and
# the user has to click "Log in" again.
CALLBACK_TIMEOUT_SECS = 300


# Process-wide lock so two simultaneous /api/hf-auth-login-start
# requests don't fight over port CALLBACK_PORT.
_OAUTH_LOCK = threading.Lock()


class OAuthInProgressError(Exception):
    """Another OAuth attempt is already running."""


class OAuthCallbackError(Exception):
    """The OAuth callback returned an error (HF denied, port stolen, etc.)."""


# --- PKCE primitives ------------------------------------------------------ #


def _make_pkce() -> tuple[str, str, str]:
    """Return ``(verifier, challenge, state)``.

    Verifier never leaves this process. Challenge and state travel
    through the user's browser. State is checked on the callback to
    prevent a malicious cross-origin redirect from injecting a token.
    """
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    state = secrets.token_urlsafe(32)
    return verifier, challenge, state


def _build_authorize_url(challenge: str, state: str) -> str:
    from urllib.parse import urlencode

    params = {
        "client_id": HF_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# --- Token exchange ------------------------------------------------------- #


async def _exchange_code(code: str, verifier: str) -> Token:
    """Trade the authorization code for an access+refresh token pair."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": HF_CLIENT_ID,
        "code_verifier": verifier,
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(TOKEN_URL, data=data) as resp:
            resp.raise_for_status()
            body = await resp.json()
    return Token(
        access_token=body["access_token"],
        refresh_token=body.get("refresh_token"),
        expires_at=time.time() + float(body.get("expires_in", 3600)),
        scope=body.get("scope", SCOPE),
    )


async def refresh_access_token(refresh_token: str) -> Token:
    """Trade a refresh_token for a new access (+ possibly refresh) token."""
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": HF_CLIENT_ID,
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(TOKEN_URL, data=data) as resp:
            resp.raise_for_status()
            body = await resp.json()
    return Token(
        access_token=body["access_token"],
        # If HF doesn't rotate refresh tokens, keep using the existing one.
        refresh_token=body.get("refresh_token", refresh_token),
        expires_at=time.time() + float(body.get("expires_in", 3600)),
        scope=body.get("scope", SCOPE),
    )


# --- Callback server ------------------------------------------------------ #


async def start_login_flow() -> str:
    """Begin one OAuth attempt: spawn the callback server, return the URL.

    Returns the URL the frontend should open in a new tab. Raises
    ``OAuthInProgressError`` if another attempt is already running.
    The callback server runs in the background until the user
    completes consent or until ``CALLBACK_TIMEOUT_SECS`` elapses;
    either way the lock + port are released afterward.
    """
    if not _OAUTH_LOCK.acquire(blocking=False):
        raise OAuthInProgressError()

    verifier, challenge, state = _make_pkce()
    authorize_url = _build_authorize_url(challenge, state)

    # Fire the callback server on the running loop and return.
    asyncio.create_task(_run_callback_server(verifier, state))
    return authorize_url


async def _run_callback_server(verifier: str, expected_state: str) -> None:
    """Listen for HF's redirect once, capture the token, then shut down."""
    received: asyncio.Future[Token] = asyncio.get_event_loop().create_future()

    async def handler(request: web.Request) -> web.Response:
        try:
            if request.query.get("state") != expected_state:
                return web.Response(status=400, text="state mismatch")
            err = request.query.get("error")
            if err:
                received.set_exception(OAuthCallbackError(f"HF returned: {err}"))
                return web.Response(status=400, text=f"OAuth error: {err}")
            code = request.query.get("code")
            if not code:
                return web.Response(status=400, text="missing code")
            tok = await _exchange_code(code, verifier)
            if not received.done():
                received.set_result(tok)
            return web.Response(
                content_type="text/html",
                text=(
                    "<html><body style='font-family:sans-serif;padding:40px'>"
                    "<h2>HuggingFace login successful</h2>"
                    "<p>You can close this tab and return to ComfyUI.</p>"
                    "</body></html>"
                ),
            )
        except Exception as exc:
            if not received.done():
                received.set_exception(exc)
            return web.Response(status=500, text=str(exc))

    app = web.Application()
    app.router.add_get(CALLBACK_PATH, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, CALLBACK_HOST, CALLBACK_PORT, reuse_address=True)
    try:
        await site.start()
    except OSError as e:
        # Port already in use (or some other socket-bind failure). Release
        # the lock so a future attempt has a chance to succeed.
        logging.warning("[hf_auth] could not bind callback port: %s", e)
        _OAUTH_LOCK.release()
        return

    try:
        token = await asyncio.wait_for(received, timeout=CALLBACK_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        logging.info("[hf_auth] OAuth login timed out after %ds", CALLBACK_TIMEOUT_SECS)
        return
    except OAuthCallbackError as e:
        logging.warning("[hf_auth] OAuth callback error: %s", e)
        return
    except Exception as e:
        logging.warning("[hf_auth] unexpected OAuth failure: %s", e)
        return
    else:
        HF_AUTH_STORE.set_token(token)
        logging.info("[hf_auth] OAuth login complete")
    finally:
        await runner.cleanup()
        if _OAUTH_LOCK.locked():
            _OAUTH_LOCK.release()


def is_login_in_progress() -> bool:
    """True iff a callback server is currently bound + waiting."""
    return _OAUTH_LOCK.locked()


# Re-export for callers that only want the URL builder (e.g. tests).
__all__ = [
    "start_login_flow",
    "refresh_access_token",
    "is_login_in_progress",
    "OAuthInProgressError",
    "CALLBACK_TIMEOUT_SECS",
]
