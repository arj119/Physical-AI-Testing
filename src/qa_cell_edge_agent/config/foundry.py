"""Foundry clients — typed OSDK via ``physical_ai_qa_cell_sdk`` for ontology
actions and object queries, raw ``requests.Session`` for stream push.

The SDK's ``ConfidentialClientAuth`` manages OAuth2 token refresh automatically.
Stream push uses a separate token scoped to ``api:use-streams-write``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests
from physical_ai_qa_cell_sdk import ConfidentialClientAuth, FoundryClient

from qa_cell_edge_agent.config.settings import Settings

logger = logging.getLogger(__name__)

OSDK_SCOPES = [
    "api:use-ontologies-read",
    "api:use-ontologies-write",
    "api:use-streams-read",
    "api:use-streams-write",
    "api:use-mediasets-read",
    "api:use-mediasets-write",
]


class FoundryClients:
    """Lazy-initialised Foundry clients that share a single OAuth2 identity.

    The SDK ``ConfidentialClientAuth`` handles the OSDK token (ontology scopes).
    Stream push requires a **separate** token minted with ``api:use-streams-write``
    because the high-scale streams endpoint rejects multi-scope tokens.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._auth = None
        self._client = None
        self._stream_session: Optional[requests.Session] = None
        self._stream_token: Optional[str] = None
        self._stream_token_expiry: float = 0.0

    # ── SDK auth + client (OSDK scopes) ──────────────────────────────

    @property
    def auth(self):
        """``ConfidentialClientAuth`` with auto-refresh for OSDK operations."""
        if self._auth is None:
            self._auth = ConfidentialClientAuth(
                client_id=self.settings.client_id,
                client_secret=self.settings.client_secret,
                hostname=self.settings.foundry_url,
                should_refresh=True,
                scopes=OSDK_SCOPES,
            )
        return self._auth

    @property
    def client(self):
        """Typed OSDK ``FoundryClient`` for actions and object queries."""
        if self._client is None:
            self._client = FoundryClient(
                auth=self.auth,
                hostname=self.settings.foundry_url,
            )
        return self._client

    # ── Stream push (separate token with streams-write scope) ────────

    def _refresh_stream_token(self) -> str:
        """Obtain a token scoped specifically for stream push."""
        if self._stream_token and time.time() < self._stream_token_expiry - 30:
            return self._stream_token

        resp = requests.post(
            f"{self.settings.foundry_url}/multipass/api/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.settings.client_id,
                "client_secret": self.settings.client_secret,
                "scope": "api:use-streams-write",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        self._stream_token = body["access_token"]
        self._stream_token_expiry = time.time() + body.get("expires_in", 3600)
        logger.info("Stream push token refreshed")
        return self._stream_token

    @property
    def session(self) -> requests.Session:
        """Return a ``requests.Session`` with the stream-scoped token."""
        if self._stream_session is None:
            self._stream_session = requests.Session()
        token = self._refresh_stream_token()
        self._stream_session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        return self._stream_session

    # ── Stream push (v2 high-scale streams API) ──────────────────────

    def push_to_stream(
        self,
        stream_rid: str,
        records: List[Dict[str, Any]],
    ) -> bool:
        """Push records to a Foundry v2 high-scale stream.

        Returns True on success, False on failure (after retries).
        """
        url = (
            f"{self.settings.foundry_url}/api/v2/highScale/streams/datasets"
            f"/{stream_rid}/streams/master/publishRecords?preview=true"
        )
        payload = {"records": records}

        for attempt in range(1, self.settings.stream_retry_count + 1):
            try:
                resp = self.session.post(
                    url,
                    data=json.dumps(payload),
                    timeout=self.settings.stream_push_timeout_sec,
                )
                if resp.status_code not in (200, 204):
                    logger.warning(
                        "Stream push attempt %d/%d failed for %s: HTTP %d\n"
                        "  Response: %s\n"
                        "  Payload sample: %s",
                        attempt,
                        self.settings.stream_retry_count,
                        stream_rid,
                        resp.status_code,
                        resp.text[:500],
                        json.dumps(records[0] if records else {})[:300],
                    )
                    time.sleep(min(2**attempt, 10))
                    continue
                return True
            except requests.RequestException as exc:
                logger.warning(
                    "Stream push attempt %d/%d failed for %s: %s",
                    attempt,
                    self.settings.stream_retry_count,
                    stream_rid,
                    exc,
                )
                time.sleep(min(2**attempt, 10))
        return False
