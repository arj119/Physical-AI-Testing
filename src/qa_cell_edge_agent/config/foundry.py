"""Foundry clients — typed OSDK via ``physical_ai_qa_cell_sdk`` for ontology
actions and object queries, raw ``requests.Session`` for stream push.

The SDK's ``ConfidentialClientAuth`` manages OAuth2 token refresh automatically.
The same auth token is reused for stream push via ``auth.get_token()``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

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

# Lazy imports — avoid import errors when SDK isn't installed (e.g. in tests)
_FoundryClient = None
_ConfidentialClientAuth = None


def _ensure_sdk():
    global _FoundryClient, _ConfidentialClientAuth
    if _FoundryClient is None:
        from physical_ai_qa_cell_sdk import (
            ConfidentialClientAuth,
            FoundryClient,
        )
        _FoundryClient = FoundryClient
        _ConfidentialClientAuth = ConfidentialClientAuth


class FoundryClients:
    """Lazy-initialised Foundry clients that share a single OAuth2 identity."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._auth = None
        self._client = None
        self._session: Optional[requests.Session] = None

    # ── SDK auth + client ────────────────────────────────────────────

    @property
    def auth(self):
        """``ConfidentialClientAuth`` with auto-refresh."""
        if self._auth is None:
            _ensure_sdk()
            self._auth = _ConfidentialClientAuth(
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
            _ensure_sdk()
            self._client = _FoundryClient(
                auth=self.auth,
                hostname=self.settings.foundry_url,
            )
        return self._client

    # ── Authenticated session for stream push ────────────────────────

    @property
    def session(self) -> requests.Session:
        """Return a ``requests.Session`` with the SDK auth token."""
        if self._session is None:
            self._session = requests.Session()
        token = self.auth.get_token()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        return self._session

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
                resp.raise_for_status()
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
