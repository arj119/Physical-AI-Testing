"""Foundry client factories — OSDK for Ontology actions, requests.Session for stream push."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from qa_cell_edge_agent.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class FoundryClients:
    """Lazy-initialised Foundry clients that share a single OAuth2 token."""

    settings: Settings
    _token: Optional[str] = None
    _token_expiry: float = 0.0
    _session: Optional[requests.Session] = None

    # ── OAuth2 ────────────────────────────────────────────────────────

    def _refresh_token(self) -> str:
        """Obtain or refresh an OAuth2 bearer token via client_credentials grant."""
        if self._token and time.time() < self._token_expiry - 30:
            return self._token

        resp = requests.post(
            f"{self.settings.foundry_url}/multipass/api/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.settings.client_id,
                "client_secret": self.settings.client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        self._token_expiry = time.time() + body.get("expires_in", 3600)
        logger.info("OAuth2 token refreshed, expires in %ss", body.get("expires_in"))
        return self._token

    @property
    def session(self) -> requests.Session:
        """Return an authenticated requests.Session (token auto-refreshed)."""
        if self._session is None:
            self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self._refresh_token()}"}
        )
        return self._session

    # ── Stream push ───────────────────────────────────────────────────

    def push_to_stream(self, stream_rid: str, records: list[Dict[str, Any]]) -> bool:
        """Push one or more JSON records to a Foundry Stream.

        Returns True on success, False on failure (after retries).
        """
        url = (
            f"{self.settings.foundry_url}/stream-proxy/api/streams"
            f"/{stream_rid}/publishRecords"
        )
        payload = {"records": [{"value": r} for r in records]}

        for attempt in range(1, self.settings.stream_retry_count + 1):
            try:
                resp = self.session.post(
                    url, json=payload, timeout=self.settings.stream_push_timeout_sec
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

    # ── OSDK helpers ──────────────────────────────────────────────────

    def apply_action(self, action_type_rid: str, params: Dict[str, Any]) -> bool:
        """Execute an Ontology action via the OSDK REST API.

        Returns True on success, False on failure.
        """
        url = (
            f"{self.settings.foundry_url}/api/v2/ontologies"
            f"/ri.ontology.main.ontology.8d41bc1c-890d-4702-972b-98035f877b96"
            f"/actions/{action_type_rid}/apply"
        )
        try:
            resp = self.session.post(url, json={"parameters": params}, timeout=10)
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.error("Action %s failed: %s", action_type_rid, exc)
            return False

    def query_objects(
        self,
        object_type: str,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        page_size: int = 100,
    ) -> list[Dict[str, Any]]:
        """Load objects from the Ontology via the OSDK REST API."""
        url = (
            f"{self.settings.foundry_url}/api/v2/ontologies"
            f"/ri.ontology.main.ontology.8d41bc1c-890d-4702-972b-98035f877b96"
            f"/objects/{object_type}/search"
        )
        body: Dict[str, Any] = {"pageSize": page_size}
        if where:
            body["where"] = where
        if order_by:
            body["orderBy"] = {"fields": [{"field": order_by, "direction": "desc"}]}
        try:
            resp = self.session.post(url, json=body, timeout=10)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except requests.RequestException as exc:
            logger.error("Object query for %s failed: %s", object_type, exc)
            return []
