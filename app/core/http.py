"""A small requests session with retries/backoff and a default timeout."""
from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None  # type: ignore


class HttpClient:
    def __init__(self, retries: int = 3, backoff: float = 0.5, timeout: int = 30,
                 auth=None, headers: dict | None = None) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        if auth is not None:
            self.session.auth = auth
        self.session.headers.update(headers or {"Accept": "application/json"})
        if Retry is not None:
            retry = Retry(
                total=retries, backoff_factor=backoff,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

    def get(self, url: str, params: dict | None = None) -> Any:
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def post(self, url: str, data: dict | None = None, json: dict | None = None) -> Any:
        r = self.session.post(url, data=data, json=json, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
