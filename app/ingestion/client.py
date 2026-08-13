import requests
from requests.auth import HTTPBasicAuth

from app.core.config import (
    CONFLUENCE_BASE_URL,
    CONFLUENCE_EMAIL,
    CONFLUENCE_API_TOKEN,
)


class ConfluenceClient:
    """
    Client for interacting with the Confluence Cloud REST API.
    """

    def __init__(self):
        self.base_url = CONFLUENCE_BASE_URL.rstrip("/")
        self.auth = HTTPBasicAuth(
            CONFLUENCE_EMAIL,
            CONFLUENCE_API_TOKEN
        )

        self.headers = {
            "Accept": "application/json"
        }

    def get(self, endpoint: str, params: dict | None = None):
        """
        Perform a GET request against the Confluence API.
        """

        url = f"{self.base_url}{endpoint}"

        response = requests.get(
            url=url,
            auth=self.auth,
            headers=self.headers,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()