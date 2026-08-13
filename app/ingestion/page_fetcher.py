from app.ingestion.client import ConfluenceClient


class PageFetcher:
    """
    Fetches individual Confluence pages.
    """

    def __init__(self):
        self.client = ConfluenceClient()

    def get_page(self, page_id: str):
        """
        Fetch a page by its ID.
        """

        endpoint = f"/wiki/rest/api/content/{page_id}"

        params = {
            "expand": (
                "body.storage,"
                "body.export_view,"
                "version,"
                "space,"
                "history,"
                "ancestors"
            )
        }

        return self.client.get(endpoint, params=params)