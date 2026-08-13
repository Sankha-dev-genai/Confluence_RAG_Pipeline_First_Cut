from app.ingestion.client import ConfluenceClient
from app.ingestion.models import ConfluencePage


class HierarchyBuilder:

    def __init__(self):
        self.client = ConfluenceClient()

    def build_tree(self, page_id: str, level: int = 0):

        page = self.client.get(
            f"/wiki/rest/api/content/{page_id}"
        )

        node = ConfluencePage(
            page_id=page["id"],
            title=page["title"],
            level=level
        )

        children = self.client.get(
            f"/wiki/rest/api/content/{page_id}/child/page"
        )

        for child in children.get("results", []):

            child_node = self.build_tree(
                child["id"],
                level + 1
            )

            child_node.parent_id = page["id"]

            node.children.append(child_node)

        return node