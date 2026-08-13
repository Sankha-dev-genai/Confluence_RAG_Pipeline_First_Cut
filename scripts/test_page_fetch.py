from app.core.config import CONFLUENCE_PARENT_PAGE_ID
from app.ingestion.page_fetcher import PageFetcher


def main():
    fetcher = PageFetcher()

    page = fetcher.get_page(CONFLUENCE_PARENT_PAGE_ID)

    print("=" * 60)
    print(f"Title      : {page['title']}")
    print(f"Page ID    : {page['id']}")
    print(f"Version    : {page['version']['number']}")
    print(f"Space Key  : {page['space']['key']}")
    print(f"Body Length: {len(page['body']['storage']['value'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()