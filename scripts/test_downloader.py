from app.core.config import CONFLUENCE_PARENT_PAGE_ID
from app.ingestion.downloader import PageDownloader
from app.ingestion.hierarchy import HierarchyBuilder


def main():

    builder = HierarchyBuilder()
    downloader = PageDownloader()

    root = builder.build_tree(CONFLUENCE_PARENT_PAGE_ID)

    downloader.download_tree(root)

    print("\nDownload Complete.")


if __name__ == "__main__":
    main()