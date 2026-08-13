from app.core.config import CONFLUENCE_PARENT_PAGE_ID
from app.ingestion.hierarchy import HierarchyBuilder
from app.ingestion.tree_utils import print_tree


def main():

    builder = HierarchyBuilder()

    root = builder.build_tree(CONFLUENCE_PARENT_PAGE_ID)

    print("\nConfluence Hierarchy\n")
    print_tree(root)


if __name__ == "__main__":
    main()