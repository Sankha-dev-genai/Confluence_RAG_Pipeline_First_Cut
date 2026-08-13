from app.ingestion.models import ConfluencePage


def print_tree(node: ConfluencePage):

    indent = "    " * node.level

    print(f"{indent}- {node.title}")

    for child in node.children:
        print_tree(child)