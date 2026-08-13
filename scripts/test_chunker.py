from app.processing.chunker import MarkdownChunker


def main():

    chunker = MarkdownChunker()

    chunker.process_all()


if __name__ == "__main__":
    main()