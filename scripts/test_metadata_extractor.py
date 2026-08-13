from app.processing.metadata_extractor import MetadataExtractor


def main() -> None:
    extractor = MetadataExtractor()
    extractor.extract_all()


if __name__ == "__main__":
    main()