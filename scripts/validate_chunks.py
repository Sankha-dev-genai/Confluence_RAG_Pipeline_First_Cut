import json
from pathlib import Path
from statistics import mean


CHUNK_DIR = Path("data/chunks")


REQUIRED_FIELDS = {
    "chunk_id",
    "page_id",
    "title",
    "section",
    "heading_path",
    "source_url",
    "text",
    "word_count",
    "character_count",
    "chunk_number",
    "total_chunks",
}


def load_chunks():
    files = sorted(CHUNK_DIR.glob("*.json"))

    page_count = 0
    chunk_count = 0

    word_counts = []

    duplicate_ids = set()
    seen_ids = set()

    empty_chunks = []
    missing_fields = []
    numbering_errors = []

    for file in files:

        page_count += 1

        with open(file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        expected_total = len(chunks)

        for index, chunk in enumerate(chunks, start=1):

            chunk_count += 1

            chunk_id = chunk.get("chunk_id")

            if chunk_id in seen_ids:
                duplicate_ids.add(chunk_id)

            seen_ids.add(chunk_id)

            missing = REQUIRED_FIELDS - set(chunk.keys())

            if missing:
                missing_fields.append(
                    (chunk_id, sorted(missing))
                )

            if not chunk.get("text", "").strip():
                empty_chunks.append(chunk_id)

            if (
                chunk.get("chunk_number") != index
                or chunk.get("total_chunks") != expected_total
            ):
                numbering_errors.append(chunk_id)

            word_counts.append(
                chunk.get("word_count", 0)
            )

    return {
        "pages": page_count,
        "chunks": chunk_count,
        "avg_words": round(mean(word_counts), 2)
        if word_counts
        else 0,
        "min_words": min(word_counts)
        if word_counts
        else 0,
        "max_words": max(word_counts)
        if word_counts
        else 0,
        "duplicate_ids": duplicate_ids,
        "missing_fields": missing_fields,
        "empty_chunks": empty_chunks,
        "numbering_errors": numbering_errors,
    }


def print_report(report):

    print("\n" + "=" * 65)
    print("CHUNK VALIDATION REPORT")
    print("=" * 65)

    print(f"Pages Processed      : {report['pages']}")
    print(f"Chunks Created       : {report['chunks']}")
    print(f"Average Words        : {report['avg_words']}")
    print(f"Minimum Words        : {report['min_words']}")
    print(f"Maximum Words        : {report['max_words']}")

    print("-" * 65)

    print(
        f"Duplicate Chunk IDs  : {len(report['duplicate_ids'])}"
    )
    print(
        f"Missing Fields       : {len(report['missing_fields'])}"
    )
    print(
        f"Empty Chunks         : {len(report['empty_chunks'])}"
    )
    print(
        f"Numbering Errors     : {len(report['numbering_errors'])}"
    )

    passed = (
        len(report["duplicate_ids"]) == 0
        and len(report["missing_fields"]) == 0
        and len(report["empty_chunks"]) == 0
        and len(report["numbering_errors"]) == 0
    )

    print("-" * 65)

    if passed:
        print("STATUS               : PASS")
    else:
        print("STATUS               : FAIL")

    print("=" * 65)


if __name__ == "__main__":

    report = load_chunks()

    print_report(report)