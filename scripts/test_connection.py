from app.ingestion.client import ConfluenceClient


def main():
    client = ConfluenceClient()

    response = client.get(
        "/wiki/rest/api/space"
    )

    results = response.get("results", [])

    print("=" * 50)
    print("Connected Successfully")
    print(f"Spaces Found : {len(results)}")
    print("=" * 50)


if __name__ == "__main__":
    main()