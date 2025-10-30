"""Script to seed the database with test data."""

import asyncio

import httpx


async def seed_data() -> None:
    """Seed the database with test data."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        # Create a library
        print("Creating library...")
        library_response = await client.post(
            f"{base_url}/api/v1/libraries/",
            json={
                "name": "Sample Library",
                "description": "A sample library with test data",
                "index_type": "brute_force",
            },
        )
        library = library_response.json()
        library_id = library["id"]
        print(f"Created library: {library_id}")

        # Create a document
        print("Creating document...")
        document_response = await client.post(
            f"{base_url}/api/v1/libraries/{library_id}/documents/",
            json={
                "name": "Sample Document",
                "metadata": {"author": "Test Author"},
            },
        )
        document = document_response.json()
        document_id = document["id"]
        print(f"Created document: {document_id}")

        # Create chunks
        print("Creating chunks...")
        sample_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text.",
        ]

        for text in sample_texts:
            chunk_response = await client.post(
                f"{base_url}/api/v1/documents/{document_id}/chunks/",
                json={"content": text},
            )
            chunk = chunk_response.json()
            print(f"Created chunk: {chunk['id']}")

        # Build index
        print("Building index...")
        await client.post(f"{base_url}/api/v1/libraries/{library_id}/index")
        print("Index built successfully!")

        print("\nSeed data created successfully!")


if __name__ == "__main__":
    asyncio.run(seed_data())
