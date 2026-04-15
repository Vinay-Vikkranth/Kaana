# seed_runner.py — One-time seeding script called at startup from main.py

import sys
from seed_data import RESTAURANTS
from models import RestaurantEncoder
from vector_store import VectorStore


def seed(vector_store: VectorStore | None = None) -> VectorStore:
    """
    Encode all 50 restaurants and upsert them into Qdrant.
    If vector_store is None, creates a new in-memory instance.
    Returns the populated VectorStore.
    """
    print("Seeding vector store...")

    if vector_store is None:
        vector_store = VectorStore()

    encoder = RestaurantEncoder()

    for i, restaurant in enumerate(RESTAURANTS):
        vector = encoder.encode(restaurant)
        vector_store.upsert_restaurant(restaurant, vector)
        if (i + 1) % 10 == 0:
            print(f"  Encoded {i + 1}/{len(RESTAURANTS)} restaurants...")

    count = vector_store.count()
    print(f"Loaded {count} restaurants into vector store.")
    return vector_store


if __name__ == "__main__":
    # Can also be run standalone: python seed_runner.py
    store = seed()
    print("Done. projection_weights.pt saved.")
    sys.exit(0)
