# vector_store.py — Qdrant in-memory vector store wrapper

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "restaurants"
VECTOR_DIM = 384


class VectorStore:
    def __init__(self):
        # Fully in-memory: no Docker, no server, no disk persistence needed for POC
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )

    def upsert_restaurant(self, restaurant: dict, vector: np.ndarray) -> None:
        """Store a restaurant vector alongside its full data as payload."""
        # Extract numeric id from "rest_001" → 1
        point_id = int(restaurant["id"].split("_")[1])
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=restaurant,
                )
            ],
        )

    def query(self, user_vector: np.ndarray, top_k: int = 20) -> list:
        """
        Return top_k closest restaurants by cosine similarity.
        Results include payload AND the stored vector (needed for score_match).
        """
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=user_vector.tolist(),
            limit=top_k,
            with_vectors=True,
            with_payload=True,
        )
        return results

    def get_all(self) -> list:
        """Return all restaurant payloads (for the /restaurants endpoint)."""
        results, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        return [r.payload for r in results]

    def count(self) -> int:
        info = self.client.get_collection(COLLECTION_NAME)
        return info.points_count


# ---------------------------------------------------------------------------
# Hard filters applied after vector search
# ---------------------------------------------------------------------------

def is_open(restaurant: dict, current_time: int) -> bool:
    hours = restaurant.get("open_hours", {})
    open_h = hours.get("open", 0)
    close_h = hours.get("close", 24)
    if close_h > open_h:
        return open_h <= current_time < close_h
    # Handles overnight restaurants (rare, but safe to include)
    return current_time >= open_h or current_time < close_h


def apply_hard_filters(results: list, user_profile: dict) -> list:
    """
    Post-query filters that remove restaurants violating hard constraints:
      - price_range > price_max
      - restaurant is currently closed
    """
    price_max = user_profile.get("price_max", 4)
    current_time = user_profile.get("current_time", 12)

    filtered = []
    for result in results:
        restaurant = result.payload
        if restaurant.get("price_range", 4) > price_max:
            continue
        if not is_open(restaurant, current_time):
            continue
        filtered.append(result)
    return filtered
