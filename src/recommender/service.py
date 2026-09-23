from __future__ import annotations

from functools import lru_cache
from typing import Any

from .model import MovieRecommender


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    return MovieRecommender()


def recommend_movies(movie_title: str, top_n: int = 5) -> list[dict[str, Any]]:
    return get_recommender().recommend(movie_title, top_n=top_n)
