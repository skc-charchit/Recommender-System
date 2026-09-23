from __future__ import annotations

from .model import MovieRecommender


def train_model() -> MovieRecommender:
    recommender = MovieRecommender()
    recommender.save_model()
    print(
        f"Model saved to {recommender.movies_path} and {recommender.similarity_path}."
    )
    return recommender
