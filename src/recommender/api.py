from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import recommend_movies

app = FastAPI(
    title="TMDB Movie Recommender API",
    version="1.0.0",
    description="Movie recommendations built from TF-IDF similarity.",
)


class RecommendationRequest(BaseModel):
    movie_title: str = Field(..., min_length=1)
    top_n: int = Field(default=5, ge=1, le=10)


class RecommendationResponse(BaseModel):
    movie_title: str
    recommendations: list[dict]


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_movie(payload: RecommendationRequest) -> RecommendationResponse:
    movie_title = payload.movie_title.strip()
    recommendations = recommend_movies(movie_title, top_n=payload.top_n)
    if not recommendations:
        raise HTTPException(
            status_code=404, detail=f"No recommendations found for '{movie_title}'"
        )
    return RecommendationResponse(
        movie_title=movie_title, recommendations=recommendations
    )
