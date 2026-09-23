import pandas as pd

from src.recommender.model import MovieRecommender


def test_recommender_returns_movies_for_valid_input():
    movie = "The Shawshank Redemption"
    recommender = MovieRecommender()
    recommendations = recommender.recommend(movie, top_n=5)

    assert isinstance(recommendations, list)
    assert len(recommendations) == 5
    assert all(isinstance(item, dict) for item in recommendations)
    assert recommendations[0]["title"] != movie


def test_recommender_handles_unknown_movie_gracefully():
    recommender = MovieRecommender()
    result = recommender.recommend("Definitely not a movie title", top_n=5)

    assert isinstance(result, list)
    assert len(result) == 0


def test_pipeline_builds_model_from_dataset():
    df = pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "genre": ["Drama", "Comedy", "Drama"],
            "overview": ["A prison story.", "A funny story.", "A family story."],
            "original_language": ["en", "en", "fr"],
        }
    )
    recommender = MovieRecommender()
    result = recommender.build_from_dataframe(df)

    assert result is not None
    assert hasattr(recommender, "movies_df")
    assert hasattr(recommender, "similarity_matrix")
