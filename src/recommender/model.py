from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "src" / "data" / "top10K-TMDB-movies.csv"
DEFAULT_MOVIES_PATH = ROOT_DIR / "src" / "models" / "movies.pkl"
DEFAULT_SIMILARITY_PATH = ROOT_DIR / "src" / "models" / "similarity.pkl"


class MovieRecommender:
    """Recommend movies using TF-IDF over title, genre, language, and overview."""

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        movies_path: str | Path = DEFAULT_MOVIES_PATH,
        similarity_path: str | Path = DEFAULT_SIMILARITY_PATH,
    ) -> None:
        self.data_path = Path(data_path)
        self.movies_path = Path(movies_path)
        self.similarity_path = Path(similarity_path)
        self.movies_df: pd.DataFrame | None = None
        self.similarity_matrix: Any | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self._load_or_train()

    def _load_or_train(self) -> None:
        if self.movies_path.exists() and self.similarity_path.exists():
            try:
                with self.movies_path.open("rb") as movie_file:
                    self.movies_df = pickle.load(movie_file)
                with self.similarity_path.open("rb") as sim_file:
                    self.similarity_matrix = pickle.load(sim_file)
                return
            except (
                pickle.PickleError,
                EOFError,
                AttributeError,
                ImportError,
                ValueError,
            ):
                pass

        df = self._read_dataset()
        self.build_from_dataframe(df)
        self.save_model()

    def _read_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        required_columns = {"title", "genre", "overview", "original_language"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        return df

    def _build_text_features(self, row: pd.Series) -> str:
        title = str(row.get("title") or "")
        genre = str(row.get("genre") or "")
        language = str(row.get("original_language") or "")
        overview = str(row.get("overview") or "")
        return " ".join(
            part for part in [title, genre, language, overview] if part
        ).strip()

    def build_from_dataframe(self, df: pd.DataFrame) -> "MovieRecommender":
        working_df = df.copy()
        for column in ["title", "genre", "overview", "original_language"]:
            if column not in working_df.columns:
                working_df[column] = ""
            working_df[column] = working_df[column].fillna("")

        for column in [
            "id",
            "popularity",
            "release_date",
            "vote_average",
            "vote_count",
        ]:
            if column not in working_df.columns:
                if column == "id":
                    working_df[column] = range(len(working_df))
                else:
                    working_df[column] = 0

        working_df["combined_text"] = working_df.apply(
            self._build_text_features, axis=1
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2
        )
        text_matrix = self.vectorizer.fit_transform(working_df["combined_text"])
        self.similarity_matrix = cosine_similarity(text_matrix)

        metadata_columns = [
            "id",
            "title",
            "genre",
            "original_language",
            "overview",
            "popularity",
            "release_date",
            "vote_average",
            "vote_count",
        ]
        self.movies_df = working_df[metadata_columns].copy()
        self.movies_df["popularity"] = pd.to_numeric(
            self.movies_df["popularity"], errors="coerce"
        ).fillna(0.0)
        self.movies_df["vote_average"] = pd.to_numeric(
            self.movies_df["vote_average"], errors="coerce"
        ).fillna(0.0)
        self.movies_df["vote_count"] = pd.to_numeric(
            self.movies_df["vote_count"], errors="coerce"
        ).fillna(0)
        self.movies_df = self.movies_df.reset_index(drop=True)
        return self

    def recommend(self, movie_title: str, top_n: int = 5) -> list[dict[str, Any]]:
        if self.movies_df is None or self.similarity_matrix is None:
            raise ValueError("Model has not been trained yet.")

        if not movie_title or not str(movie_title).strip():
            return []

        cleaned_title = str(movie_title).strip()
        movie_matches = self.movies_df[
            self.movies_df["title"].str.lower() == cleaned_title.lower()
        ]
        if movie_matches.empty:
            return []

        movie_index = movie_matches.index[0]
        distances = sorted(
            enumerate(self.similarity_matrix[movie_index]),
            key=lambda item: item[1],
            reverse=True,
        )

        recommendations: list[dict[str, Any]] = []
        for idx, score in distances[1 : top_n + 1]:
            if idx == movie_index:
                continue
            row = self.movies_df.iloc[idx]
            recommendations.append(
                {
                    "title": str(row["title"]),
                    "genre": str(row["genre"]),
                    "language": str(row["original_language"]),
                    "overview": str(row["overview"]),
                    "vote_average": float(
                        pd.to_numeric(row["vote_average"], errors="coerce") or 0.0
                    ),
                    "popularity": float(
                        pd.to_numeric(row["popularity"], errors="coerce") or 0.0
                    ),
                    "score": float(score),
                }
            )

        return recommendations

    def save_model(self) -> None:
        if self.movies_df is None:
            raise ValueError("No movie dataframe available to save.")
        if self.similarity_matrix is None:
            raise ValueError("No similarity matrix available to save.")

        with self.movies_path.open("wb") as movie_file:
            pickle.dump(self.movies_df, movie_file)
        with self.similarity_path.open("wb") as similarity_file:
            pickle.dump(self.similarity_matrix, similarity_file)


def load_recommender() -> MovieRecommender:
    return MovieRecommender()
