"""Content-based filtering using CountVectorizer + cosine similarity on TMDB tags."""

import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer


class ContentBasedRecommender:
    """
    Builds a tag vector (genres + keywords + top-3 cast + director + overview)
    for each movie, then finds nearest neighbours via cosine similarity.
    """

    def __init__(self):
        self.movies: pd.DataFrame | None = None
        self.similarity: np.ndarray | None = None
        self._ps = PorterStemmer()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_list(self, obj, n: int | None = None) -> list[str]:
        try:
            items = [i["name"] for i in ast.literal_eval(obj)]
            return items[:n] if n else items
        except Exception:
            return []

    def _extract_director(self, obj) -> list[str]:
        try:
            for person in ast.literal_eval(obj):
                if person.get("job") == "Director":
                    return [person["name"].replace(" ", "")]
        except Exception:
            pass
        return []

    def _stem(self, text: str) -> str:
        return " ".join(self._ps.stem(w) for w in text.split())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, tmdb_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Train on the merged TMDB dataframe.
        Expected columns: id, title, overview, genres, keywords, cast, crew
        """
        df = tmdb_df[["id", "title", "overview", "genres", "keywords", "cast", "crew"]].copy()
        df.dropna(subset=["overview", "genres", "keywords", "cast", "crew"], inplace=True)

        df["genres"] = df["genres"].apply(self._extract_list)
        df["keywords"] = df["keywords"].apply(self._extract_list)
        df["cast"] = df["cast"].apply(lambda x: self._extract_list(x, n=3))
        df["crew"] = df["crew"].apply(self._extract_director)
        df["overview"] = df["overview"].apply(lambda x: str(x).split())

        # Remove spaces so multi-word names stay as one token
        for col in ["genres", "keywords", "cast", "crew"]:
            df[col] = df[col].apply(lambda x: [i.replace(" ", "") for i in x])

        df["tags"] = (
            df["overview"] + df["genres"] + df["keywords"] + df["cast"] + df["crew"]
        )
        df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())
        df["tags"] = df["tags"].apply(self._stem)

        self.movies = df[["id", "title", "tags"]].reset_index(drop=True)

        cv = CountVectorizer(max_features=5000, stop_words="english")
        vectors = cv.fit_transform(self.movies["tags"]).toarray()
        self.similarity = cosine_similarity(vectors)

        return self

    def recommend(self, title: str, n: int = 10) -> pd.DataFrame:
        """
        Return a DataFrame with columns [title, movie_id, similarity_score]
        for the top-N most similar movies.
        """
        if self.movies is None:
            raise RuntimeError("Call fit() before recommend().")

        matches = self.movies[self.movies["title"] == title]
        if matches.empty:
            # Fallback: case-insensitive partial match
            matches = self.movies[
                self.movies["title"].str.contains(title, case=False, na=False)
            ]
        if matches.empty:
            return pd.DataFrame(columns=["title", "movie_id", "similarity_score"])

        idx = matches.index[0]
        scores = sorted(
            enumerate(self.similarity[idx]), key=lambda x: x[1], reverse=True
        )[1 : n + 1]

        rows = []
        for i, score in scores:
            row = self.movies.iloc[i]
            rows.append(
                {
                    "title": row["title"],
                    "movie_id": int(row["id"]),
                    "similarity_score": round(float(score), 4),
                }
            )
        return pd.DataFrame(rows)

    def avg_similarity_score(self, n: int = 10, sample: int = 200) -> float:
        """
        Compute the average cosine similarity score of top-N recommendations
        across a random sample of movies. Used as a quality metric.
        """
        if self.similarity is None:
            return 0.0
        rng = np.random.default_rng(42)
        indices = rng.choice(len(self.movies), min(sample, len(self.movies)), replace=False)
        all_scores: list[float] = []
        for idx in indices:
            top = sorted(
                enumerate(self.similarity[idx]), key=lambda x: x[1], reverse=True
            )[1 : n + 1]
            all_scores.extend(s for _, s in top)
        return round(float(np.mean(all_scores)), 4) if all_scores else 0.0

    def get_movie_titles(self) -> list[str]:
        return self.movies["title"].tolist() if self.movies is not None else []
