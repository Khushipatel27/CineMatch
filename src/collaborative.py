"""Collaborative filtering using Surprise SVD on MovieLens ratings."""

import pandas as pd
import numpy as np

try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
    from surprise import accuracy as surprise_accuracy

    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False


class CollaborativeRecommender:
    """
    SVD-based collaborative filter trained on MovieLens ratings.

    Metrics reported after training:
        rmse  – Root Mean Squared Error on 20 % held-out test set
        mae   – Mean Absolute Error on 20 % held-out test set
    """

    def __init__(self, n_factors: int = 100, n_epochs: int = 20, random_state: int = 42):
        if not SURPRISE_AVAILABLE:
            raise ImportError(
                "scikit-surprise is required. Install with: pip install scikit-surprise"
            )
        self.model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
        self.ratings: pd.DataFrame | None = None
        self.rmse: float | None = None
        self.mae: float | None = None
        self._all_movie_ids: np.ndarray | None = None
        self._movie_title_map: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        ratings_df: pd.DataFrame,
        ml_movies_df: pd.DataFrame | None = None,
    ) -> "CollaborativeRecommender":
        """
        Train on a MovieLens-style ratings DataFrame.

        Args:
            ratings_df:   columns [userId, movieId, rating]
            ml_movies_df: optional MovieLens movies.csv columns [movieId, title]
                          used to map IDs → human-readable titles
        """
        self.ratings = ratings_df[["userId", "movieId", "rating"]].copy()

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.ratings, reader)

        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        self.model.fit(trainset)

        predictions = self.model.test(testset)
        self.rmse = round(float(surprise_accuracy.rmse(predictions, verbose=False)), 4)
        self.mae = round(float(surprise_accuracy.mae(predictions, verbose=False)), 4)

        self._all_movie_ids = self.ratings["movieId"].unique()

        if ml_movies_df is not None:
            self._movie_title_map = dict(
                zip(ml_movies_df["movieId"], ml_movies_df["title"])
            )

        return self

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """
        Return a DataFrame with columns [title, movie_id, predicted_rating]
        for the top-N unrated movies predicted for user_id.
        """
        if self.ratings is None:
            raise RuntimeError("Call fit() before recommend().")

        rated_ids = set(
            self.ratings.loc[self.ratings["userId"] == user_id, "movieId"]
        )
        unrated = [mid for mid in self._all_movie_ids if mid not in rated_ids]

        preds = [
            (mid, round(float(self.model.predict(user_id, mid).est), 3))
            for mid in unrated
        ]
        preds.sort(key=lambda x: x[1], reverse=True)

        rows = []
        for movie_id, pred_rating in preds[:n]:
            title = self._movie_title_map.get(int(movie_id), f"Movie {movie_id}")
            rows.append(
                {
                    "title": title,
                    "movie_id": int(movie_id),
                    "predicted_rating": pred_rating,
                }
            )
        return pd.DataFrame(rows)

    def get_user_ids(self) -> list[int]:
        if self.ratings is None:
            return []
        return sorted(self.ratings["userId"].unique().tolist())

    def n_users(self) -> int:
        return int(self.ratings["userId"].nunique()) if self.ratings is not None else 0

    def n_movies(self) -> int:
        return int(self.ratings["movieId"].nunique()) if self.ratings is not None else 0

    def n_ratings(self) -> int:
        return len(self.ratings) if self.ratings is not None else 0
