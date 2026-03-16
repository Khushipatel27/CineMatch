"""Collaborative filtering using scipy SVD on MovieLens ratings — no scikit-surprise needed."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class CollaborativeRecommender:
    """
    SVD-based collaborative filter trained on MovieLens ratings.
    Uses scipy sparse SVD — no scikit-surprise dependency required.

    Metrics reported after training:
        rmse  – Root Mean Squared Error on 20% held-out test set
        mae   – Mean Absolute Error on 20% held-out test set
    """

    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.ratings: pd.DataFrame | None = None
        self.rmse: float | None = None
        self.mae: float | None = None
        self._all_movie_ids: np.ndarray | None = None
        self._movie_title_map: dict[int, str] = {}
        self._user_idx: dict = {}
        self._movie_idx: dict = {}
        self._idx_movie: dict = {}
        self._U: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._Vt: np.ndarray | None = None
        self._user_mean: pd.Series | None = None
        self._global_mean: float = 3.5

    def fit(
        self,
        ratings_df: pd.DataFrame,
        ml_movies_df: pd.DataFrame | None = None,
    ) -> "CollaborativeRecommender":
        self.ratings = ratings_df[["userId", "movieId", "rating"]].copy()
        self._global_mean = float(self.ratings["rating"].mean())

        train_df, test_df = train_test_split(
            self.ratings, test_size=0.2, random_state=self.random_state
        )

        # Build index mappings
        users = sorted(self.ratings["userId"].unique())
        movies = sorted(self.ratings["movieId"].unique())
        self._user_idx = {u: i for i, u in enumerate(users)}
        self._movie_idx = {m: i for i, m in enumerate(movies)}
        self._idx_movie = {i: m for m, i in self._movie_idx.items()}
        self._all_movie_ids = np.array(movies)

        # Per-user mean from training data (for bias removal)
        self._user_mean = (
            train_df.groupby("userId")["rating"].mean()
            .reindex(users)
            .fillna(self._global_mean)
        )

        # Build sparse user-item matrix (mean-centered)
        u_idx = train_df["userId"].map(self._user_idx).values
        m_idx = train_df["movieId"].map(self._movie_idx).values
        means = train_df["userId"].map(self._user_mean).values
        vals = train_df["rating"].values - means

        valid = ~(np.isnan(u_idx) | np.isnan(m_idx) | np.isnan(vals))
        mat = csr_matrix(
            (vals[valid].astype(float),
             (u_idx[valid].astype(int), m_idx[valid].astype(int))),
            shape=(len(users), len(movies)),
        )

        # SVD
        k = min(self.n_factors, min(len(users), len(movies)) - 1)
        U, sigma, Vt = svds(mat, k=k)
        order = np.argsort(sigma)[::-1]
        self._U = U[:, order]
        self._sigma = sigma[order]
        self._Vt = Vt[order, :]

        # Evaluate on test set
        preds, actuals = self._batch_predict(test_df)
        self.rmse = round(float(np.sqrt(np.mean((preds - actuals) ** 2))), 4)
        self.mae = round(float(np.mean(np.abs(preds - actuals))), 4)

        if ml_movies_df is not None:
            self._movie_title_map = dict(
                zip(ml_movies_df["movieId"], ml_movies_df["title"])
            )

        return self

    def _batch_predict(self, df: pd.DataFrame):
        """Vectorised predictions for a DataFrame of (userId, movieId, rating)."""
        u_idx = df["userId"].map(self._user_idx)
        m_idx = df["movieId"].map(self._movie_idx)
        means = df["userId"].map(self._user_mean).fillna(self._global_mean).values
        actuals = df["rating"].values

        preds = np.full(len(df), self._global_mean)
        valid = u_idx.notna() & m_idx.notna()
        if valid.any():
            ui = u_idx[valid].astype(int).values
            mi = m_idx[valid].astype(int).values
            # U[ui] * sigma element-wise, then dot Vt columns
            latent = self._U[ui] * self._sigma          # (N, k)
            item_factors = self._Vt[:, mi].T             # (N, k)
            residuals = (latent * item_factors).sum(axis=1)
            preds[valid] = np.clip(means[valid] + residuals, 0.5, 5.0)
        return preds, actuals

    def _predict_rating(self, user_id: int, movie_id: int) -> float:
        mean = float(self._user_mean.get(user_id, self._global_mean))
        if user_id not in self._user_idx or movie_id not in self._movie_idx:
            return mean
        ui = self._user_idx[user_id]
        mi = self._movie_idx[movie_id]
        residual = float((self._U[ui] * self._sigma).dot(self._Vt[:, mi]))
        return float(np.clip(mean + residual, 0.5, 5.0))

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if self.ratings is None:
            raise RuntimeError("Call fit() before recommend().")

        rated_ids = set(
            self.ratings.loc[self.ratings["userId"] == user_id, "movieId"]
        )
        unrated = [mid for mid in self._all_movie_ids if mid not in rated_ids]

        # Vectorised prediction for all unrated movies at once
        mean = float(self._user_mean.get(user_id, self._global_mean))
        if user_id in self._user_idx:
            ui = self._user_idx[user_id]
            m_indices = np.array(
                [self._movie_idx[mid] for mid in unrated if mid in self._movie_idx]
            )
            valid_movies = [mid for mid in unrated if mid in self._movie_idx]
            if len(m_indices):
                residuals = (self._U[ui] * self._sigma).dot(self._Vt[:, m_indices])
                scores = np.clip(mean + residuals, 0.5, 5.0)
            else:
                valid_movies, scores = [], np.array([])
            # Movies not in index get mean score
            unknown = [(mid, mean) for mid in unrated if mid not in self._movie_idx]
            pairs = list(zip(valid_movies, scores.tolist())) + unknown
        else:
            pairs = [(mid, mean) for mid in unrated]

        pairs.sort(key=lambda x: x[1], reverse=True)

        rows = []
        for movie_id, pred_rating in pairs[:n]:
            title = self._movie_title_map.get(int(movie_id), f"Movie {movie_id}")
            rows.append({
                "title": title,
                "movie_id": int(movie_id),
                "predicted_rating": round(pred_rating, 3),
            })
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
