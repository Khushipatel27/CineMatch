"""Hybrid recommender: weighted combination of content-based + collaborative scores."""

import pandas as pd
import numpy as np

from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender


class HybridRecommender:
    """
    Combines a ContentBasedRecommender and a CollaborativeRecommender.

    Both score columns are min-max normalised to [0, 1] before weighting so
    that neither dominates purely due to scale differences.

    hybrid_score = content_weight * norm_content + (1 - content_weight) * norm_collab
    """

    def __init__(
        self,
        content_rec: ContentBasedRecommender,
        collab_rec: CollaborativeRecommender,
        default_content_weight: float = 0.5,
    ):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.default_content_weight = default_content_weight

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _min_max(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        movie_title: str,
        user_id: int,
        n: int = 10,
        content_weight: float | None = None,
    ) -> pd.DataFrame:
        """
        Return top-N hybrid recommendations.

        Returns a DataFrame with columns:
            title, hybrid_score, content_score, collab_score
        """
        cw = content_weight if content_weight is not None else self.default_content_weight
        collw = 1.0 - cw

        # Fetch a generous pool from each source
        pool = max(n * 5, 50)
        content_df = self.content_rec.recommend(movie_title, n=pool)
        collab_df = self.collab_rec.recommend(user_id, n=pool)

        # ---- handle degenerate cases ----
        if content_df.empty and collab_df.empty:
            return pd.DataFrame(columns=["title", "hybrid_score", "content_score", "collab_score"])
        if content_df.empty:
            collab_df = collab_df.copy()
            collab_df["hybrid_score"] = self._min_max(collab_df["predicted_rating"])
            collab_df["content_score"] = 0.0
            collab_df["collab_score"] = collab_df["hybrid_score"]
            return collab_df[["title", "hybrid_score", "content_score", "collab_score"]].head(n)
        if collab_df.empty:
            content_df = content_df.copy()
            content_df["hybrid_score"] = self._min_max(content_df["similarity_score"])
            content_df["collab_score"] = 0.0
            content_df["content_score"] = content_df["hybrid_score"]
            return content_df[["title", "hybrid_score", "content_score", "collab_score"]].head(n)

        # ---- normalise each score pool ----
        content_df = content_df.copy()
        content_df["norm_content"] = self._min_max(content_df["similarity_score"])

        collab_df = collab_df.copy()
        collab_df["norm_collab"] = self._min_max(collab_df["predicted_rating"])

        # ---- merge on title (case-insensitive) ----
        content_df["_key"] = content_df["title"].str.lower().str.strip()
        collab_df["_key"] = collab_df["title"].str.lower().str.strip()

        merged = pd.merge(
            content_df[["title", "norm_content", "_key"]],
            collab_df[["norm_collab", "_key"]],
            on="_key",
            how="outer",
        )
        merged["norm_content"] = merged["norm_content"].fillna(0.0)
        merged["norm_collab"] = merged["norm_collab"].fillna(0.0)

        merged["hybrid_score"] = (
            cw * merged["norm_content"] + collw * merged["norm_collab"]
        ).round(4)
        merged = merged.sort_values("hybrid_score", ascending=False).head(n)

        merged = merged.rename(
            columns={"norm_content": "content_score", "norm_collab": "collab_score"}
        )
        return merged[["title", "hybrid_score", "content_score", "collab_score"]].reset_index(
            drop=True
        )
