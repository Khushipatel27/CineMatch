"""Shared helpers: API calls, data loading, path utilities."""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"


def fetch_poster(movie_id: int) -> str:
    """Return poster URL for a TMDB movie_id, or a placeholder on failure."""
    if not TMDB_API_KEY:
        return PLACEHOLDER_POSTER
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        path = resp.json().get("poster_path", "")
        if path:
            return f"{TMDB_IMAGE_BASE}{path}"
    except requests.exceptions.RequestException:
        pass
    return PLACEHOLDER_POSTER


def load_tmdb_data(
    movies_path: str = "data/tmdb_5000_movies.csv",
    credits_path: str = "data/tmdb_5000_credits.csv",
) -> pd.DataFrame:
    """Load and merge TMDB movies + credits datasets."""
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    # Keep only cast/crew from credits to avoid title_x/title_y collision
    id_col = "movie_id" if "movie_id" in credits.columns else "id"
    credits = credits[[id_col, "cast", "crew"]].rename(columns={id_col: "id"})
    merged = movies.merge(credits, on="id")
    return merged


def load_ratings(ratings_path: str = "data/ratings_small.csv") -> pd.DataFrame | None:
    """Load MovieLens ratings. Returns None if file is missing."""
    if not os.path.exists(ratings_path):
        return None
    return pd.read_csv(ratings_path)


def load_ml_movies(ml_movies_path: str = "data/ml_movies.csv") -> pd.DataFrame | None:
    """Load MovieLens movies (movieId, title) for title lookup. Returns None if missing."""
    if not os.path.exists(ml_movies_path):
        return None
    return pd.read_csv(ml_movies_path)
