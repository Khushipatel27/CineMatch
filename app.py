"""CineMatch — friendly movie recommender for everyone."""

import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils import fetch_poster, load_tmdb_data, load_ratings, load_ml_movies

PLACEHOLDER = "https://via.placeholder.com/300x450/1a1a2e/c3a6ff?text=🎬"

def safe_poster(row):
    """Return poster URL — uses TMDB id if present, otherwise placeholder."""
    mid = row.get("movie_id") if isinstance(row, dict) else (
        row["movie_id"] if "movie_id" in row.index else None
    )
    if mid is None:
        return PLACEHOLDER
    return fetch_poster(mid)
from src.content_based import ContentBasedRecommender

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Find Your Next Movie",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 20px; padding: 40px 48px; margin-bottom: 28px; text-align: center;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; color: #fff; margin: 0 0 10px; letter-spacing: -1px; }
.hero p  { font-size: 1.05rem; color: #c0b8f0; margin: 0 0 20px; }
.stat-row { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
.stat-pill {
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px; padding: 6px 18px; font-size: 0.83rem; color: #e8e3ff;
}

/* Movie cards */
.movie-card {
    background: #16162a; border-radius: 14px; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s; cursor: pointer;
}
.movie-card:hover { transform: translateY(-5px); box-shadow: 0 12px 32px rgba(0,0,0,0.6); }
.card-body { padding: 10px 10px 12px; }
.card-title {
    font-size: 0.78rem; font-weight: 600; color: #e8e8ff;
    text-align: center; margin-bottom: 6px; line-height: 1.35; min-height: 2.5em;
}
.badge {
    display: block; text-align: center; font-size: 0.73rem;
    font-weight: 700; padding: 4px 0; border-radius: 8px; margin-top: 4px;
}
.badge-green  { background: linear-gradient(90deg,#11998e,#38ef7d); color: #000; }
.badge-orange { background: linear-gradient(90deg,#f7971e,#ffd200); color: #000; }
.badge-purple { background: linear-gradient(90deg,#7f53ac,#647dee); color: #fff; }
.badge-star   { background: linear-gradient(90deg,#f7971e,#ffd200); color: #000; }

/* Section headers */
.sec-head { font-size: 1.3rem; font-weight: 700; color: #e0daff; margin-bottom: 4px; }
.sec-sub  { font-size: 0.85rem; color: #888; margin-bottom: 20px; }

/* Taste profile card */
.taste-card {
    background: #1e1e38; border: 1px solid #3a3a6a; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 16px;
}
.taste-card h4 { color: #c3a6ff; margin: 0 0 6px; font-size: 0.95rem; }
.taste-card p  { color: #b0b0d0; font-size: 0.82rem; margin: 0; line-height: 1.5; }

/* Setup step */
.step {
    background: #1a1a2e; border-left: 4px solid #8e54e9;
    border-radius: 0 10px 10px 0; padding: 14px 18px; margin-bottom: 10px;
    font-size: 0.86rem; color: #c8c8e8; line-height: 1.6;
}
.step strong { color: #c3a6ff; }

/* How it works cards */
.how-card {
    background: #1a1a2e; border: 1px solid #2e2e5a;
    border-radius: 14px; padding: 22px; text-align: center;
}
.how-icon  { font-size: 2rem; margin-bottom: 10px; }
.how-title { font-size: 0.95rem; font-weight: 700; color: #e0daff; margin-bottom: 6px; }
.how-desc  { font-size: 0.8rem; color: #888; line-height: 1.55; }

/* Genre chips */
.genre-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 20px; }
.genre-chip {
    background: #2a2a4a; border: 1px solid #4a4a7a; border-radius: 20px;
    padding: 5px 14px; font-size: 0.78rem; color: #c3c3ff; cursor: pointer;
}

/* Metric override */
[data-testid="stMetric"] {
    background: #1a1a2e; border: 1px solid #2e2e5e;
    border-radius: 12px; padding: 14px !important;
}

/* Tab styling */
[data-testid="stTabs"] button { font-size: 0.9rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading movie database…")
def load_content_model():
    tmdb_df = load_tmdb_data()
    rec = ContentBasedRecommender()
    rec.fit(tmdb_df)
    return rec

@st.cache_resource(show_spinner="⭐ Loading ratings data…")
def load_collab_model():
    ratings_df = load_ratings()
    if ratings_df is None:
        return None, "missing_data"
    try:
        from src.collaborative import CollaborativeRecommender
        ml_movies = load_ml_movies()
        rec = CollaborativeRecommender()
        rec.fit(ratings_df, ml_movies_df=ml_movies)
        return rec, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner="🔀 Setting up personalised mix…")
def load_hybrid(_c, _r):
    from src.hybrid import HybridRecommender
    return HybridRecommender(_c, _r)

@st.cache_data
def build_user_profiles(_collab_rec, _ml_movies):
    """Build a plain-English taste description for each user."""
    if _collab_rec is None or _ml_movies is None:
        return {}
    profiles = {}
    ratings = _collab_rec.ratings
    for uid in _collab_rec.get_user_ids():
        user_ratings = ratings[ratings["userId"] == uid]
        top_movies = user_ratings.nlargest(5, "rating")
        titles = []
        for mid in top_movies["movieId"]:
            match = _ml_movies[_ml_movies["movieId"] == mid]
            if not match.empty:
                # strip year from MovieLens title e.g. "Toy Story (1995)"
                t = match.iloc[0]["title"]
                t = t.rsplit("(", 1)[0].strip()
                titles.append(t)
        n = len(user_ratings)
        avg = round(user_ratings["rating"].mean(), 1)
        loved = ", ".join(titles[:3]) if titles else "various movies"
        profiles[uid] = {
            "label": f"Fan #{uid}",
            "movies_rated": n,
            "avg_rating": avg,
            "loved": loved,
            "desc": f"Rated {n} movies · avg ★{avg} · loved: {loved}",
        }
    return profiles


# ── Load everything ────────────────────────────────────────────────────────────
content_rec             = load_content_model()
collab_rec, collab_err  = load_collab_model()
hybrid_rec              = load_hybrid(content_rec, collab_rec) if collab_rec else None
titles                  = content_rec.get_movie_titles()
ml_movies_df            = load_ml_movies() if collab_rec else None
user_profiles           = build_user_profiles(collab_rec, ml_movies_df)
collab_ready            = collab_rec is not None

POPULAR = [
    "The Dark Knight", "Inception", "Interstellar", "The Avengers",
    "Forrest Gump", "The Lion King", "Titanic", "Toy Story",
    "Jurassic Park", "The Matrix", "Pulp Fiction", "Goodfellas",
]
popular_in_db = [m for m in POPULAR if m in titles]


# ── Hero ───────────────────────────────────────────────────────────────────────
n_movies  = len(titles)
n_ratings = collab_rec.n_ratings() if collab_ready else 0

_ratings_pill = f'<span class="stat-pill">⭐ {n_ratings:,} ratings analysed</span>' if collab_ready else ""
st.markdown(f"""
<div class="hero">
  <h1>🎬 CineMatch</h1>
  <p>Tell us one movie you love — we'll find your next favourite.</p>
  <div class="stat-row">
    <span class="stat-pill">🎥 {n_movies:,} movies</span>
    {_ratings_pill}
    <span class="stat-pill">🤖 AI-powered · free · no sign-up</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎥  Find Similar Movies",
    "✨  Recommended For You",
    "🎯  Best of Both Worlds",
    "ℹ️  How It Works",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Find Similar Movies (Content-Based)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown('<p class="sec-head">🎥 Find Similar Movies</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Pick a movie you already love and we\'ll find others just like it — same vibe, genre, cast, and mood.</p>', unsafe_allow_html=True)

    # Use a staging key so we never touch "cb_movie" after the widget renders
    if "lucky_pick" not in st.session_state:
        st.session_state["lucky_pick"] = None

    # Quick-pick buttons — store choice BEFORE selectbox renders
    st.markdown("**⚡ Quick picks — click to select:**")
    qcols = st.columns(len(popular_in_db[:6]))
    for i, m in enumerate(popular_in_db[:6]):
        if qcols[i].button(m, key=f"qp_{i}", use_container_width=True):
            st.session_state["lucky_pick"] = m

    # Surprise me — also stored before selectbox renders
    if st.button("🎲 Surprise me!", key="lucky", help="Pick a random movie for you"):
        st.session_state["lucky_pick"] = random.choice(titles)

    # Resolve index from lucky_pick, then clear it
    if st.session_state["lucky_pick"] and st.session_state["lucky_pick"] in titles:
        default_idx = titles.index(st.session_state["lucky_pick"])
        st.session_state["lucky_pick"] = None
    else:
        default_idx = 0

    st.markdown("---")

    col_sel, col_n = st.columns([5, 1])
    with col_sel:
        selected_movie = st.selectbox(
            "Or type any movie name:",
            titles,
            index=default_idx,
            key="cb_movie",
            help="Start typing to search through 4,800+ movies",
        )
    with col_n:
        n_cb = st.number_input("How many?", 5, 20, 10, key="cb_n",
                               help="Number of similar movies to show")

    go_cb = st.button("🔍 Find Similar Movies", key="btn_cb", type="primary", use_container_width=True)

    if go_cb:
        with st.spinner(f"Finding movies similar to **{selected_movie}**…"):
            recs = content_rec.recommend(selected_movie, n=int(n_cb))

        if recs.empty:
            st.warning("Hmm, we couldn't find matches. Try a different title!")
        else:
            st.success(f"✅ Here are {len(recs)} movies similar to **{selected_movie}**")

            # Show match explanation
            with st.expander("🤔 Why these movies?"):
                st.write(
                    f"We compared **{selected_movie}** to every movie in our database "
                    f"looking at shared genres, cast members, director, and story keywords. "
                    f"The **match %** shows how closely they overlap — "
                    f"higher = more similar vibe."
                )

            cols = st.columns(5)
            for i, row in recs.iterrows():
                with cols[i % 5]:
                    poster = fetch_poster(row["movie_id"])
                    pct = row["similarity_score"] * 100
                    badge_cls = "badge-green" if pct >= 22 else ("badge-orange" if pct >= 17 else "badge-purple")
                    st.markdown(f"""
                    <div class="movie-card">
                      <img src="{poster}" style="width:100%;border-radius:14px 14px 0 0;display:block;aspect-ratio:2/3;object-fit:cover;">
                      <div class="card-body">
                        <div class="card-title">{row['title']}</div>
                        <span class="badge {badge_cls}">{pct:.0f}% similar</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Recommended For You (Collaborative)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown('<p class="sec-head">✨ Recommended For You</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Choose a viewer profile that matches your taste — we\'ll predict which movies they\'d love next.</p>', unsafe_allow_html=True)

    if collab_err == "missing_data":
        # Friendly setup guide — no technical jargon
        st.markdown("""
        <div class="taste-card">
          <h4>📦 One-time setup needed — takes 2 minutes!</h4>
          <p>To give personal recommendations, we need a ratings dataset.
          It's completely free and just 3 steps:</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="step">
          <strong>Step 1 —</strong> Open this link in your browser:<br>
          <a href="https://grouplens.org/datasets/movielens/latest/" target="_blank"
             style="color:#c3a6ff">grouplens.org/datasets/movielens/latest/</a><br>
          Click <strong>"ml-latest-small.zip"</strong> to download it (about 1 MB).
        </div>
        <div class="step">
          <strong>Step 2 —</strong> Open the zip file. Inside you'll see several files.
          Copy two of them into the <code>data/</code> folder of this project:<br>
          &nbsp;&nbsp;• Rename <code>ratings.csv</code> → <code>data/ratings_small.csv</code><br>
          &nbsp;&nbsp;• Rename <code>movies.csv</code> → <code>data/ml_movies.csv</code>
        </div>
        <div class="step">
          <strong>Step 3 —</strong> Close and restart the app
          (<code>streamlit run app.py</code>). That's it!
          This tab will be fully unlocked with 600+ viewer profiles to explore.
        </div>
        """, unsafe_allow_html=True)

    elif collab_err:
        st.error(f"Something went wrong loading ratings: {collab_err}")
    else:
        user_ids = collab_rec.get_user_ids()

        # Build friendly profile labels
        profile_options = {}
        for uid in user_ids:
            p = user_profiles.get(uid, {})
            n  = p.get("movies_rated", "?")
            avg = p.get("avg_rating", "?")
            profile_options[uid] = f"Viewer #{uid}  ·  {n} movies rated  ·  avg ★{avg}"

        st.markdown("#### 👤 Choose a viewer profile")
        st.caption("Each profile is an anonymised real person's movie history from MovieLens.")

        c1, c2 = st.columns([4, 1])
        with c1:
            chosen_label = st.selectbox(
                "Pick a profile:",
                list(profile_options.values()),
                key="cf_label",
                label_visibility="collapsed",
            )
            # Map label back to user_id
            chosen_uid = [k for k, v in profile_options.items() if v == chosen_label][0]

        with c2:
            n_cf = st.number_input("Show", 5, 20, 10, key="cf_n",
                                   help="Number of movie recommendations")

        # Show taste snapshot
        p = user_profiles.get(chosen_uid, {})
        if p.get("loved"):
            st.markdown(f"""
            <div class="taste-card">
              <h4>🍿 This viewer's taste</h4>
              <p>
                Rated <strong>{p['movies_rated']}</strong> movies &nbsp;·&nbsp;
                Average rating <strong>★{p['avg_rating']}</strong><br>
                Previously loved: <em>{p['loved']}</em>
              </p>
            </div>
            """, unsafe_allow_html=True)

        if st.button("✨ Get My Recommendations", key="btn_cf", type="primary"):
            with st.spinner("Finding the best unwatched movies for this profile…"):
                recs = collab_rec.recommend(int(chosen_uid), n=int(n_cf))

            if recs.empty:
                st.warning("Nothing found for this profile — try a different one.")
            else:
                st.success(f"🎉 Top {len(recs)} movies this viewer would probably love:")

                with st.expander("🤔 How do we know they'd like these?"):
                    st.write(
                        "Our AI studied the rating patterns of hundreds of viewers. "
                        "It found people with similar tastes, looked at what they loved "
                        "that this viewer hasn't seen yet, and predicted how they'd rate each one. "
                        "Higher predicted stars = stronger recommendation."
                    )

                card_cols = st.columns(5)
                for i, row in recs.iterrows():
                    with card_cols[i % 5]:
                        poster = safe_poster(row)
                        stars = row["predicted_rating"]
                        st.markdown(f"""
                        <div class="movie-card">
                          <img src="{poster}" style="width:100%;border-radius:14px 14px 0 0;display:block;aspect-ratio:2/3;object-fit:cover;">
                          <div class="card-body">
                            <div class="card-title">{row['title']}</div>
                            <span class="badge badge-star">★ {stars:.1f} predicted</span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

                # Simple chart
                fig = px.bar(
                    recs.sort_values("predicted_rating"),
                    x="predicted_rating", y="title", orientation="h",
                    title="Predicted Enjoyment Score",
                    labels={"predicted_rating": "Predicted ★ (out of 5)", "title": ""},
                    color="predicted_rating",
                    color_continuous_scale="YlOrRd",
                    range_x=[0, 5],
                )
                fig.update_layout(
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", height=320,
                    margin=dict(l=0, r=10, t=40, b=0),
                )
                fig.update_xaxes(gridcolor="#2a2a4a")
                st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Best of Both Worlds (Hybrid)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown('<p class="sec-head">🎯 Best of Both Worlds</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Combine "movies like this one" with "movies you\'d personally enjoy" — tune the mix with a simple slider.</p>', unsafe_allow_html=True)

    if hybrid_rec is None:
        st.markdown("""
        <div class="taste-card">
          <h4>🔒 Needs the ratings data to be set up first</h4>
          <p>Complete the one-time setup in the <strong>✨ Recommended For You</strong> tab,
          then restart the app to unlock this feature.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("#### 🎬 Step 1 — Pick a movie you like")
        h_movie = st.selectbox("Movie:", titles, key="hyb_movie",
                               label_visibility="collapsed")

        st.markdown("#### 👤 Step 2 — Pick a viewer profile")
        profile_opts2 = {uid: f"Viewer #{uid}  ·  {user_profiles.get(uid,{}).get('movies_rated','?')} movies rated"
                         for uid in collab_rec.get_user_ids()}
        h_label = st.selectbox("Profile:", list(profile_opts2.values()), key="hyb_user",
                               label_visibility="collapsed")
        h_user = [k for k, v in profile_opts2.items() if v == h_label][0]

        st.markdown("#### ⚖️ Step 3 — Choose your mix")

        mix_choice = st.radio(
            "What matters most to you?",
            ["🎭 Very similar to the movie I picked",
             "🔀 A balanced mix of both",
             "🌟 Personalised to this viewer's taste",
             "🎛️ Let me set it manually"],
            index=1, horizontal=True, key="mix_radio",
        )
        weight_map = {
            "🎭 Very similar to the movie I picked": 0.85,
            "🔀 A balanced mix of both": 0.5,
            "🌟 Personalised to this viewer's taste": 0.15,
            "🎛️ Let me set it manually": None,
        }
        preset_weight = weight_map[mix_choice]

        if preset_weight is None:
            weight = st.slider(
                "Slide left = more personal taste  ·  Slide right = more similar movie",
                0.0, 1.0, 0.5, 0.05, key="hyb_weight",
            )
            wc, wl = st.columns(2)
            wc.metric("Similarity", f"{weight:.0%}")
            wl.metric("Personal taste", f"{1-weight:.0%}")
        else:
            weight = preset_weight

        h_n = st.number_input("How many recommendations?", 5, 20, 10, key="hyb_n")

        if st.button("🚀 Get My Personalised Mix", key="btn_hyb", type="primary"):
            with st.spinner("Blending both approaches for you…"):
                recs = hybrid_rec.recommend(h_movie, int(h_user), n=int(h_n),
                                            content_weight=weight)

            if recs.empty:
                st.warning("No results found. Try a different movie or profile.")
            else:
                p = user_profiles.get(h_user, {})
                st.success(
                    f"🎉 {len(recs)} picks combining **{h_movie}** "
                    f"with Viewer #{h_user}'s taste"
                )

                with st.expander("🤔 How did we pick these?"):
                    st.write(
                        f"We gave **{weight:.0%}** weight to movies thematically similar to "
                        f"*{h_movie}* and **{1-weight:.0%}** weight to what Viewer #{h_user} "
                        "would personally enjoy. Both scores were combined to find movies "
                        "that tick both boxes."
                    )

                # Movie cards
                card_cols = st.columns(5)
                for i, row in recs.iterrows():
                    with card_cols[i % 5]:
                        poster = safe_poster(row)
                        score = row["hybrid_score"]
                        st.markdown(f"""
                        <div class="movie-card">
                          <img src="{poster}" style="width:100%;border-radius:14px 14px 0 0;display:block;aspect-ratio:2/3;object-fit:cover;">
                          <div class="card-body">
                            <div class="card-title">{row['title']}</div>
                            <span class="badge badge-green">Score {score:.2f}</span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

                # Score breakdown
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="How similar to movie", x=recs["title"], y=recs["content_score"],
                    marker_color="#7f53ac",
                ))
                fig.add_trace(go.Bar(
                    name="How much this viewer would enjoy", x=recs["title"], y=recs["collab_score"],
                    marker_color="#f7971e",
                ))
                fig.update_layout(
                    barmode="group", title="Score Breakdown",
                    xaxis_tickangle=-30,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc", height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=0, r=0, t=60, b=80),
                )
                fig.update_xaxes(gridcolor="#2a2a4a")
                fig.update_yaxes(gridcolor="#2a2a4a")
                st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — How It Works
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown('<p class="sec-head">ℹ️ How CineMatch Works</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">No jargon — here\'s exactly what happens when you ask for a recommendation.</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="how-card">
          <div class="how-icon">🎥</div>
          <div class="how-title">Find Similar Movies</div>
          <div class="how-desc">
            We read the genres, cast, director, and story keywords of every movie.
            Then we measure how much overlap exists between your chosen movie and all others.
            The more overlap → the higher the match %.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="how-card">
          <div class="how-icon">✨</div>
          <div class="how-title">Recommended For You</div>
          <div class="how-desc">
            We analysed 100,000+ real movie ratings. When you pick a viewer profile,
            we find other viewers with similar tastes and look at what they loved
            that this viewer hasn't seen yet.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="how-card">
          <div class="how-icon">🎯</div>
          <div class="how-title">Best of Both Worlds</div>
          <div class="how-desc">
            Combines both approaches. The slider lets you decide:
            do you want movies thematically close to your pick,
            or movies that match a viewer's personal taste?
            Or a bit of each!
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 📊 What's under the hood")
    with st.spinner("Computing…"):
        avg_sim = content_rec.avg_similarity_score(n=10, sample=200)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Movies analysed", f"{len(titles):,}")
    col_b.metric("Avg similarity score", f"{avg_sim:.3f}")
    col_c.metric("Ratings used", f"{n_ratings:,}" if collab_ready else "Not loaded")

    if collab_ready:
        col_d, col_e = st.columns(2)
        col_d.metric("Prediction accuracy (RMSE)", f"±{collab_rec.rmse} stars",
                     help="On a 0.5–5 star scale, predictions are this many stars off on average")
        col_e.metric("Mean error (MAE)", f"±{collab_rec.mae} stars")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ❓ FAQ")

    with st.expander("Why doesn't my favourite movie show up in the search?"):
        st.write(
            "Our database comes from TMDB 5000 — the 5,000 most popular movies on "
            "The Movie Database. Very new or very obscure films may not be included. "
            "Try searching for well-known titles for best results."
        )
    with st.expander("What does the match % mean?"):
        st.write(
            "It's how much vocabulary (genres, cast, keywords) the two movies share. "
            "20–25% is actually a strong match — most movies share very little overlap. "
            "A 20% match for an action thriller means it's genuinely in the same world."
        )
    with st.expander("What are the viewer profiles in the 'For You' tab?"):
        st.write(
            "They're anonymised real people from the MovieLens research dataset — "
            "a project by the University of Minnesota where volunteers rated movies. "
            "Nobody's name or personal info is stored, just their star ratings."
        )
    with st.expander("Why do I need to download extra data for the 'For You' tab?"):
        st.write(
            "The MovieLens ratings file is ~1 MB but can't be bundled with the app "
            "for licensing reasons. The one-time download takes under 2 minutes "
            "and unlocks both the 'For You' and 'Best of Both Worlds' tabs permanently."
        )
