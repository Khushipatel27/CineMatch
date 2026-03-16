<div align="center">

# CineMatch

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.22-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

<br/>

[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)](.)
[![Models](https://img.shields.io/badge/Models-3%20AI%20Engines-blueviolet?style=flat-square)](.)
[![Movies](https://img.shields.io/badge/Dataset-4%2C806%20Movies-blue?style=flat-square)](.)
[![Ratings](https://img.shields.io/badge/Ratings-100k%2B-orange?style=flat-square)](.)

<br/>

> **Tell CineMatch one movie you love — it finds your next favourite using three AI engines.**

</div>

---

## 🌟 Three AI Engines, One App

|     | Engine                  | How it thinks                         | Best for                              |
| :-: | :---------------------- | :------------------------------------ | :------------------------------------ |
| 🎥  | **Find Similar Movies** | Genres, cast, director, keywords      | _"I loved Inception, what else?"_     |
| ✨  | **Recommended For You** | 100k real viewer rating patterns      | _"What would someone like me enjoy?"_ |
| 🎯  | **Best of Both Worlds** | Blends both with an adjustable slider | _Similar content + personal taste_    |

---

## 📊 Model Performance

| Model             | Algorithm                               |           Performance            |
| :---------------- | :-------------------------------------- | :------------------------------: |
| **Content-Based** | CountVectorizer + Cosine Similarity     |    Avg similarity **≈ 0.20**     |
| **Collaborative** | SVD Matrix Factorization (100k ratings) | RMSE **≈ 0.87** · MAE **≈ 0.67** |
| **Hybrid**        | Min-max normalised weighted blend       |        Slider-adjustable         |

> SVD RMSE < 0.90 matches published baselines for the MovieLens-100K benchmark.

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/Khushipatel27/Movie-Recommendation.git
cd Movie-Recommendation

# 2. Create environment (conda recommended on Windows)
conda create -n movie_rec python=3.11
conda activate movie_rec
conda install -c conda-forge numpy pandas scikit-learn scikit-surprise

# 3. Install remaining packages
pip install streamlit==1.35.0 plotly==5.22.0 nltk==3.8.1 requests==2.32.3 python-dotenv==1.0.1 statsmodels nbformat

# 4. (Optional) Add TMDB API key for movie posters
echo "TMDB_API_KEY=your_key_here" > .env

# 5. Run
streamlit run app.py
```

App opens at **`http://localhost:8501`**

---

## 📁 Project Structure

```
CineMatch/
├── src/
│   ├── content_based.py   ← CountVectorizer + cosine similarity
│   ├── collaborative.py   ← SVD · RMSE/MAE evaluation
│   ├── hybrid.py          ← Min-max normalisation + weighted blend
│   └── utils.py           ← TMDB API · data loaders
├── data/                  ← TMDB 5000 + MovieLens CSVs
├── notebooks/             ← EDA + model evaluation
└── app.py                 ← Streamlit app
```

---

## 📚 Datasets

- **TMDB 5000** — [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) · 4,806 films
- **MovieLens Small** — [GroupLens, UMN](https://grouplens.org/datasets/movielens/latest/) · 100,836 ratings

---

<div align="center">

Built with ❤️ by [Khushi Patel](https://github.com/Khushipatel27)

</div>
