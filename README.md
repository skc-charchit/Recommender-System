# TMDB Movie Recommender System

A production-ready movie recommendation service built around a TF-IDF + cosine-similarity pipeline for TMDB movie metadata. The project exposes both a FastAPI service and a Gradio UI, without using Streamlit.

## Overview

This recommender uses movie metadata such as title, genre, original language, and overview to identify similar films. The pipeline builds a vector representation for each title and computes cosine similarity across the catalog.

## Project structure

- `src/recommender/model.py` – core recommender and model-building logic
- `src/recommender/service.py` – shared service layer for API/UI integration
- `src/recommender/api.py` – FastAPI application
- `src/recommender/ui.py` – Gradio application
- `src/recommender/train.py` – model training function
- `api.py`, `gradio_app.py`, `train_model.py` – small compatibility launchers
- `src/data/top10K-TMDB-movies.csv` – TMDB source dataset
- `src/models/` – serialized movie and similarity artifacts

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

The script creates or refreshes:

- `src/models/movies.pkl`
- `src/models/similarity.pkl`

## Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Test endpoint:

```bash
curl http://localhost:8000/health
```

Example recommendation call:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"movie_title":"The Shawshank Redemption","top_n":5}'
```

## Run the Gradio UI

```bash
python gradio_app.py
```

Then open the local Gradio URL shown in the terminal.

No CI/CD or GitHub Actions are included.
