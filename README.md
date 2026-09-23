# TMDB Movie Recommender System

A metadata-based movie recommender built with TF-IDF and cosine similarity. It provides a FastAPI API and a Gradio frontend. Streamlit, CI/CD, and GitHub Actions are not used.

## How it works

The model combines each movie's title, genre, language, and overview into one text field. TF-IDF converts those fields into vectors, and cosine similarity finds movies with similar metadata.

## Project structure

```text
src/
├── data/top10K-TMDB-movies.csv   # Input dataset
├── models/                       # Locally generated model artifacts
├── notebook/00_eda.ipynb         # Notebook training workflow
└── recommender/
    ├── api.py                    # FastAPI application
    ├── model.py                  # Model training and recommendations
    ├── service.py                # Shared cached model service
    ├── train.py                  # Training function
    └── ui.py                     # Gradio frontend

api.py                            # FastAPI launcher
gradio_app.py                     # Gradio launcher
train_model.py                    # Training launcher
tests/                            # Regression tests
```

## Setup

Python 3.12 or newer is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

Run this once after setup, or whenever the dataset changes:

```bash
python train_model.py
```

This creates the local files below:

```text
src/models/movies.pkl
src/models/similarity.pkl
```

These generated files are ignored by Git because `similarity.pkl` is larger than GitHub's file-size limit.

## Run the API

```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive API documentation at <http://127.0.0.1:8000/docs>.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Recommendation request:

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"movie_title":"The Shawshank Redemption","top_n":5}'
```

## Run the frontend

```bash
python gradio_app.py
```

Open <http://127.0.0.1:7860>. The frontend includes a searchable dropdown containing the movie titles in the dataset. Users can also type a title manually.

## Run tests

```bash
pytest -q
```
