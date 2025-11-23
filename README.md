#README.md

## Overview
A hybrid recommender combining Collaborative Filtering (SVD) and Content-Based (TF-IDF genres) approaches. Includes training pipeline, evaluation, and a Flask API for serving recommendations.

## Features
- SVD-based collaborative filtering (Surprise)
- TF-IDF content-based model on genres
- Weighted hybrid blending
- Flask API: `/recommend?user_id=1&k=10&alpha=0.8`

## Dataset
MovieLens 100k (put `u.data` and `u.item` in `data/`)

## Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
