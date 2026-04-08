# SARSA Movie Recommendation System (Python)

This project implements a simple **movie recommendation system** using the **SARSA reinforcement learning algorithm**.

## What it does

- Treats recommending a movie as an action
- Learns action values with SARSA (`Q[s, a]` updates from current and next action)
- Simulates user feedback as reward:
  - higher reward for preferred genre
  - reward boost from movie quality
  - novelty penalty for repetitive recommendations

## Project files

- `sarsa_movie_recommender.py` - main training and recommendation script
- `movie_data.csv` - small sample movie dataset
- `requirements.txt` - Python dependencies

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python sarsa_movie_recommender.py
```

## Customize

In `run_demo()` inside `sarsa_movie_recommender.py`, you can change:

- `preferred_genre`
- `exploration_tolerance`
- training parameters (`episodes`, `alpha`, `gamma`, `epsilon_decay`, etc.)

For better real-world performance, replace `movie_data.csv` with larger data and connect rewards to real user interactions (clicks, watch-time, ratings).
