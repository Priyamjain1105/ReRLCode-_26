from __future__ import annotations

from typing import Dict, List
from uuid import uuid4

import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for

from sarsa_movie_recommender import (
    MovieRecommendationEnv,
    SARSAMovieRecommender,
    UserProfile,
    recommend_movies,
    train_sarsa_recommender,
)


app = Flask(__name__)
app.secret_key = "sarsa-movie-demo-secret"

# In-memory store for live sessions. Good for local demo usage.
LIVE_SESSIONS: Dict[str, Dict] = {}


def load_movies() -> pd.DataFrame:
    return pd.read_csv("movie_data.csv")


def train_agent_for_user(preferred_genre: str, exploration_tolerance: float):
    movies = load_movies()
    user = UserProfile(
        preferred_genre=preferred_genre,
        exploration_tolerance=exploration_tolerance,
    )
    env = MovieRecommendationEnv(movies, user)
    n_match_bits, n_genres = env.state_size
    agent = SARSAMovieRecommender(
        n_match_bits=n_match_bits,
        n_genres=n_genres,
        n_actions=len(movies),
        alpha=0.15,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )
    rewards = train_sarsa_recommender(env, agent, episodes=1000, steps_per_episode=8)
    return env, agent, rewards


def get_genres(movies: pd.DataFrame) -> List[str]:
    return sorted(movies["genre"].unique().tolist())


def ensure_live_session_id() -> str:
    sid = session.get("live_session_id")
    if not sid:
        sid = str(uuid4())
        session["live_session_id"] = sid
    return sid


def start_live_session(preferred_genre: str, exploration_tolerance: float) -> None:
    sid = ensure_live_session_id()
    movies = load_movies()
    user = UserProfile(
        preferred_genre=preferred_genre,
        exploration_tolerance=exploration_tolerance,
    )
    env = MovieRecommendationEnv(movies, user)
    n_match_bits, n_genres = env.state_size
    agent = SARSAMovieRecommender(
        n_match_bits=n_match_bits,
        n_genres=n_genres,
        n_actions=len(movies),
        alpha=0.2,
        gamma=0.9,
        epsilon=0.35,
        epsilon_min=0.05,
        epsilon_decay=0.999,
    )

    state = env.reset()
    action = agent.choose_action(state)
    movie = env.movies.iloc[action]
    LIVE_SESSIONS[sid] = {
        "env": env,
        "agent": agent,
        "state": state,
        "action": action,
        "current_movie": movie,
        "steps": 0,
        "mistakes": 0,
        "reward_sum": 0.0,
        "history": [],
    }


def get_live_view_model():
    sid = ensure_live_session_id()
    data = LIVE_SESSIONS.get(sid)
    if not data:
        return None

    env = data["env"]
    movie = data["current_movie"]
    is_mismatch = bool(movie["genre"] != env.user.preferred_genre)
    avg_reward = data["reward_sum"] / data["steps"] if data["steps"] else 0.0
    return {
        "active": True,
        "preferred_genre": env.user.preferred_genre,
        "exploration_tolerance": env.user.exploration_tolerance,
        "current_title": movie["title"],
        "current_genre": movie["genre"],
        "current_quality": float(movie["quality"]),
        "is_mismatch": is_mismatch,
        "steps": data["steps"],
        "mistakes": data["mistakes"],
        "avg_reward": round(avg_reward, 3),
        "history": data["history"][-8:][::-1],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    movies = load_movies()
    genres = get_genres(movies)
    selected_genre = "Sci-Fi"
    exploration_tolerance = 0.35
    results = None
    avg_reward = None

    if request.method == "POST":
        selected_genre = request.form.get("preferred_genre", "Sci-Fi")
        try:
            exploration_tolerance = float(request.form.get("exploration_tolerance", 0.35))
        except ValueError:
            exploration_tolerance = 0.35

        exploration_tolerance = min(max(exploration_tolerance, 0.0), 1.0)
        env, agent, rewards = train_agent_for_user(selected_genre, exploration_tolerance)
        avg_reward = round(sum(rewards[-100:]) / min(100, len(rewards)), 3)

        genre_bucket = env.genre_to_idx.get(selected_genre, 0)
        context_state = (1, genre_bucket)
        recs_df = recommend_movies(env, agent, context_state=context_state, top_k=5)
        results = recs_df.to_dict(orient="records")

    return render_template(
        "index.html",
        genres=genres,
        selected_genre=selected_genre,
        exploration_tolerance=exploration_tolerance,
        results=results,
        avg_reward=avg_reward,
        live=get_live_view_model(),
    )


@app.route("/live/start", methods=["POST"])
def live_start():
    selected_genre = request.form.get("preferred_genre", "Sci-Fi")
    try:
        exploration_tolerance = float(request.form.get("exploration_tolerance", 0.35))
    except ValueError:
        exploration_tolerance = 0.35

    exploration_tolerance = min(max(exploration_tolerance, 0.0), 1.0)
    start_live_session(selected_genre, exploration_tolerance)
    return redirect(url_for("index"))


@app.route("/live/feedback", methods=["POST"])
def live_feedback():
    sid = ensure_live_session_id()
    data = LIVE_SESSIONS.get(sid)
    if not data:
        return redirect(url_for("index"))

    env = data["env"]
    agent = data["agent"]
    state = data["state"]
    action = data["action"]
    movie = data["current_movie"]
    feedback = request.form.get("feedback", "like")

    # User feedback nudges reward so you can watch live correction.
    next_state, env_reward, _ = env.step(action)
    user_bonus = 1.2 if feedback == "like" else -1.2
    reward = env_reward + user_bonus
    next_action = agent.choose_action(next_state)
    agent.learn(state, action, reward, next_state, next_action)
    agent.decay_exploration()

    data["steps"] += 1
    data["reward_sum"] += reward
    was_mismatch = bool(movie["genre"] != env.user.preferred_genre)
    if was_mismatch and feedback == "dislike":
        data["mistakes"] += 1

    data["history"].append(
        {
            "title": movie["title"],
            "genre": movie["genre"],
            "feedback": feedback,
            "reward": round(reward, 3),
        }
    )

    next_movie = env.movies.iloc[next_action]
    data["state"] = next_state
    data["action"] = next_action
    data["current_movie"] = next_movie

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
