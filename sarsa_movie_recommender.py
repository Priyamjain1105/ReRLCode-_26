from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import numpy as np
import pandas as pd


@dataclass
class UserProfile:
    preferred_genre: str
    exploration_tolerance: float


class MovieRecommendationEnv:
    """
    A simple episodic environment for movie recommendations.
    - State: (last_genre_match, genre_bucket_index)
    - Action: recommend one movie by index
    - Reward: based on genre match, movie quality, and novelty penalty
    """

    def __init__(self, movies_df: pd.DataFrame, user_profile: UserProfile):
        self.movies = movies_df.copy()
        self.user = user_profile
        self.movie_count = len(self.movies)
        self.genre_to_idx = {g: i for i, g in enumerate(sorted(self.movies["genre"].unique()))}
        self.n_genres = len(self.genre_to_idx)
        self.recent_actions: List[int] = []
        self.max_recent = 3
        self.current_state = (0, 0)

    def reset(self) -> Tuple[int, int]:
        self.recent_actions = []
        # Start from neutral signal.
        self.current_state = (0, random.randint(0, self.n_genres - 1))
        return self.current_state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        movie = self.movies.iloc[action]
        genre_match = int(movie["genre"] == self.user.preferred_genre)

        # Base preference reward.
        reward = 1.0 if genre_match else -0.2

        # Movie intrinsic quality.
        reward += float(movie["quality"]) * 0.8

        # Small novelty penalty if recommending repeated items.
        if action in self.recent_actions:
            reward -= 0.6

        # Exploration bonus for non-preferred genre if user tolerates novelty.
        if not genre_match:
            reward += 0.3 * self.user.exploration_tolerance

        # Next state updates:
        # - first bit tracks immediate genre match signal
        # - second bucket is the genre index of the current movie
        next_state = (genre_match, self.genre_to_idx[movie["genre"]])
        self.current_state = next_state

        self.recent_actions.append(action)
        if len(self.recent_actions) > self.max_recent:
            self.recent_actions.pop(0)

        # Fixed horizon episodic task (set by training loop), so env itself never ends.
        done = False
        return next_state, reward, done

    @property
    def state_size(self) -> Tuple[int, int]:
        # 2 possible match bits x number of genre buckets.
        return 2, self.n_genres


class SARSAMovieRecommender:
    def __init__(
        self,
        n_match_bits: int,
        n_genres: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table dimensions: [match_bit][genre_bucket][movie_action]
        self.q_table = np.zeros((n_match_bits, n_genres, n_actions), dtype=np.float64)

    def choose_action(self, state: Tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.q_table.shape[2])
        return int(np.argmax(self.q_table[state[0], state[1], :]))

    def learn(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        next_action: int,
    ) -> None:
        current_q = self.q_table[state[0], state[1], action]
        next_q = self.q_table[next_state[0], next_state[1], next_action]
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        self.q_table[state[0], state[1], action] += self.alpha * td_error

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_movie_indices_for_state(self, state: Tuple[int, int], top_k: int = 3) -> List[int]:
        q_vals = self.q_table[state[0], state[1], :]
        ranked = np.argsort(q_vals)[::-1]
        return ranked[:top_k].tolist()


def train_sarsa_recommender(
    env: MovieRecommendationEnv,
    agent: SARSAMovieRecommender,
    episodes: int = 1000,
    steps_per_episode: int = 10,
) -> List[float]:
    reward_history: List[float] = []

    for _ in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0.0

        for _ in range(steps_per_episode):
            next_state, reward, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)

            state, action = next_state, next_action
            total_reward += reward

        agent.decay_exploration()
        reward_history.append(total_reward)

    return reward_history


def recommend_movies(
    env: MovieRecommendationEnv,
    agent: SARSAMovieRecommender,
    context_state: Tuple[int, int] = (1, 0),
    top_k: int = 3,
) -> pd.DataFrame:
    indices = agent.best_movie_indices_for_state(context_state, top_k=top_k)
    recs = env.movies.iloc[indices][["title", "genre", "quality"]].copy()
    recs["score"] = [agent.q_table[context_state[0], context_state[1], idx] for idx in indices]
    return recs.sort_values("score", ascending=False)


def run_demo() -> None:
    movies = pd.read_csv("movie_data.csv")

    # Example target user profile.
    user = UserProfile(preferred_genre="Sci-Fi", exploration_tolerance=0.35)
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

    rewards = train_sarsa_recommender(env, agent, episodes=1200, steps_per_episode=8)
    moving_avg = float(np.mean(rewards[-100:]))

    print("Training complete.")
    print(f"Average reward (last 100 episodes): {moving_avg:.3f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print()

    # Recommend in two contexts:
    # (1, genre_bucket_of_sci_fi) means last recommendation matched preference.
    sci_fi_bucket = env.genre_to_idx.get("Sci-Fi", 0)
    preferred_context = (1, sci_fi_bucket)
    recommendations = recommend_movies(env, agent, context_state=preferred_context, top_k=5)
    print("Top movie recommendations:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    run_demo()
