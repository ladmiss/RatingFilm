from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GENRES = ["Драма", "Комедия", "Фантастика", "Детектив", "Приключения"]


@dataclass
class Catalog:
    movies: pd.DataFrame
    users: pd.DataFrame
    movie_parameters: pd.DataFrame
    user_parameters: pd.DataFrame


def make_catalog(profile, rules, rng):
    n_movies, n_users = profile["movies"], profile["users"]
    catalog_path = Path(__file__).with_name("movie_titles.tsv")
    titles = pd.read_csv(catalog_path, sep="\t", encoding="utf-8")
    if not 1 <= n_movies <= len(titles):
        raise ValueError(f"Число фильмов должно быть от 1 до {len(titles)}")
    movie_ids = np.arange(1, n_movies + 1)
    user_ids = np.arange(1, n_users + 1)

    # Перемешиваем названия и распределяем жанры двумя независимыми перестановками.
    shuffled = titles.iloc[rng.permutation(len(titles))]
    genres = rng.permutation(np.resize(GENRES, n_movies))
    choices = {genre: iter(rows["title"]) for genre, rows in shuffled.groupby("genre")}
    movies = pd.DataFrame({
        "movie_id": movie_ids,
        "title": [next(choices[genre]) for genre in genres],
        "genre": genres,
    })
    users = pd.DataFrame({"user_id": user_ids})
    movie_parameters = pd.DataFrame({
        "movie_id": movie_ids,
        "base_rating": rng.normal(rules["quality_mean"], rules["quality_sd"], n_movies),
        "popularity_weight": rng.lognormal(0, rules["popularity_log_sd"], n_movies),
    })
    user_parameters = pd.DataFrame({
        "user_id": user_ids,
        "rating_bias": rng.normal(0, rules["user_bias_sd"], n_users),
    })
    return Catalog(movies, users, movie_parameters, user_parameters)


def make_ratings(catalog, profile, rules, rng):
    n_movies, n_users = len(catalog.movies), len(catalog.users)
    n_ratings = profile["ratings"]
    weights = catalog.movie_parameters["popularity_weight"].to_numpy()
    probabilities = np.tile(weights, n_users)
    probabilities /= probabilities.sum()
    # Один номер соответствует одной паре зритель–фильм. Повторные пары исключены.
    pairs = rng.choice(n_users * n_movies, n_ratings, replace=False, p=probabilities)
    user_positions, movie_positions = np.divmod(pairs, n_movies)
    scores = (
        catalog.movie_parameters["base_rating"].to_numpy()[movie_positions]
        + catalog.user_parameters["rating_bias"].to_numpy()[user_positions]
        + rng.normal(0, rules["noise_sd"], n_ratings)
    )
    days = rng.integers(0, profile["days"], size=n_ratings)
    return pd.DataFrame({
        "rating_id": np.arange(1, n_ratings + 1),
        "user_id": catalog.users["user_id"].to_numpy()[user_positions],
        "movie_id": catalog.movies["movie_id"].to_numpy()[movie_positions],
        "timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(days, unit="D"),
        "rating": np.clip(np.rint(scores * 2) / 2, 1, 5),
    }).sort_values(["timestamp", "rating_id"], ignore_index=True)


def generate(profile, rules):
    rng = np.random.default_rng(profile["seed"])
    catalog = make_catalog(profile, rules, rng)
    return catalog, make_ratings(catalog, profile, rules, rng)
