import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from analysis import (
    METHODS, fit, genres_summary, metrics, period_stability, predict, rankings,
    select_alpha, smooth, temporal_split,
)
from generate import generate, make_ratings
from research import paired_intervals, error_examples, write_html
from reporting import charts, csv, write_report


ROOT = Path(__file__).resolve().parent


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, description, passed):
        if not passed:
            raise ValueError(f"Проверка не прошла: {description}")
        self.rows.append({"check": description, "status": "ok"})


def check_data(checks, catalog, ratings, profile, rules, parts):
    movies, users = catalog.movies, catalog.users
    checks.add("Размеры трёх таблиц совпадают с конфигом",
               (len(movies), len(users), len(ratings)) ==
               (profile["movies"], profile["users"], profile["ratings"]))
    checks.add("Идентификаторы фильмов, пользователей и оценок уникальны",
               movies["movie_id"].is_unique and users["user_id"].is_unique
               and ratings["rating_id"].is_unique)
    checks.add("Пять жанров и уникальные названия фильмов",
               movies["genre"].nunique() == 5 and movies["title"].is_unique)
    checks.add("Пары пользователь–фильм уникальны",
               not ratings.duplicated(["user_id", "movie_id"]).any())
    checks.add("У каждой оценки есть фильм и пользователь",
               ratings["movie_id"].isin(movies["movie_id"]).all()
               and ratings["user_id"].isin(users["user_id"]).all())
    checks.add("Объединение с фильмами сохраняет число строк",
               len(ratings.merge(movies, on="movie_id", validate="many_to_one")) == len(ratings))
    checks.add("Оценки лежат в диапазоне 1–5 с шагом 0,5",
               ratings["rating"].between(1, 5).all() and ratings["rating"].mul(2).mod(1).eq(0).all())
    checks.add("Оценки находятся в заданном интервале дат",
               ratings["timestamp"].between(pd.Timestamp("2025-01-01"),
                       pd.Timestamp("2025-01-01") + pd.Timedelta(days=profile["days"] - 1)).all())
    train, validation, test = parts
    counts = train.groupby("movie_id").size()
    checks.add("В обучении есть фильмы с 1–9 и с 20+ оценками",
               counts.between(1, 9).any() and counts.ge(20).any())
    checks.add("Части покрывают все оценки и строго следуют по времени",
               sum(map(len, parts)) == len(ratings)
               and train["timestamp"].max() < validation["timestamp"].min()
               and validation["timestamp"].max() < test["timestamp"].min())
    saved_catalog, same_ratings = generate(profile, rules)
    checks.add("Одинаковый seed воспроизводит открытые и скрытые таблицы",
               all(table.equals(getattr(saved_catalog, name)) for name, table in vars(catalog).items())
               and ratings.equals(same_ratings))


def check_forecast(checks):
    fixture = pd.DataFrame({
        "rating_id": range(1, 11),
        "timestamp": pd.to_datetime(["2025-01-01"] * 7 + ["2025-01-02"] * 2 + ["2025-01-03"]),
        "movie_id": [1, 1, 2, 2, 1, 2, 1, 99, 1, 99],
        "rating": [4, 4, 2, 2, 4, 2, 3, 5, 5, 5],
    })
    past, validation, _ = temporal_split(fixture)
    checks.add("При совпадении меток пограничные события уходят в раннюю часть",
               (len(past), len(validation)) == (7, 2))
    history = fit(past)
    cold = predict(validation, history, 5).query("movie_id == 99")
    checks.add("Фильм без истории получает общее среднее во всех трёх методах",
               cold[list(METHODS)].eq(history.global_mean).all().all())
    checks.add("Ручной пример сглаживания даёт 3,5", smooth(8, 2, 3, 2) == 3.5)
    try:
        predict(past, history, 5)
    except ValueError:
        rejects_overlap = True
    else:
        rejects_overlap = False
    checks.add("Прогноз проверяет, что события строго следуют за историей", rejects_overlap)
    chosen, _ = select_alpha(validation.query("movie_id == 99"), history, [5, 20, 50], 0.001)
    checks.add("При равных RMSE выбирается больший α", chosen == 50)


def top_overlap(catalog, profile, rules, alpha, seeds):
    tops_by_seed = {}
    for seed in seeds:
        ratings = make_ratings(catalog, profile, rules, np.random.default_rng(seed))
        train, _, _ = temporal_split(ratings)
        tops = rankings(catalog.movies, fit(train), alpha)
        tops_by_seed[seed] = {
            method: set(tops[method].head(10)["movie_id"])
            for method in ("raw_mean", "smoothed")
        }
    rows = []
    for first, second in zip(seeds[::2], seeds[1::2]):
        base, other = tops_by_seed[first], tops_by_seed[second]
        rows.append({
            "seed_a": first, "seed_b": second, "alpha": alpha,
            "raw_overlap": len(base["raw_mean"] & other["raw_mean"]),
            "smoothed_overlap": len(base["smoothed"] & other["smoothed"]),
        })
    return pd.DataFrame(rows)


def run(config, profile_name, output):
    start = perf_counter()
    profile = config["profiles"][profile_name]
    rules = config["generator"]
    settings = config["analysis"]
    catalog, ratings = generate(profile, rules)
    parts = temporal_split(ratings)
    train, validation, test = parts
    checks = Checks()
    check_data(checks, catalog, ratings, profile, rules, parts)
    check_forecast(checks)

    train_history = fit(train)
    alpha, validation_scores = select_alpha(
        validation, train_history, settings["alphas"], settings["alpha_tolerance"],
    )
    tops = rankings(catalog.movies, train_history, alpha)
    genre_stats = genres_summary(catalog.movies, train)
    pretest = pd.concat([train, validation], ignore_index=True)
    period_scores = period_stability(pretest, settings["alphas"], settings["alpha_tolerance"])
    checks.add("В каждом окне коэффициент выбран до первой проверяемой оценки",
               (period_scores["tuning_end"] < period_scores["start"]).all())
    overlaps = pd.DataFrame()
    if profile_name == "main":
        overlaps = top_overlap(catalog, profile, rules, alpha, settings["stability_seeds"])

    # На тесте используем общую историю и уже выбранный коэффициент.
    test_predictions = predict(test, fit(pretest), alpha)
    checks.add("Все три метода получили одни и те же тестовые события",
               test_predictions["rating_id"].equals(test["rating_id"])
               and test_predictions[list(METHODS)].notna().all().all())
    scores = metrics(test_predictions, groups=True)
    checks.add("Тестовые группы покрывают все события без пересечений",
               scores.loc[scores["group"].ne("Все")].groupby("method")["n_events"].sum().eq(len(test)).all())

    csv(catalog.movies, output / "data/movies.csv")
    csv(catalog.users, output / "data/users.csv")
    csv(ratings, output / "data/ratings.csv")
    csv(catalog.movie_parameters, output / "data/private/movie_parameters.csv")
    csv(catalog.user_parameters, output / "data/private/user_parameters.csv")
    for name, table in tops.items():
        csv(table, output / f"results/rankings/{name}.csv")
    split_rows = [
        {"part": name, "n_events": len(part), "start": part["timestamp"].min(),
         "end": part["timestamp"].max()}
        for name, part in zip(("train", "validation", "test"), parts)
    ]
    intervals = paired_intervals(test_predictions, settings["bootstrap_seed"], settings["bootstrap_repeats"])
    examples = error_examples(test_predictions, catalog.movies)
    errors = scores.pivot(index="group", columns="method", values="rmse")
    deltas = errors["smoothed_mean"] - errors["movie_mean"]
    measured = intervals.set_index("group")["delta"].reindex(deltas.index)
    checks.add("Разница RMSE по фильмам совпадает с разницей итоговых метрик",
               np.allclose(measured, deltas, equal_nan=True))
    result_tables = {
        "paired_intervals": intervals,
        "error_examples": examples,
        "checks": pd.DataFrame(checks.rows),
        "split": pd.DataFrame(split_rows),
        "genres": genre_stats,
        "validation_metrics": validation_scores,
        "test_predictions": test_predictions,
        "test_metrics": scores,
        "period_stability": period_scores,
    }
    if not overlaps.empty:
        result_tables["top_overlap"] = overlaps
    for name, table in result_tables.items():
        csv(table, output / f"results/{name}.csv")
    charts(output / "results/figures", catalog.movies, train, tops, scores)
    write_report(output / "report.md", profile_name, profile, parts, alpha, validation_scores,
                 scores, tops, genre_stats, period_scores, overlaps, checks.rows, intervals, test_predictions)
    write_html(output, profile, alpha, train_history, tops, scores, intervals, examples,
               settings["bootstrap_repeats"], validation_scores, split_rows)
    print(f"{profile_name}: alpha={alpha}, train={len(train)}, validation={len(validation)}, "
          f"test={len(test)}, checks={len(checks.rows)}, seconds={perf_counter() - start:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run the FilmRating project")
    parser.add_argument("--profile", choices=("quick", "main", "fallback"), default="main")
    parser.add_argument("--output", type=Path, help="Output directory (default: runs/<profile>)")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    output = args.output if args.output is not None else ROOT / "runs" / args.profile
    run(config, args.profile, output)


if __name__ == "__main__":
    main()
