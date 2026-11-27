"""Проверка сохранённых результатов средствами стандартной библиотеки Python."""
import argparse
import csv
import json
from collections import defaultdict
from math import fsum, isclose, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ("global_mean", "movie_mean", "smoothed_mean")
GROUPS = ("Все", "0", "1–9", "10+")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(ok, message):
    if not ok:
        raise ValueError(message)


def compare(actual, expected, label):
    # При сохранении CSV числа округляются до десяти значащих цифр.
    ok = actual == "" if expected is None else isclose(float(actual), expected, abs_tol=5e-9, rel_tol=0)
    check(ok, f"{label}: в CSV {actual!r}, ожидалось {expected!r}")


def split_data(events, saved):
    ordered = sorted(events, key=lambda r: (r["timestamp"], int(r["rating_id"])))
    day60, day80 = (ordered[len(ordered) * pct // 100 - 1]["timestamp"] for pct in (60, 80))
    parts = {"train": [], "validation": [], "test": []}
    for r in ordered:
        day = r["timestamp"]
        part = "train" if day <= day60 else "validation" if day <= day80 else "test"
        parts[part].append(r)
    check(len(saved) == 3 and {r["part"] for r in saved} == set(parts), "Состав частей в split.csv")
    for row in saved:
        records = parts[row["part"]]
        check(len(records) == int(row["n_events"]), "Размер части " + row["part"])
        check(records and (row["start"], row["end"]) == (records[0]["timestamp"], records[-1]["timestamp"]),
              "Временные границы части " + row["part"])
    return parts["train"], parts["validation"], parts["test"]


def collect(history):
    votes = defaultdict(list)
    for row in history:
        votes[row["movie_id"]].append(float(row["rating"]))
    totals = {key: (len(points), fsum(points)) for key, points in votes.items()}
    average = fsum(total for _, total in totals.values()) / len(history)
    return totals, average


def forecast(totals, average, movie, alpha):
    n, total = totals.get(movie, (0, 0.0))
    if not n:
        return 0, (average, average, average)
    return n, (average, total / n, (total + alpha * average) / (n + alpha))


def errors_for(events, history, alpha, saved=None):
    totals, average = history
    errors = defaultdict(list)
    if saved is not None:
        check(len(events) == len(saved), "Размер таблицы тестовых прогнозов")
    for i, row in enumerate(events):
        n, values = forecast(totals, average, row["movie_id"], alpha)
        group = "0" if n == 0 else "1–9" if n < 10 else "10+"
        if saved is not None:
            actual = saved[i]
            check(all(actual[k] == row[k] for k in ("rating_id", "user_id", "movie_id", "timestamp", "rating")),
                  "Перепутано тестовое событие " + row["rating_id"])
            check(int(actual["n_ratings"]) == n and actual["group"] == group, "Группа события " + row["rating_id"])
        for method, value in zip(METHODS, values):
            if saved is not None:
                compare(actual[method], value, f"Оценка {row['rating_id']} / {method}")
            error = value - float(row["rating"])
            errors["Все", method].append(error)
            errors[group, method].append(error)
    return errors


def rmse(errors):
    return sqrt(fsum(e * e for e in errors) / len(errors))


def check_metrics(rows, errors, groups):
    keys = [(r["group"], r["method"]) for r in rows]
    check(sorted(keys) == sorted((g, m) for g in groups for m in METHODS), "Состав строк метрик")
    for row, key in zip(rows, keys):
        values = errors[key]
        check(len(values) == int(row["n_events"]), f"Численность группы {key}")
        check(row["status"] == ("ok" if values else "нет наблюдений"), f"Статус группы {key}")
        compare(row["rmse"], rmse(values) if values else None, f"{key} RMSE")
        compare(row["mae"], fsum(map(abs, values)) / len(values) if values else None, f"{key} MAE")


def choose_alpha(events, history, settings, rows):
    losses = {}
    check({int(r["alpha"]) for r in rows} == set(settings["alphas"]), "Список коэффициентов")
    for a in settings["alphas"]:
        errors = errors_for(events, history, a)
        check_metrics([r for r in rows if int(r["alpha"]) == a], errors, ("Все",))
        losses[a] = rmse(errors["Все", "smoothed_mean"])
    limit = min(losses.values()) + settings["alpha_tolerance"]
    best = max(a for a, value in losses.items() if value <= limit)
    check(all(r["selected"] == str(int(r["alpha"]) == best) for r in rows), "Выбор α на валидации")
    return best


def check_rankings(folder, movies, history, alpha):
    by_id = {r["movie_id"]: r for r in movies}
    values = {mid: forecast(*history, mid, alpha) for mid in by_id}
    specs = (("popularity", 0, "count"), ("raw_mean", 1, 1), ("mean_min_20", 20, 1), ("smoothed", 0, 2))
    for name, minimum, column in specs:
        pool = [mid for mid, (n, _) in values.items() if n >= minimum]
        # При одинаковом балле или числе голосов первым идёт меньший числовой ID.
        def order(mid):
            count, predictions = values[mid]
            score = count if column == "count" else predictions[column]
            return -score, int(mid)
        ranked = sorted(pool, key=order)
        rows = read_csv(folder / f"{name}.csv")
        check(len(rows) == len(ranked), f"Размер рейтинга {name}")
        for rank, (mid, row) in enumerate(zip(ranked, rows), 1):
            count, (_, raw, smooth) = values[mid]
            label = f"{name}, место {rank}"
            check(int(row["rank"]) == rank and row["movie_id"] == mid, label)
            check(all(row[k] == by_id[mid][k] for k in ("title", "genre")), label + ", подпись")
            check(int(row["n_ratings"]) == count, label + ", число голосов")
            compare(row["raw_mean"], raw if count else None, label + ", средний балл")
            compare(row["smoothed_mean"], smooth, label + ", сглаженный балл")


def verify(profile):
    folder = ROOT / "runs" / profile
    results = folder / "results"
    settings = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["analysis"]
    train, validation, test = split_data(read_csv(folder / "data/ratings.csv"), read_csv(results / "split.csv"))
    history = collect(train)
    alpha = choose_alpha(validation, history, settings, read_csv(results / "validation_metrics.csv"))
    check_rankings(results / "rankings", read_csv(folder / "data/movies.csv"), history, alpha)
    errors = errors_for(test, collect(train + validation), alpha, read_csv(results / "test_predictions.csv"))
    check_metrics(read_csv(results / "test_metrics.csv"), errors, GROUPS)
    print(f"{profile}: совпали границы периодов, выбор α = {alpha}, четыре рейтинга, "
          f"прогнозы для {len(test)} тестовых событий и все значения RMSE и MAE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("main", "fallback", "quick"), default="main")
    verify(parser.parse_args().profile)
