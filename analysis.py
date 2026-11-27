from dataclasses import dataclass

import numpy as np
import pandas as pd


METHODS = {
    "global_mean": "Общее среднее",
    "movie_mean": "Среднее фильма",
    "smoothed_mean": "Сглаженное среднее",
}
GROUPS = ["0", "1–9", "10+"]
TOPS = {
    "popularity": "По числу оценок",
    "raw_mean": "По среднему баллу",
    "mean_min_20": "По среднему баллу при 20+ оценках",
    "smoothed": "По сглаженному баллу",
}


def temporal_split(ratings):
    ordered = ratings.sort_values(["timestamp", "rating_id"], ignore_index=True)
    dates = ordered["timestamp"]
    last_train_date = dates.iloc[len(ordered) * 60 // 100 - 1]
    last_validation_date = dates.iloc[len(ordered) * 80 // 100 - 1]
    # Включаем в раннюю часть все оценки, поставленные в пограничный день.
    train_end = dates.searchsorted(last_train_date, side="right")
    known_end = dates.searchsorted(last_validation_date, side="right")
    parts = ordered.iloc[:train_end], ordered.iloc[train_end:known_end], ordered.iloc[known_end:]
    if any(part.empty for part in parts):
        raise ValueError("Временные метки должны позволять получить три непустые части")
    return parts


@dataclass
class History:
    global_mean: float
    by_movie: pd.DataFrame
    end: pd.Timestamp


def fit(ratings):
    return History(
        float(ratings["rating"].mean()),
        ratings.groupby("movie_id")["rating"].agg(n_ratings="size", rating_sum="sum"),
        ratings["timestamp"].max(),
    )


def add_history(movies_or_events, history):
    known = movies_or_events.join(history.by_movie, on="movie_id", validate="many_to_one")
    known["n_ratings"] = known["n_ratings"].fillna(0).astype(int)
    known["rating_sum"] = known["rating_sum"].fillna(0.0)
    known["raw_mean"] = known["rating_sum"] / known["n_ratings"].replace(0, np.nan)
    return known


def smooth(rating_sum, count, global_mean, alpha):
    score = (rating_sum + alpha * global_mean) / (count + alpha)
    if isinstance(count, pd.Series):
        return score.where(count.gt(0), global_mean)
    return global_mean if count == 0 else score


def predict(events, history, alpha):
    if not events["timestamp"].gt(history.end).all():
        raise ValueError("Прогнозируемые оценки должны строго следовать за историей")
    result = add_history(events, history)
    result["global_mean"] = history.global_mean
    result["movie_mean"] = result["raw_mean"].fillna(history.global_mean)
    result["smoothed_mean"] = smooth(
        result["rating_sum"], result["n_ratings"], history.global_mean, alpha,
    )
    result["group"] = np.select(
        [result["n_ratings"].eq(0), result["n_ratings"].lt(10)],
        GROUPS[:2],
        default=GROUPS[2],
    )
    return result.drop(columns=["rating_sum", "raw_mean"])


def metrics(predictions, groups=False):
    subsets = {"Все": predictions}
    if groups:
        subsets.update({group: predictions.loc[predictions["group"].eq(group)] for group in GROUPS})
    rows = []
    for group, data in subsets.items():
        for method in METHODS:
            errors = data[method] - data["rating"]
            rows.append({
                "group": group,
                "method": method,
                "n_events": len(data),
                "rmse": float(np.sqrt(errors.pow(2).mean())),
                "mae": float(errors.abs().mean()),
                "status": "нет наблюдений" if data.empty else "ok",
            })
    return pd.DataFrame(rows)


def select_alpha(validation, history, alphas, tolerance):
    """Выбрать α по RMSE на валидации.

    Берём наибольший α среди вариантов, у которых ошибка отличается
    от минимальной не больше чем на заданный допуск.
    """
    candidates = []
    for alpha in alphas:
        scores = metrics(predict(validation, history, alpha))
        candidates.append(scores.assign(alpha=alpha))
    scores = pd.concat(candidates, ignore_index=True)
    smooth_scores = scores.loc[scores["method"].eq("smoothed_mean")]
    limit = smooth_scores["rmse"].min() + tolerance
    eligible = smooth_scores.loc[smooth_scores["rmse"] <= limit, "alpha"]
    alpha = int(eligible.max())
    scores["selected"] = scores["alpha"].eq(alpha)
    return alpha, scores


def rankings(movies, history, alpha):
    scores = add_history(movies, history)
    scores["smoothed_mean"] = smooth(
        scores["rating_sum"], scores["n_ratings"], history.global_mean, alpha,
    )
    rated = scores.loc[scores["n_ratings"].gt(0)]
    pools = {
        "popularity": (scores, "n_ratings"),
        "raw_mean": (rated, "raw_mean"),
        "mean_min_20": (rated.loc[rated["n_ratings"].ge(20)], "raw_mean"),
        "smoothed": (scores, "smoothed_mean"),
    }
    tops = {}
    for name, (pool, sort_column) in pools.items():
        top = pool.sort_values([sort_column, "movie_id"], ascending=[False, True]).copy()
        top.insert(0, "rank", np.arange(1, len(top) + 1))
        tops[name] = top.drop(columns="rating_sum")
    return tops


def genres_summary(movies, train):
    count = movies.groupby("genre")["movie_id"].size().rename("n_movies")
    joined = train.merge(movies[["movie_id", "genre"]], on="movie_id", validate="many_to_one")
    scores = joined.groupby("genre")["rating"].agg(n_events="size", mean_rating="mean")
    result = count.to_frame().join(scores)
    result["n_events"] = result["n_events"].fillna(0).astype(int)
    return result.reset_index()


def period_stability(before_test, alphas, tolerance):
    days = before_test["timestamp"].drop_duplicates().sort_values().to_numpy()
    # Три последовательных окна в конце периода до теста.
    windows = np.array_split(days[len(days) * 5 // 8:], 3)
    comparisons = []
    for days_in_window in windows:
        past = before_test.loc[before_test["timestamp"].lt(days_in_window[0])]
        window = before_test.loc[before_test["timestamp"].isin(days_in_window)]
        # Последнюю пятую часть прошлых дат оставляем для выбора коэффициента.
        past_days = past["timestamp"].drop_duplicates().sort_values()
        tuning_start = past_days.iloc[len(past_days) * 4 // 5]
        early = past.loc[past["timestamp"].lt(tuning_start)]
        tuning = past.loc[past["timestamp"].ge(tuning_start)]
        alpha, _ = select_alpha(tuning, fit(early), alphas, tolerance)
        scored = metrics(predict(window, fit(past), alpha), groups=True)
        comparisons.append(scored.assign(
            start=window["timestamp"].min(), end=window["timestamp"].max(),
            alpha=alpha, tuning_start=tuning_start, tuning_end=tuning["timestamp"].max(),
        ))
    return pd.concat(comparisons, ignore_index=True)
