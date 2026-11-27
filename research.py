from base64 import b64encode
from html import escape
from pathlib import Path
from string import Template
import json

import numpy as np
import pandas as pd

from analysis import METHODS, GROUPS


def paired_intervals(predictions, seed, repeats):
    """Оценить разброс разницы RMSE при повторном выборе фильмов.

    В каждой выборке набираем с возвращением столько фильмов, сколько их
    в группе. Все оценки выбранного фильма и оба прогноза идут вместе.
    Границы интервала берём по процентилям 2,5% и 97,5% разницы RMSE.
    """
    rng = np.random.default_rng(seed)
    subsets = {"Все": predictions}
    subsets.update({g: predictions.loc[predictions["group"].eq(g)] for g in GROUPS})
    rows = []
    for group, events in subsets.items():
        squared = pd.DataFrame({
            "movie_id": events["movie_id"],
            "raw": (events["movie_mean"] - events["rating"]) ** 2,
            "smooth": (events["smoothed_mean"] - events["rating"]) ** 2,
        })
        totals = squared.groupby("movie_id").agg(
            n=("raw", "size"),
            raw=("raw", "sum"),
            smooth=("smooth", "sum"),
        )
        result = {"group": group, "films": len(totals), "events": len(events),
                  "delta": np.nan, "low": np.nan, "high": np.nan}
        if not events.empty:
            rmse_raw = np.sqrt(totals["raw"].sum() / len(events))
            rmse_smooth = np.sqrt(totals["smooth"].sum() / len(events))
            result["delta"] = rmse_smooth - rmse_raw
        if len(totals) >= 2:
            draws = rng.integers(len(totals), size=(repeats, len(totals)))
            counts = totals.n.to_numpy()[draws].sum(axis=1)
            raw = np.sqrt(totals.raw.to_numpy()[draws].sum(axis=1) / counts)
            smoothed = np.sqrt(totals.smooth.to_numpy()[draws].sum(axis=1) / counts)
            result["low"], result["high"] = np.quantile(smoothed - raw, [0.025, 0.975])
        rows.append(result)
    return pd.DataFrame(rows)


def error_examples(predictions, movies):
    rare = predictions.loc[predictions["group"].eq("1–9")].copy()
    raw_error = (rare["movie_mean"] - rare["rating"]).abs()
    smooth_error = (rare["smoothed_mean"] - rare["rating"]).abs()
    rare["gain"] = raw_error - smooth_error
    ranked = rare.sort_values(["gain", "rating_id"], ascending=[False, True])
    examples = pd.concat([ranked.head(1), ranked.tail(1)]).drop_duplicates("rating_id")
    return examples.merge(movies[["movie_id", "title"]], on="movie_id", validate="many_to_one")


def num(value, digits=3):
    return "—" if pd.isna(value) else f"{value:.{digits}f}".replace(".", ",")


def table(headers, rows):
    head = "".join(f"<th scope='col'>{escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        body.append("<tr>" + cells + "</tr>")
    return "<div class='scroll'><table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"


def write_html(output, profile, alpha, history, rankings, scores, intervals, examples, repeats, validation, split):
    films = rankings["smoothed"][["movie_id", "title", "genre", "n_ratings", "raw_mean"]].copy()
    films = films.join(history.by_movie["rating_sum"], on="movie_id").fillna({"rating_sum": 0})
    first = rankings["raw_mean"].query("1 <= n_ratings <= 9").iloc[0]
    data = {
        "movies": json.loads(films.to_json(orient="records", force_ascii=False, double_precision=15)),
        "mean": history.global_mean, "alpha": alpha, "initial": int(first.movie_id),
    }
    interval_rows = []
    for row in intervals.itertuples():
        if row.events == 0:
            delta, interval = "нет наблюдений", "нет наблюдений"
        elif pd.isna(row.low):
            delta, interval = num(row.delta, 4), "для интервала нужен хотя бы ещё один фильм"
        else:
            delta = num(row.delta, 4)
            interval = f"{num(row.low, 4)} … {num(row.high, 4)}"
        interval_rows.append([row.group, row.films, row.events, delta, interval])
    interval_table = table(["Прошлых голосов", "Фильмов на тесте", "Оценок", "Δ RMSE", "95% интервал"],
                           interval_rows)
    error_table = table(["Фильм", "Голосов до теста", "Зритель поставил", "Обычный прогноз", "Сглаженный", "Выигрыш, баллы"],
        ([r.title, r.n_ratings, num(r.rating, 1), num(r.movie_mean, 2), num(r.smoothed_mean, 2), num(r.gain, 2)]
         for r in examples.itertuples()))
    if examples.empty:
        error_table = "<p>В этой группе нет наблюдений.</p>"
    all_scores = scores.loc[scores["group"].eq("Все")]
    metrics = table(["Прогноз", "RMSE", "MAE"],
                    ([METHODS[r.method], num(r.rmse, 4), num(r.mae, 4)] for r in all_scores.itertuples()))
    rare = intervals.set_index("group").loc["1–9"]
    if rare.events == 0:
        interpretation = "В тесте нет оценок фильмов с 1–9 прошлыми голосами."
    elif pd.isna(rare.low):
        interpretation = "Для оценки разброса в группе редких фильмов недостаточно разных фильмов."
    elif rare.low <= 0 <= rare.high:
        interpretation = "Интервал для редких фильмов включает ноль. На этой выборке направление эффекта остаётся неопределённым."
    elif rare.high < 0:
        interpretation = "У редких фильмов весь интервал ниже нуля. В повторных выборках по фильмам сглаживание обычно уменьшало ошибку."
    else:
        interpretation = "У редких фильмов весь интервал выше нуля. В повторных выборках по фильмам сглаживание обычно увеличивало ошибку."
    wins = validation.loc[validation["method"].eq("smoothed_mean")]
    calibration = " · ".join(f"α = {r.alpha} → {num(r.rmse,4)}" for r in wins.itertuples())
    source = Template(Path(__file__).with_name("report-template.html").read_text(encoding="utf-8"))
    graphic = b64encode((output / "results/figures/rating-vs-count.png").read_bytes()).decode()
    winner = rankings["smoothed"].iloc[0]
    html = source.substitute(
        data=json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"),
        sample=escape(first.title), initial_score=num(first.raw_mean, 2),
        initial_n=int(first.n_ratings), winner=escape(winner.title), winner_n=int(winner.n_ratings),
        winner_score=num(winner.smoothed_mean, 2), event_count=f"{profile['ratings']:,}".replace(",", " "),
        film_count=profile["movies"], user_count=profile["users"], alpha=alpha, metrics=metrics,
        intervals=interval_table, repeats=repeats, examples=error_table,
        interpretation=interpretation, calibration=calibration, graphic=graphic,
        split=" / ".join(str(row["n_events"]) for row in split),
    )
    (output / "report.html").write_text(html, encoding="utf-8")
