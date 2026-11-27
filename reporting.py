from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import GROUPS, METHODS, TOPS


COLORS = ["#456B9B", "#127E78", "#BD592D"]


def csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def charts(folder, movies, train, tops, scores):
    folder.mkdir(parents=True, exist_ok=True)
    movie_counts = train.groupby("movie_id").size().reindex(movies["movie_id"], fill_value=0)
    bins = [0, 1, 10, 20, 50, 100, 200, 500, np.inf]
    counts, _ = np.histogram(movie_counts, bins=bins)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(["0", "1–9", "10–19", "20–49", "50–99", "100–199", "200–499", "500+"],
                  counts, color=COLORS[0])
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set(title="Сколько оценок получили фильмы в обучающем периоде",
           xlabel="Число оценок у фильма", ylabel="Число фильмов")
    save_plot(fig, folder / "training-counts.png")

    rated = tops["raw_mean"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.scatter(rated["n_ratings"], rated["raw_mean"], color=COLORS[0], alpha=0.56,
               label="Обычное среднее", s=26)
    ax.scatter(rated["n_ratings"], rated["smoothed_mean"], color=COLORS[2], alpha=0.56,
               label="Сглаженное среднее", s=26)
    ax.set(xscale="log", ylim=(0.85, 5.15), xlabel="Число оценок (логарифмическая шкала)",
           ylabel="Балл фильма", title="Как сглаживание меняет балл фильма")
    ax.legend(loc="lower right")
    save_plot(fig, folder / "rating-vs-count.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    groups = ["Все", *GROUPS]
    positions = np.arange(len(groups))
    for index, (method, label) in enumerate(METHODS.items()):
        data = scores.loc[scores["method"].eq(method)].set_index("group").loc[groups]
        ax.bar(positions + (index - 1) * 0.24, data["rmse"], width=0.23,
               label=label, color=COLORS[index])
    totals = scores.loc[scores["method"].eq("global_mean")].set_index("group")["n_events"]
    labels = [f"{g}\nn = {totals[g]}" if totals[g] else f"{g}\nнет наблюдений" for g in groups]
    ax.set_xticks(positions, labels)
    ax.set(title="Ошибка прогноза на тесте", xlabel="Число известных оценок у фильма",
           ylabel="RMSE, баллы")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
    save_plot(fig, folder / "test-rmse.png")


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def number(value, digits=4):
    return "—" if pd.isna(value) else f"{value:.{digits}f}".replace(".", ",")


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def count_table(parts):
    total = sum(map(len, parts))
    return markdown_table(
        ["Часть", "Оценок", "Доля", "Первая дата", "Последняя дата"],
        ([name, len(part), f"{len(part)/total:.1%}".replace(".", ","),
          part["timestamp"].min().date(), part["timestamp"].max().date()]
         for name, part in zip(["Обучение", "Валидация", "Тест"], parts)),
    )


def comparison(delta, subject, reference):
    if np.isclose(delta, 0, atol=1e-10):
        return f"{subject} и {reference} дали одинаковую RMSE."
    if delta < 0:
        return f"RMSE у метода «{subject}» ниже, чем у метода «{reference}», на {number(-delta)} балла."
    return f"RMSE у метода «{subject}» выше, чем у метода «{reference}», на {number(delta)} балла."


def write_report(path, profile_name, profile, parts, alpha, validation, scores, tops,
                 genre_stats, periods, overlaps, checks, intervals, test_predictions):
    train, _, test = parts
    counts = scores.drop_duplicates("group").set_index("group")["n_events"]
    test_rows = []
    for group, rows in scores.groupby("group", sort=False):
        by_method = rows.set_index("method")
        values = [
            f"{number(by_method.at[m, 'rmse'])} / {number(by_method.at[m, 'mae'])}"
            if counts[group] else "нет наблюдений"
            for m in METHODS
        ]
        test_rows.append([group, counts[group], *values])

    leaders = []
    for method, ranking in tops.items():
        first = ranking.iloc[0]
        leaders.append([TOPS[method], first["title"], first["n_ratings"],
                        number(first["raw_mean"], 2), number(first["smoothed_mean"], 2)])

    example = tops["raw_mean"].query("1 <= n_ratings <= 9").iloc[0]
    smoothed_rank = tops["smoothed"].set_index("movie_id").at[example.movie_id, "rank"]
    sample_sum = train.loc[train["movie_id"].eq(example.movie_id), "rating"].sum()
    global_mean = train["rating"].mean()
    top_popular = set(tops["popularity"].head(10)["movie_id"])
    top_rated = set(tops["raw_mean"].head(10)["movie_id"])
    candidates = validation.loc[validation["method"].eq("smoothed_mean")]
    val_rows = [[r.alpha, number(r.rmse), number(r.mae)] for r in candidates.itertuples()]

    rmse = scores.pivot(index="group", columns="method", values="rmse")
    window_rows = []
    for (start, end), frame in periods.groupby(["start", "end"], sort=True):
        rare = frame.loc[frame["group"].eq("1–9")].set_index("method")
        window_rows.append([start.date(), end.date(), frame["alpha"].iloc[0],
                            rare.at["movie_mean", "n_events"], number(rare.at["movie_mean", "rmse"]),
                            number(rare.at["smoothed_mean", "rmse"])])

    stability = ""
    if not overlaps.empty:
        stability = "\n\n".join([
            "В дополнительном опыте каталог, скрытые параметры и выбранный α зафиксированы. "
            "Для каждой пары запусков генератор заново выбирает пары зритель–фильм и разыгрывает оценки. "
            "Число общих фильмов в двух десятках показывает, насколько устойчив их состав.",
            markdown_table(["Пара запусков", "Совпадений в обычном топ-10", "В сглаженном топ-10"],
                           ([f"{r.seed_a} и {r.seed_b}", r.raw_overlap, r.smoothed_overlap]
                            for r in overlaps.itertuples())),
        ])

    # Положительный выигрыш в абсолютной ошибке означает более точный прогноз.
    gain = ((test_predictions["movie_mean"] - test_predictions["rating"]).abs()
            - (test_predictions["smoothed_mean"] - test_predictions["rating"]).abs())
    error_rows = []
    for group in GROUPS:
        values = gain.loc[test_predictions["group"].eq(group)]
        better = int(values.gt(1e-10).sum())
        worse = int(values.lt(-1e-10).sum())
        error_rows.append([group, len(values), better, worse, len(values) - better - worse,
                           number(values.median(), 3) if len(values) else "—"])

    first_delta = rmse.at["Все", "movie_mean"] - rmse.at["Все", "global_mean"]
    first_result = comparison(first_delta, "Среднее фильма", "Общее среднее")
    rare_interval = intervals.set_index("group").loc["1–9"]
    if pd.isna(rare_interval.low):
        rare_evidence = "Для оценки разброса в этой группе недостаточно разных фильмов."
    elif rare_interval.low <= 0 <= rare_interval.high:
        rare_evidence = (
            "Интервал включает ноль. Знак разницы зависит от того, какие фильмы попали в выборку, "
            "поэтому уверенно переносить преимущество одного метода на новые фильмы пока рано."
        )
    else:
        direction = "ниже" if rare_interval.high < 0 else "выше"
        rare_evidence = (
            f"Весь интервал лежит {direction} нуля. При повторном выборе фильмов знак разницы "
            "сохранялся в центральных 95% повторов."
        )
    rare_result = (
        f"В группе с 1–9 голосами RMSE обычного среднего составила {number(rmse.at['1–9', 'movie_mean'])}, "
        f"сглаженного — {number(rmse.at['1–9', 'smoothed_mean'])}. "
        f"Оценок в тесте — {int(rare_interval.events)}, разных фильмов — {int(rare_interval.films)}."
        if counts["1–9"] else "В тесте нет оценок для фильмов с 1–9 голосами в истории."
    )

    team = Path(__file__).with_name("TEAM.md").read_text(encoding="utf-8-sig").strip()
    names = {"main": "Основной", "quick": "Пробный", "fallback": "Запасной"}
    report = f"""# Вес голоса · ФильмРейтинг

У одного фильма единственный зритель поставил пятёрку. У другого средняя оценка 4,5 по сотне голосов. Какой балл лучше предсказывает, что поставит следующий зритель? На этот вопрос можно ответить, если рассчитать оба рейтинга по прошлым оценкам и проверить их на более поздних.

В реальном кинокаталоге такое сравнение зависит от способа подсчёта. Например, IMDb публикует взвешенную оценку и отдельно показывает распределение голосов. В проекте используется открытая учебная формула, для которой можно проверить каждый шаг на своих данных. [IMDb, Ratings FAQ](https://help.imdb.com/article/imdb/track-movies-tv/faq-for-imdb-votes/G67Y87TFYYP6TWAV)

**Первая гипотеза.** Среднее конкретного фильма лучше предсказывает его следующую оценку, чем среднее по всем голосам каталога. **Вторая гипотеза.** Для фильмов с 1–9 прошлыми голосами сглаживание к общему среднему снижает ошибку прогноза.

## Данные и постановка опыта

{names[profile_name]} вариант включает {profile['movies']} фильмов, {profile['users']} зрителей и {profile['ratings']} вымышленных оценок за {profile['days']} дней. Базовый балл фильма и вес его популярности выбираются независимо. Популярные фильмы чаще получают голоса. У каждого зрителя есть постоянная склонность завышать или занижать баллы. К базовому баллу прибавляются это смещение и случайный шум, затем результат округляется с шагом 0,5 и ограничивается диапазоном от 1 до 5. Каждый зритель может оценить конкретный фильм один раз. Прогнозы строятся по зарегистрированным оценкам. Скрытые параметры хранятся отдельно.

Границы разбиения проведены по датам в пропорции примерно 60/20/20. Все голоса одного дня входят в одну часть. Обучение даёт исходные рейтинги, валидация служит для выбора силы сглаживания, тест — для окончательного сравнения. При такой постановке для прогноза доступны только более ранние наблюдения. Тот же принцип лежит в основе проверки с последовательным сдвигом временной границы. [Hyndman, Athanasopoulos, «Time series cross-validation»](https://otexts.com/fpp3/tscv.html)

{count_table(parts)}

## Как один фильм оказывается в четырёх разных списках

{markdown_table(["Сортировка", "Первое место", "Голосов", "Средний балл", "После сглаживания"], leaders)}

У десяти лидеров по числу голосов и десяти лидеров по среднему баллу число общих фильмов равно {len(top_popular & top_rated)}. Первый список измеряет популярность, второй отражает мнение проголосовавших зрителей. Порог в 20 голосов исключает слабо оценённые фильмы из списка. Сглаживание сохраняет их в сравнении и учитывает объём истории в самом балле. Полные списки лежат в `results/rankings/`.

## Сглаживание и выбор коэффициента

Для фильма с суммой оценок *S* и числом голосов *n* сглаженный балл равен **(S + α × μ) / (n + α)**. Здесь μ — среднее по всем прошлым голосам. Коэффициент α показывает, сколько условных голосов со средним баллом каталога добавлено к истории фильма. При нулевой истории прогноз равен μ. Контрольный расчёт при сумме 8, двух голосах, μ = 3 и α = 2 даёт (8 + 2 × 3) / (2 + 2) = **3,5**.

{markdown_table(["α", "RMSE на валидации", "MAE на валидации"], val_rows)}

По валидационной RMSE выбран **α = {alpha}**. Если значения отличаются от минимума не более чем на 0,001, выбирается больший α. У фильма «{example.title}» в обучении число голосов равно {example.n_ratings}, их сумма — {number(sample_sum, 1)}, а μ = {number(global_mean, 3)}. После сглаживания балл равен ({number(sample_sum, 1)} + {alpha} × {number(global_mean, 3)}) / ({example.n_ratings} + {alpha}) ≈ **{number(example.smoothed_mean, 2)}**. Место этого фильма в обычном рейтинге — {example['rank']}, в сглаженном — {smoothed_rank}.

![Обычный и сглаженный баллы фильмов в зависимости от числа голосов](results/figures/rating-vs-count.png)

## Что произошло на тесте

После выбора α средние пересчитаны по обучению и валидации вместе и зафиксированы до конца теста. Каждый метод прогнозирует одни и те же оценки. В ячейках приведены **RMSE / MAE** в баллах. MAE показывает среднюю абсолютную ошибку, RMSE придаёт больший вес крупным промахам. Чем меньше значение, тем точнее прогноз.

{markdown_table(["Прошлых голосов", "Оценок в тесте", *METHODS.values()], test_rows)}

Чтобы увидеть типичную ошибку, для каждого тестового голоса из абсолютной ошибки обычного среднего вычтена ошибка сглаженного. Положительная разница означает выигрыш от сглаживания. В последней колонке — медиана этих разниц в баллах.

{markdown_table(["Прошлых голосов", "Оценок", "Точнее после сглаживания", "Хуже", "Без изменения", "Медианный выигрыш"], error_rows)}

На фильмы с десятью и более прошлыми голосами пришлось {counts['10+']} из {len(test)} тестовых оценок. Поэтому небольшая группа фильмов с малым числом голосов слабо влияет на общую RMSE, даже когда для отдельного прогноза поправка заметна.

## Проверка на других периодах и историях голосования

Для каждого раннего периода коэффициент выбран заново на последних 20% предшествующих дат. После выбора средние рассчитаны по всей истории, доступной к началу этого периода. В таблице приведены результаты для фильмов с 1–9 прошлыми голосами.

{markdown_table(["С", "По", "Выбранный α", "Оценок", "RMSE среднего фильма", "RMSE сглаживания"], window_rows)}

{stability}

## Почему жанры получили разные баллы

{markdown_table(["Жанр", "Фильмов в каталоге", "Оценок в обучении", "Средний балл"],
    ([r.genre, r.n_movies, r.n_events, number(r.mean_rating, 2)] for r in genre_stats.itertuples()))}

Жанровое среднее рассчитано по отдельным голосам, поэтому популярные фильмы сильнее влияют на результат своего жанра. Жанр присвоен независимо от качества фильма. Эти различия описывают состав конкретной выборки.

## Разброс результата и выводы

В таблице Δ RMSE равна RMSE сглаженного прогноза минус RMSE среднего фильма. Отрицательное значение означает выигрыш от сглаживания. Внутри каждой группы проведён парный бутстрэп по фильмам. При каждом повторе случайно выбирались фильмы с возвращением, причём в выборку попадали все их тестовые оценки и оба соответствующих прогноза. Нижняя и верхняя границы — процентили 2,5% и 97,5% полученных разниц.

{markdown_table(["Прошлых голосов", "Фильмов", "Оценок", "Δ RMSE", "Нижняя граница", "Верхняя граница"],
    ([r.group, r.films, r.events, number(r.delta), number(r.low), number(r.high)] for r in intervals.itertuples()))}

**По первой гипотезе.** {first_result} Это сравнение относится к следующим голосам в заданной модели зрителей и фильмов.

**По второй гипотезе.** {rare_result} {rare_evidence}

Прогнозы зафиксированы на весь тест. Интервал характеризует разброс при смене состава фильмов. Зависимость между оценками одного зрителя у разных фильмов отдельно не моделируется. Для переноса выводов на реальный кинокаталог понадобятся наблюдения за его зрителями.

## Воспроизведение

Параметры генератора заданы в `config.json`, seed этого набора — {profile['seed']}. При запуске выполнено {len(checks)} автоматических проверок. Команды приведены в README.

{team}
"""
    Path(path).write_text(report, encoding="utf-8")
