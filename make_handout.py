"""Печатный отчёт основного прогона. Запуск: python make_handout.py."""
import csv
import json
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "runs" / "main" / "results"
WIDTH, HEIGHT = A4
LEFT = 48
BODY = WIDTH - 2 * LEFT
INK = "#202824"
GREEN = "#164F43"
MUTED = "#4C5D56"
LIGHT = "#E4EEE8"
LINE = "#BACAC0"
RUST = "#924329"
PAPER = "#FFFEFA"


def read(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def team_rows(path):
    """Прочитать две колонки таблицы участников из общего Markdown-файла."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip().startswith("|")]
    rows = []
    for line in lines[2:]:
        cells = re.split(r"(?<!\\)\|", line[1:].removesuffix("|"))
        if len(cells) != 2:
            raise ValueError("В строке TEAM.md нужны имя участника и его вклад")
        rows.append([cell.strip().replace(r"\|", "|") for cell in cells])
    return rows


def number(value, places=4):
    return f"{float(value):.{places}f}".replace(".", ",")


def integer(value):
    return f"{int(value):,}".replace(",", " ")


def register_fonts():
    # Пакет matplotlib из requirements.txt содержит эти шрифты для всех платформ.
    try:
        import matplotlib
        folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    except ImportError:
        folder = Path(os.environ.get("WINDIR", "")) / "Fonts"
    names = {"Text": "DejaVuSans", "Bold": "DejaVuSans-Bold", "Display": "DejaVuSerif"}
    for name, filename in names.items():
        pdfmetrics.registerFont(TTFont(name, str(folder / (filename + ".ttf"))))


def styled(text, font="Text", size=10.5, color=INK, leading=None, align=0):
    style = ParagraphStyle("text", fontName=font, fontSize=size, leading=leading or size * 1.43,
                           textColor=colors.HexColor(color), alignment=align, splitLongWords=1)
    return Paragraph(text, style)


class Page:
    def __init__(self, canvas, label):
        self.canvas = canvas
        canvas.setFillColor(colors.HexColor(PAPER))
        canvas.rect(0, 0, WIDTH, HEIGHT, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor(GREEN))
        canvas.setFont("Bold", 8.5)
        canvas.drawString(LEFT, HEIGHT - 36, "ВЕС ГОЛОСА")
        canvas.setFont("Text", 8.5)
        canvas.drawRightString(WIDTH - LEFT, HEIGHT - 36, label)
        self.line(47)
        self.y = 72

    def line(self, top):
        self.canvas.setStrokeColor(colors.HexColor(LINE))
        self.canvas.setLineWidth(0.6)
        self.canvas.line(LEFT, HEIGHT - top, WIDTH - LEFT, HEIGHT - top)

    def text(self, text, *, font="Text", size=10.5, color=INK, bottom=10):
        para = styled(text, font, size, color)
        _, height = para.wrap(BODY, HEIGHT)
        para.drawOn(self.canvas, LEFT, HEIGHT - self.y - height)
        self.y += height + bottom

    def title(self, text):
        self.text(text, font="Display", size=25, color=GREEN, bottom=14)

    def heading(self, text):
        self.text(text, font="Bold", size=12, color=GREEN, bottom=7)

    def table(self, heads, rows, widths, *, size=9.1, numeric=(), row_height=None):
        values = []
        for index, row in enumerate([heads, *rows]):
            values.append([
                styled(escape(str(cell)), "Bold" if index == 0 else "Text",
                       size - .4 if index == 0 else size, GREEN if index == 0 else INK,
                       leading=size * 1.3, align=TA_RIGHT if column in numeric else 0)
                for column, cell in enumerate(row)
            ])
        heights = ([0] + [row_height] * len(rows)) if row_height else None
        table = Table(values, colWidths=[BODY * w for w in widths], minRowHeights=heights)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(LIGHT)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, -1), .45, colors.HexColor(LINE)),
        ]))
        _, h = table.wrap(BODY, HEIGHT)
        table.drawOn(self.canvas, LEFT, HEIGHT - self.y - h)
        self.y += h + 15

    def callout(self, text):
        para = styled(text, "Text", 11, GREEN)
        _, h = para.wrap(BODY - 30, HEIGHT)
        self.canvas.setFillColor(colors.HexColor(LIGHT))
        self.canvas.roundRect(LEFT, HEIGHT - self.y - h - 22, BODY, h + 22, 5, stroke=0, fill=1)
        para.drawOn(self.canvas, LEFT + 15, HEIGHT - self.y - h - 11)
        self.y += h + 36

    def image(self, path, max_height):
        image = ImageReader(path)
        w, h = image.getSize()
        scale = min(BODY / w, max_height / h)
        self.canvas.drawImage(image, LEFT + (BODY - w * scale) / 2, HEIGHT - self.y - h * scale,
                              w * scale, h * scale)
        self.y += h * scale + 10

    def end(self):
        if self.y > HEIGHT - 46:
            raise ValueError(f"Содержание не помещается на странице {self.canvas.getPageNumber()}")
        self.line(HEIGHT - 39)
        self.canvas.setFont("Text", 8)
        self.canvas.setFillColor(colors.HexColor(MUTED))
        self.canvas.drawString(LEFT, 23, "ФильмРейтинг   /   основной эксперимент")
        self.canvas.drawRightString(WIDTH - LEFT, 23, f"{self.canvas.getPageNumber()} / 4")
        self.canvas.showPage()


def main():
    register_fonts()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    profile = cfg["profiles"]["main"]
    ranks = {key: read(f"rankings/{key}.csv") for key in ("popularity","raw_mean","mean_min_20","smoothed")}
    top_raw = ranks["raw_mean"][0]
    winner = ranks["smoothed"][0]
    changed = next(r for r in ranks["smoothed"] if r["movie_id"] == top_raw["movie_id"])
    validation = [r for r in read("validation_metrics.csv") if r["method"] == "smoothed_mean"]
    alpha = next(r["alpha"] for r in validation if r["selected"] == "True")
    scores = {(r["group"],r["method"]): r for r in read("test_metrics.csv")}
    intervals = {r["group"]: r for r in read("paired_intervals.csv")}
    rare = intervals["1–9"]
    split = read("split.csv")

    canvas = Canvas(str(ROOT / "film-rating.pdf"), pagesize=A4, pageCompression=1, invariant=1)
    canvas.setTitle("Вес голоса · ФильмРейтинг")
    canvas.setAuthor("ФильмРейтинг")
    canvas.setSubject("Рейтинг фильмов и точность прогноза зрительских оценок")

    p = Page(canvas, "Данные и исследовательский вопрос")
    p.text("Вес голоса", font="Display", size=38, color=GREEN, bottom=0)
    p.text("Сколько стоит одна пятёрка?", font="Display", size=21, bottom=16)
    p.callout(f"У фильма «{escape(top_raw['title'])}» средний балл {number(top_raw['raw_mean'],2)} "
              f"по одной оценке. У фильма «{escape(winner['title'])}» средний балл "
              f"{number(winner['raw_mean'],2)} по {winner['n_ratings']} оценкам. Какой рейтинг точнее "
              "предскажет голос следующего зрителя?")
    p.heading("Две гипотезы")
    p.text("1. Прошлые оценки самого фильма дают более точный прогноз, чем одно среднее по всему каталогу.<br/>"
           "2. Если голосов мало, прогноз выигрывает от сглаживания к общему среднему.")
    p.heading("Откуда взялись данные")
    p.text(f"{integer(profile['movies'])} фильмов, {integer(profile['users'])} зрителей и "
           f"{integer(profile['ratings'])} синтетических голосов за {profile['days']} дней. Популярность "
           "определяет шанс фильма получить голос. Базовый балл выбирается независимо. К нему добавляются "
           "смещение зрителя и случайный шум. Генератор округляет оценку до половины балла и ограничивает "
           "её шкалой 1–5. Каждый зритель оценивает конкретный фильм не более одного раза.")
    p.heading("Как разделены оценки")
    labels = {"train":"Обучение","validation":"Валидация","test":"Тест"}
    p.table(["Этап","Первый день","Последний день","Оценок"],
            [[labels[r["part"]],r["start"],r["end"],integer(r["n_events"])] for r in split],
            [.29,.245,.245,.22], numeric=(3,))
    p.text("Граница всегда проходит между датами. На обучении строятся рейтинги. На валидации "
           "выбирается коэффициент сглаживания. Перед тестом средние пересчитываются по двум первым "
           "отрезкам и фиксируются до конца тестового периода.", bottom=0)
    p.end()

    p = Page(canvas, "Рейтинги и сглаживание")
    p.title("Кто оказывается первым")
    labels = {"popularity":"Число голосов","raw_mean":"Средний балл","mean_min_20":"Средний при 20+","smoothed":"Сглаженный балл"}
    p.table(["Сортировка","Фильм","Голосов","Средний","Сглаженный"],
            [[labels[k],v[0]["title"],integer(v[0]["n_ratings"]),number(v[0]["raw_mean"],2),
              number(v[0]["smoothed_mean"],2)] for k,v in ranks.items()],
            [.245,.32,.13,.145,.16], numeric=(2,3,4))
    p.heading("Что меняют пять условных голосов")
    p.text("Сглаженный балл = (сумма баллов фильма + α × общее среднее) / (число голосов + α). "
           "Если фильм пока никто не оценил, прогноз равен общему среднему. При сумме 8, двух голосах, "
           "общем среднем 3 и α = 2 формула даёт (8 + 2 × 3) / (2 + 2) = 3,5.")
    p.text(f"При выбранном α = {alpha} рейтинг фильма «{escape(top_raw['title'])}» снижается с "
           f"{number(top_raw['raw_mean'],2)} до {number(changed['smoothed_mean'],2)}. В обычном топе "
           f"он на первом месте, после сглаживания занимает {changed['rank']}-е.")
    p.image(RESULTS / "figures/rating-vs-count.png", 236)
    p.text("По горизонтали отложено число голосов в логарифмической шкале. При одном-двух голосах "
           "оценки особенно сильно сдвигаются к среднему. У фильмов с сотнями голосов пары точек "
           "почти совпадают.", size=9.3, color=MUTED)
    p.heading("Выбор α")
    p.text("Ошибка RMSE на валидации: " + " · ".join(f"α = {r['alpha']} → {number(r['rmse'])}" for r in validation)
           + f". Выбран α = {alpha}.", bottom=0)
    p.end()

    p = Page(canvas, "Прогноз и ошибки")
    p.title("Что случилось на тесте")
    p.text("В таблице показаны RMSE / MAE в баллах. RMSE сильнее штрафует крупные промахи. "
           "MAE показывает среднюю абсолютную ошибку. Для обеих метрик меньший результат лучше.")
    groups = ["Все","0","1–9","10+"]
    methods = ["global_mean","movie_mean","smoothed_mean"]
    rows = []
    for g in groups:
        count = int(scores[g,methods[0]]["n_events"])
        cells = []
        for m in methods:
            s = scores[g,m]
            cells.append(f"{number(s['rmse'])} / {number(s['mae'])}" if count else "нет наблюдений")
        rows.append([g,integer(count),*cells])
    p.table(["Прошлых голосов","Оценок","Общее среднее","Среднее фильма","Сглаженное среднее"],
            rows,[.135,.13,.235,.245,.255],size=8.2,numeric=(1,))
    p.callout(f"У {rare['films']} фильмов с 1–9 прошлыми голосами на тесте было {rare['events']} оценок. "
              f"Разница RMSE «сглаженное минус обычное» составила {number(rare['delta'])} балла. "
              f"По всем {integer(intervals['Все']['events'])} оценкам разница составила "
              f"{number(intervals['Все']['delta'])} балла.")
    p.heading("В каких случаях сглаживание помогло")
    p.text("Из редкой группы взяты два события с наибольшим и наименьшим выигрышем в абсолютной ошибке. "
           "Положительный выигрыш означает, что прогноз приблизился к оценке зрителя.")
    ex = read("error_examples.csv")
    p.table(["Фильм","Голосов до теста","Оценка зрителя","Обычный прогноз","Сглаженный прогноз","Выигрыш"],
            [[r["title"],r["n_ratings"],number(r["rating"],1),number(r["movie_mean"],2),
              number(r["smoothed_mean"],2),number(r["gain"],2)] for r in ex],
            [.31,.13,.13,.145,.16,.125],size=8.3,numeric=(1,2,3,4,5))
    p.heading("Насколько результат зависит от состава фильмов")
    p.text(f"Для каждой группы построено {integer(cfg['analysis']['bootstrap_repeats'])} повторных выборок. "
           "Фильмы выбирались с возвращением. Если фильм попадал в выборку, вместе с ним брались все его "
           "тестовые голоса и прогнозы обоих методов.")
    p.text(f"У редких фильмов 95%-й интервал разницы RMSE идёт от {number(rare['low'])} до "
           f"{number(rare['high'])}. Интервал включает ноль. На этом тесте метод дал выигрыш, "
           "но при другом составе фильмов его знак может измениться. Интервал условен на рассчитанные "
           "прогнозы и не учитывает отдельно общих зрителей у разных фильмов.", bottom=0)
    p.end()

    p = Page(canvas, "Устойчивость и воспроизведение")
    p.title("Повторяется ли результат?")
    stability = read("period_stability.csv")
    rare_windows = [r for r in stability if r["group"]=="1–9" and r["method"]=="smoothed_mean"]
    plain = {r["start"]:r for r in stability if r["group"]=="1–9" and r["method"]=="movie_mean"}
    p.text("Проверка повторялась на трёх ранних отрезках. Для каждого окна коэффициент α выбирался "
           "по последней пятой части предшествующих дат. В таблице приведена RMSE для фильмов с 1–9 голосами.")
    p.table(["Начало окна","Конец окна","α","Оценок","Обычный","Сглаженный"],
            [[r["start"],r["end"],r["alpha"],r["n_events"],number(plain[r["start"]]["rmse"]),
              number(r["rmse"])] for r in rare_windows],[.205,.205,.08,.14,.175,.195],
            size=9.1,numeric=(2,3,4,5))
    overlaps = read("top_overlap.csv")
    p.heading("Состав десятки")
    p.text("В трёх парах повторных генераций состав фильмов и скрытые параметры сохранялись. "
           "Для обычного среднего в топ-10 совпали " + ", ".join(r["raw_overlap"] for r in overlaps)
           + " фильмов из десяти. Для сглаженного среднего совпали "
           + ", ".join(r["smoothed_overlap"] for r in overlaps) + ".")
    p.heading("Где повторить расчёт")
    p.text(f"Из папки проекта выполните python run.py --profile main. Seed основного набора — "
           f"{profile['seed']}. Точные прогнозы, метрики и срезы лежат в runs/main/results/. "
           "Полный порядок запуска приведён в README.md.", size=9.6)
    p.heading("Источники метода")
    imdb = "https://help.imdb.com/article/imdb/track-movies-tv/faq-for-imdb-votes/G67Y87TFYYP6TWAV"
    otexts = "https://otexts.com/fpp3/tscv.html"
    p.text(f'<a href="{imdb}" color="{GREEN}"><u>IMDb · Ratings FAQ</u></a> — взвешивание оценок в рейтингах фильмов.<br/>'
           f'<a href="{otexts}" color="{GREEN}"><u>R. Hyndman, G. Athanasopoulos · Forecasting: Principles and Practice, §5.10</u></a> '
           '— проверка прогнозов со сдвигающейся временной границей.', size=9.1, bottom=12)
    p.heading("Участники и фактический вклад")
    p.table(["Имя участника","Что сделал(а) и где посмотреть результат"],
            team_rows(ROOT / "TEAM.md"),[.37,.63],row_height=26)
    p.end()
    canvas.save()


if __name__ == "__main__":
    main()
