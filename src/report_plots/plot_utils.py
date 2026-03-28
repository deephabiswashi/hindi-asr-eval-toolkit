import json
import logging
import unicodedata
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns


LOGGER_NAME = "report_plots"
PREFERRED_FONTS = [
    "Nirmala UI",
    "Noto Sans Devanagari",
    "Mangal",
    "Aparajita",
    "Kohinoor Devanagari",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def choose_font_family() -> str:
    available = {font.name for font in fm.fontManager.ttflist}
    for candidate in PREFERRED_FONTS:
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def setup_plot_style() -> None:
    font_family = choose_font_family()
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": font_family,
            "axes.titlesize": 18,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def repair_text(text: str) -> str:
    text = str(text or "")
    markers = ("\u00e0\u00a4", "\u00e0\u00a5", "\u00ef\u00bf\u00bd")
    if any(marker in text for marker in markers):
        try:
            text = text.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    return unicodedata.normalize("NFKC", text)


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def annotate_bars(ax, fmt: str = "{:.2f}", rotation: int = 0) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if height is None:
            continue
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            rotation=rotation,
            fontsize=10,
        )

