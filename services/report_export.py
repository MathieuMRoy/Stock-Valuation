"""Export helpers for polished valuation reports."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle


_PAGE_WIDTH = 8.27
_PAGE_HEIGHT = 11.69
_LEFT = 0.075
_RIGHT = 0.075
_TOP = 0.895
_BOTTOM = 0.08
_CONTENT_WIDTH = 1.0 - _LEFT - _RIGHT

_INK = "#102033"
_MUTED = "#5b6b7f"
_LINE = "#dbe7f4"
_PAPER = "#eef6fb"
_CARD = "#fbfdff"
_NAVY = "#091624"
_NAVY_SOFT = "#10243a"
_TEAL = "#16a3a3"
_BLUE = "#2b78d4"
_ORANGE = "#ff9f1c"
_GREEN = "#168f52"
_RED = "#c94141"
_CREAM = "#fff4dd"


@dataclass
class ParsedReport:
    """Structured Markdown report used by the PDF renderer."""

    title: str
    meta_lines: list[str]
    sections: dict[str, list[str]]


def _new_pdf_page() -> Figure:
    """Create a clean A4 figure used as one PDF page."""
    figure = Figure(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT), facecolor=_PAPER)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.axis("off")
    return figure


def _clean_markdown_text(value: str) -> str:
    """Convert lightweight Markdown syntax into readable PDF text."""
    text = value.strip()
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    return text


def _parse_report(report_markdown: str) -> ParsedReport:
    """Parse the simple export Markdown into title, metadata and sections."""
    title = "Valuation report"
    current_section: str | None = None
    meta_lines: list[str] = []
    sections: dict[str, list[str]] = {}

    for raw_line in report_markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# "):
            title = _clean_markdown_text(line[2:])
            current_section = None
            continue
        if line.startswith("## "):
            current_section = _clean_markdown_text(line[3:])
            sections.setdefault(current_section, [])
            continue
        if current_section:
            sections[current_section].append(_clean_markdown_text(line))
        else:
            meta_lines.append(_clean_markdown_text(line))

    return ParsedReport(title=title, meta_lines=meta_lines, sections=sections)


def _split_bullet(line: str) -> tuple[str, str]:
    """Split '- Label: value' into label/value parts."""
    clean_line = line[2:].strip() if line.startswith("- ") else line.strip()
    if ":" in clean_line:
        label, value = clean_line.split(":", 1)
        return label.strip(), value.strip()
    return clean_line, ""


def _wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        replace_whitespace=False,
    ) or [text]


def _report_ticker(title: str) -> str:
    """Infer the report ticker from the generated title."""
    token = title.split()[0].strip().upper()
    return token[:8] if token else "STOCK"


def _value_color(label: str, value: str) -> str:
    """Pick a semantic color for a metric value."""
    payload = f"{label} {value}".lower()
    if any(word in payload for word in ("severe", "high drawdown", "higher", "critical", "rich")):
        return _RED
    if any(word in payload for word in ("undervalued", "lower", "constructive", "+")):
        return _GREEN
    if "n/a" in payload:
        return _MUTED
    return _INK


class _PdfReportCanvas:
    """Small stateful renderer for a multi-page professional PDF."""

    def __init__(self, pdf: PdfPages, title: str):
        self.pdf = pdf
        self.title = title
        self.page_number = 0
        self.figure = _new_pdf_page()
        self.y = _TOP
        self._start_page()

    def _patch(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        facecolor: str,
        edgecolor: str = "none",
        linewidth: float = 0.0,
        radius: float = 0.012,
        alpha: float = 1.0,
    ) -> None:
        self.figure.patches.append(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle=f"round,pad=0.008,rounding_size={radius}",
                transform=self.figure.transFigure,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        )

    def _rect(self, x: float, y: float, width: float, height: float, color: str, alpha: float = 1.0) -> None:
        self.figure.patches.append(
            Rectangle(
                (x, y),
                width,
                height,
                transform=self.figure.transFigure,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
            )
        )

    def _polygon(self, points: list[tuple[float, float]], color: str, alpha: float = 1.0) -> None:
        self.figure.patches.append(
            Polygon(
                points,
                closed=True,
                transform=self.figure.transFigure,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
            )
        )

    def _text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int = 10,
        weight: str = "normal",
        color: str = _INK,
        ha: str = "left",
        va: str = "top",
    ) -> None:
        safe_text = str(text).replace("$", r"\$")
        self.figure.text(
            x,
            y,
            safe_text,
            fontsize=size,
            fontweight=weight,
            color=color,
            family="DejaVu Sans",
            ha=ha,
            va=va,
        )

    def _wrapped_text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        width: int,
        size: int = 10,
        weight: str = "normal",
        color: str = _INK,
        line_gap: float = 0.021,
    ) -> float:
        """Draw wrapped text and return the next y-position."""
        current_y = y
        for line in _wrap_text(text, width):
            self._text(x, current_y, line, size=size, weight=weight, color=color)
            current_y -= line_gap
        return current_y

    def _start_page(self) -> None:
        self.page_number += 1
        self._rect(0, 0, 1, 1, _PAPER)
        self._polygon([(0.78, 1.0), (1.0, 1.0), (1.0, 0.74), (0.88, 0.82)], _CREAM, alpha=0.75)
        self._polygon([(0.0, 0.0), (0.16, 0.0), (0.0, 0.16)], "#d8f3f2", alpha=0.7)
        if self.page_number == 1:
            self._text(_LEFT, 0.967, "VALUATION MASTER PRO", size=8.2, weight="bold", color=_NAVY, va="center")
            self._rect(_LEFT, 0.945, 0.055, 0.004, _TEAL)
            self._text(1 - _RIGHT, 0.967, "Equity research export", size=8, color=_MUTED, ha="right", va="center")
            self.y = 0.91
        else:
            self._rect(0, 0.966, 1, 0.034, _NAVY)
            self._rect(0, 0.966, 0.22, 0.034, _TEAL, alpha=0.95)
            self._text(_LEFT, 0.983, "Valuation Master Pro", size=8.5, weight="bold", color="white", va="center")
            self._text(1 - _RIGHT, 0.983, f"PAGE {self.page_number}", size=7.5, color="#d4e7f7", ha="right", va="center")
            self.y = _TOP
        self._text(_LEFT, 0.045, "Educational use only. Not financial advice.", size=7, color=_MUTED)
        self._text(1 - _RIGHT, 0.045, self.title, size=7, color=_MUTED, ha="right")

    def _save_and_new_page(self) -> None:
        self.pdf.savefig(self.figure)
        self.figure = _new_pdf_page()
        self._start_page()

    def ensure_space(self, needed: float) -> None:
        if self.y - needed < _BOTTOM:
            self._save_and_new_page()

    def heading(self, label: str) -> None:
        self.ensure_space(0.075)
        self._text(_LEFT, self.y, label.upper(), size=8, weight="bold", color=_TEAL)
        self._text(_LEFT, self.y - 0.026, "Investment memo", size=16, weight="bold", color=_NAVY)
        self._rect(_LEFT, self.y - 0.055, 0.07, 0.004, _ORANGE)
        self.y -= 0.082

    def paragraph(self, text: str, *, width: int = 104, color: str = _INK) -> None:
        for line in _wrap_text(text, width):
            self.ensure_space(0.024)
            self._text(_LEFT, self.y, line, size=9.5, color=color)
            self.y -= 0.022
        self.y -= 0.006

    def bullet_list(self, lines: list[str]) -> None:
        for line in lines:
            if not line.strip():
                continue
            clean_line = line[2:].strip() if line.startswith("- ") else line.strip()
            for index, wrapped in enumerate(_wrap_text(clean_line, 96)):
                self.ensure_space(0.024)
                prefix = "- " if index == 0 else "  "
                self._text(_LEFT + 0.012, self.y, f"{prefix}{wrapped}", size=9, color=_INK)
                self.y -= 0.022
        self.y -= 0.012

    def hero(self, report: ParsedReport) -> None:
        hero_height = 0.285
        y = self.y - hero_height
        self._patch(_LEFT + 0.006, y - 0.006, _CONTENT_WIDTH, hero_height, facecolor="#c8d9e5", radius=0.028, alpha=0.42)
        self._patch(_LEFT, y, _CONTENT_WIDTH, hero_height, facecolor=_CARD, edgecolor="#cfe1ef", linewidth=0.9, radius=0.028)
        left_width = 0.30
        right_x = _LEFT + left_width + 0.06
        right_edge = _LEFT + _CONTENT_WIDTH - 0.035
        self._patch(_LEFT, y, left_width, hero_height, facecolor=_NAVY, radius=0.028)
        self._polygon(
            [(_LEFT + 0.19, self.y), (_LEFT + 0.31, self.y), (_LEFT + 0.27, y), (_LEFT + 0.14, y)],
            "#123d55",
            alpha=0.62,
        )
        self._polygon(
            [(_LEFT + 0.25, self.y), (_LEFT + 0.34, self.y), (_LEFT + 0.28, y + 0.07)],
            _TEAL,
            alpha=0.24,
        )
        self._polygon(
            [(right_x + 0.26, self.y - 0.035), (_LEFT + _CONTENT_WIDTH, self.y - 0.035), (_LEFT + _CONTENT_WIDTH, y + 0.17)],
            _CREAM,
            alpha=0.7,
        )
        self._polygon(
            [(right_x + 0.39, self.y), (_LEFT + _CONTENT_WIDTH, self.y), (_LEFT + _CONTENT_WIDTH, y + 0.225)],
            "#dff6f4",
            alpha=0.64,
        )
        ticker = _report_ticker(report.title)
        self._text(_LEFT + 0.032, self.y - 0.05, "EQUITY RESEARCH", size=8, weight="bold", color=_ORANGE)
        self._text(_LEFT + 0.032, self.y - 0.102, ticker, size=32, weight="bold", color="white")
        self._text(_LEFT + 0.034, self.y - 0.15, "Valuation memo", size=12.5, weight="bold", color="#d8ecfb")
        self._rect(_LEFT + 0.034, self.y - 0.174, 0.105, 0.0045, _TEAL)
        self._wrapped_text(
            _LEFT + 0.034,
            self.y - 0.205,
            "Live context, guardrails and sector-aware framing.",
            width=32,
            size=8.0,
            color="#b9d8ee",
        )

        self._text(right_x, self.y - 0.055, "COMPANY SNAPSHOT", size=8, weight="bold", color=_TEAL)
        self._wrapped_text(right_x, self.y - 0.097, report.title, width=28, size=19, weight="bold", color=_NAVY, line_gap=0.039)
        self._wrapped_text(
            right_x,
            self.y - 0.168,
            "Presentation-ready export for valuation, risk, technicals and sector context.",
            width=48,
            size=8.8,
            color=_MUTED,
            line_gap=0.021,
        )
        if report.meta_lines:
            meta = "  |  ".join(report.meta_lines[:3])
            self._patch(right_x, y + 0.032, 0.305, 0.064, facecolor="#f5fbff", edgecolor="#d7e8f5", linewidth=0.7, radius=0.015)
            self._wrapped_text(right_x + 0.017, y + 0.073, meta, width=43, size=7.4, color=_MUTED, line_gap=0.016)
        badge_width = 0.083
        badge_x = right_edge - badge_width
        self._patch(badge_x, y + 0.041, badge_width, 0.09, facecolor=_NAVY, radius=0.019)
        self._text(badge_x + badge_width / 2, y + 0.095, "PDF", size=13, weight="bold", color=_ORANGE, ha="center", va="center")
        self._text(badge_x + badge_width / 2, y + 0.068, "MEMO", size=7.3, weight="bold", color="white", ha="center", va="center")
        self.y -= hero_height + 0.04

    def metric_cards(self, snapshot_lines: list[str]) -> None:
        metrics = [_split_bullet(line) for line in snapshot_lines if line.startswith("- ")]
        if not metrics:
            return

        card_gap = 0.016
        card_width = (_CONTENT_WIDTH - card_gap * 2) / 3
        card_height = 0.104

        for index, (label, value) in enumerate(metrics[:6]):
            if index % 3 == 0:
                self.ensure_space(card_height + 0.02)
                row_y = self.y - card_height
                self.y -= card_height + 0.018
            col = index % 3
            x = _LEFT + col * (card_width + card_gap)
            color = _value_color(label, value)
            self._patch(x + 0.004, row_y - 0.004, card_width, card_height, facecolor="#d6e4ef", alpha=0.35)
            self._patch(x, row_y, card_width, card_height, facecolor=_CARD, edgecolor="#cfe1ef", linewidth=0.8)
            self._rect(x, row_y + card_height - 0.007, card_width, 0.007, color)
            self._text(x + 0.016, row_y + card_height - 0.031, label.upper(), size=7.2, weight="bold", color=_MUTED)
            self._text(x + 0.016, row_y + card_height - 0.068, value or "N/A", size=16, weight="bold", color=color)

    def section_card(
        self,
        title: str,
        lines: list[str],
        *,
        accent: str = _BLUE,
        x: float | None = None,
        width: float | None = None,
        max_lines: int = 10,
    ) -> float:
        if not lines:
            return 0.0

        rendered_lines: list[str] = []
        target_width = width or _CONTENT_WIDTH
        wrap_width = max(42, int(92 * target_width / _CONTENT_WIDTH))
        for line in lines:
            clean_line = line[2:].strip() if line.startswith("- ") else line.strip()
            rendered_lines.extend(_wrap_text(clean_line, wrap_width))

        height = min(0.315, 0.08 + 0.023 * len(rendered_lines[:max_lines]))
        x_pos = x if x is not None else _LEFT
        card_width = width if width is not None else _CONTENT_WIDTH
        self.ensure_space(height + 0.018)
        top = self.y
        y = top - height

        self._patch(x_pos + 0.004, y - 0.004, card_width, height, facecolor="#d6e4ef", alpha=0.28)
        self._patch(x_pos, y, card_width, height, facecolor=_CARD, edgecolor="#cfe1ef", linewidth=0.8, radius=0.017)
        self._rect(x_pos, top - 0.044, card_width, 0.005, accent)
        self._text(x_pos + 0.017, top - 0.023, title.upper(), size=8, weight="bold", color=accent)

        line_y = top - 0.063
        for line in rendered_lines[:max_lines]:
            self._text(x_pos + 0.02, line_y, line, size=8.55, color=_INK)
            line_y -= 0.023
        if len(rendered_lines) > max_lines:
            self._text(x_pos + 0.02, line_y, "...", size=9, color=_MUTED)

        if x is None:
            self.y -= height + 0.018
        return height

    def section_grid(self, cards: list[tuple[str, list[str], str]]) -> None:
        """Render section cards in a two-column editorial grid."""
        gap = 0.022
        col_width = (_CONTENT_WIDTH - gap) / 2

        for index in range(0, len(cards), 2):
            row = cards[index : index + 2]
            heights = []
            saved_y = self.y
            for _, lines, _ in row:
                line_count = sum(len(_wrap_text((line[2:] if line.startswith("- ") else line).strip(), 44)) for line in lines)
                heights.append(min(0.315, 0.08 + 0.023 * min(line_count, 9)))
            row_height = max(heights or [0.0])
            self.ensure_space(row_height + 0.024)
            saved_y = self.y
            for col, (title, lines, accent) in enumerate(row):
                self.y = saved_y
                self.section_card(
                    title,
                    lines,
                    accent=accent,
                    x=_LEFT + col * (col_width + gap),
                    width=col_width,
                    max_lines=9,
                )
            self.y = saved_y - row_height - 0.024

    def finish(self) -> None:
        self.pdf.savefig(self.figure)


def markdown_report_to_pdf_bytes(report_markdown: str, *, document_title: str = "Valuation report") -> bytes:
    """Render a polished valuation report as a paginated PDF."""
    report = _parse_report(report_markdown)
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = document_title
        metadata["Subject"] = "Valuation Master Pro export"
        metadata["Creator"] = "Valuation Master Pro"

        canvas = _PdfReportCanvas(pdf, report.title)
        canvas.hero(report)
        canvas.section_card("Resume executif", report.sections.get("Resume executif", []), accent=_TEAL, max_lines=8)
        canvas.heading("Executive snapshot")
        canvas.metric_cards(report.sections.get("Snapshot", []))
        canvas.section_grid(
            [
                ("Sector lens", report.sections.get("Sector lens", []), _TEAL),
                ("Core metrics", report.sections.get("Core metrics", []), _ORANGE),
                ("Technical read", report.sections.get("Technicals", []), _BLUE),
                ("Caveats and guardrails", report.sections.get("Caveats", []), _GREEN),
            ]
        )

        remaining_sections = [
            (title, lines)
            for title, lines in report.sections.items()
            if title not in {"Resume executif", "Snapshot", "Sector lens", "Core metrics", "Technicals", "Caveats"}
        ]
        for title, lines in remaining_sections:
            canvas.section_card(title, lines)

        canvas.paragraph("Educational use only. This report is not financial advice.", color=_MUTED)
        canvas.finish()

    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Earnings calendar PDF – weekly "top N" one-pager
# ---------------------------------------------------------------------------

_DAY_NAMES_FR = {
    "Monday": "Lundi",
    "Tuesday": "Mardi",
    "Wednesday": "Mercredi",
    "Thursday": "Jeudi",
    "Friday": "Vendredi",
    "Saturday": "Samedi",
    "Sunday": "Dimanche",
}

_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_TIME_BUCKETS = ["Pre-market", "After close", "Not supplied"]

_BUCKET_LABELS = {
    "Pre-market": "Before Open",
    "After close": "After Close",
    "Not supplied": "TBD",
}


def _format_cap_short(value: float) -> str:
    """Format market cap as a compact string (e.g. '3.2T', '145B')."""
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.1f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    if value > 0:
        return f"${value:,.0f}"
    return "N/A"

def _bucket_for_time(time_label: str) -> str:
    """Normalize a time label into one of the three buckets."""
    lower = str(time_label or "").lower()
    if "pre" in lower:
        return "Pre-market"
    if "after" in lower:
        return "After close"
    return "Not supplied"

def _parse_eps_float(val_str: str) -> float | None:
    try:
        c = str(val_str).replace('$', '').replace(' ', '')
        if c.startswith('(') and c.endswith(')'):
            c = '-' + c[1:-1]
        return float(c)
    except Exception:
        return None

def _color_for_ticker(ticker: str) -> str:
    import hashlib
    colors = ["#4361ee", "#3f37c9", "#4cc9f0", "#f72585", "#7209b7", "#3a0ca3", "#f8961e", "#f3722c", "#f9c74f", "#90be6d", "#4d908e", "#277da1"]
    idx = int(hashlib.md5(ticker.encode()).hexdigest(), 16) % len(colors)
    return colors[idx]

def earnings_calendar_to_pdf_bytes(
    df: "pd.DataFrame",
    *,
    top_n: int = 10,
    title_override: str | None = None,
) -> bytes:
    """Render a landscape PDF calendar of the top-N earnings **per sub-column**."""
    import pandas as pd

    if df.empty:
        raise ValueError("Cannot generate PDF from an empty DataFrame.")

    # ---- data preparation ------------------------------------------------
    work = df.copy()
    work["_bucket"] = work["Time"].apply(_bucket_for_time)
    work["_day_order"] = work["Day"].map(lambda d: _DAY_ORDER.index(d) if d in _DAY_ORDER else 99)

    work = (
        work
        .sort_values(["_day_order", "_bucket", "Market Cap Value"], ascending=[True, True, False])
        .groupby(["Day", "_bucket"], sort=False)
        .head(top_n)
    )
    work = work.sort_values(["_day_order", "_bucket", "Market Cap Value"], ascending=[True, True, False])

    present_days = [d for d in _DAY_ORDER if d in work["Day"].values]
    if not present_days:
        raise ValueError("No weekday data found in the DataFrame.")

    # ---- determine date range for title ----------------------------------
    min_date = work["Date"].min()
    max_date = work["Date"].max()
    if title_override:
        page_title = title_override
    elif min_date == max_date:
        page_title = f"Top {top_n} Earnings – {min_date.strftime('%b %d, %Y')}"
    else:
        page_title = f"Top {top_n} Earnings – {min_date.strftime('%b %d')} au {max_date.strftime('%b %d, %Y')}"

    # ---- page geometry ---------------------------------------------------
    pw, ph = 16.0, 9.5  # true landscape, plenty of horizontal space
    fig = Figure(figsize=(pw, ph), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")

    # header band
    header_h = 0.08
    fig.patches.append(Rectangle((0, 1 - header_h), 1, header_h, transform=fig.transFigure, facecolor=_NAVY, edgecolor="none"))
    fig.text(0.5, 1 - header_h / 2, page_title, fontsize=24, fontweight="bold", color="white", family="DejaVu Sans", ha="center", va="center")
    fig.text(0.98, 1 - header_h / 2, "Valuation Master Pro", fontsize=10, color="#8aafcf", family="DejaVu Sans", ha="right", va="center")

    fig.text(0.5, 0.012, "Educational use only \u2013 Not financial advice.", fontsize=8, color=_MUTED, family="DejaVu Sans", ha="center")

    # ---- column layout ---------------------------------------------------
    margin_x = 0.015
    margin_top = header_h + 0.015
    margin_bot = 0.03
    col_gap = 0.008
    usable_w = 1.0 - 2 * margin_x
    n_cols = len(present_days)
    col_w = (usable_w - col_gap * (n_cols - 1)) / n_cols
    content_top = 1 - margin_top
    content_bot = margin_bot

    card_h = 0.076  # taller card relative to page height
    card_gap = 0.006

    from matplotlib.patches import Circle

    for col_idx, day_name in enumerate(present_days):
        col_x = margin_x + col_idx * (col_w + col_gap)

        # day header bar
        day_header_h = 0.04
        day_y = content_top - day_header_h
        fig.patches.append(FancyBboxPatch(
            (col_x, day_y), col_w, day_header_h,
            boxstyle="round,pad=0,rounding_size=0.008",
            transform=fig.transFigure, facecolor=_TEAL, edgecolor="none"
        ))
        day_label = _DAY_NAMES_FR.get(day_name, day_name)
        fig.text(col_x + col_w / 2, day_y + day_header_h / 2, day_label.upper(), fontsize=12, fontweight="bold", color="white", family="DejaVu Sans", ha="center", va="center")

        day_events = work[work["Day"] == day_name]
        has_pre = not day_events[day_events["_bucket"] == "Pre-market"].empty
        has_tbd = not day_events[day_events["_bucket"] == "Not supplied"].empty
        has_post = not day_events[day_events["_bucket"] == "After close"].empty

        needs_pre_col = has_pre or has_tbd
        needs_post_col = has_post

        sub_cols = []
        if needs_pre_col and not needs_post_col:
            sub_cols.append(("Pre-market", col_x, col_w))
        elif needs_post_col and not needs_pre_col:
            sub_cols.append(("After close", col_x, col_w))
        else:
            half_w = (col_w - 0.006) / 2
            sub_cols.append(("Pre-market", col_x, half_w))
            sub_cols.append(("After close", col_x + half_w + 0.006, half_w))

        sub_h = 0.025
        sub_y = day_y - sub_h - 0.004

        for bucket, sx, sw in sub_cols:
            fig.patches.append(Rectangle((sx, sub_y), sw, sub_h, transform=fig.transFigure, facecolor="#e8f0f8", edgecolor="#c5d5e5", linewidth=0.6))
            fig.text(sx + sw / 2, sub_y + sub_h / 2, _BUCKET_LABELS[bucket], fontsize=8, fontweight="bold", color=_MUTED, family="DejaVu Sans", ha="center", va="center")

            card_start_y = sub_y - 0.008
            bucket_events = day_events[day_events["_bucket"] == bucket]
            
            # If Before Open, also mix in TBD at the end
            if bucket == "Pre-market" and has_tbd:
                bucket_events = pd.concat([bucket_events, day_events[day_events["_bucket"] == "Not supplied"]])

            card_y = card_start_y

            for _, row in bucket_events.iterrows():
                if card_y - card_h < content_bot:
                    break
                cy = card_y - card_h

                is_tbd = row.get("_bucket") == "Not supplied"
                bg_color = "#fafafa" if is_tbd else "white"

                fig.patches.append(FancyBboxPatch(
                    (sx + 0.002, cy), sw - 0.004, card_h,
                    boxstyle="round,pad=0.002,rounding_size=0.006",
                    transform=fig.transFigure, facecolor=bg_color, edgecolor="#d0dde8", linewidth=0.8
                ))

                accent_color = _BLUE if bucket == "Pre-market" else _ORANGE
                if is_tbd:
                    accent_color = _MUTED
                    
                fig.patches.append(Rectangle(
                    (sx + 0.002, cy), 0.004, card_h,
                    transform=fig.transFigure, facecolor=accent_color, edgecolor="none",
                ))

                ticker_text = str(row.get("Ticker", "")).replace("$", r"\$")

                # Ticker text (Top Left)
                fig.text(sx + 0.012, cy + card_h - 0.018, ticker_text, fontsize=13, fontweight="bold", color=_NAVY, family="DejaVu Sans", ha="left", va="top")

                # Company name (Below Ticker)
                company_name = str(row.get("Company Name", "")).replace("$", r"\$")
                if len(company_name) > 25:
                    company_name = company_name[:23] + ".."
                fig.text(sx + 0.012, cy + card_h - 0.038, company_name, fontsize=6.5, color="#7d9db5", family="DejaVu Sans", ha="left", va="top")

                # Market Cap (Top Right)
                cap_val = float(row.get("Market Cap Value", 0) or 0)
                cap_text = _format_cap_short(cap_val).replace("$", r"\$")
                fig.text(sx + sw - 0.006, cy + card_h - 0.018, cap_text, fontsize=8.5, fontweight="bold", color=_TEAL, family="DejaVu Sans", ha="right", va="top")

                # EPS line (Bottom Left)
                raw_est = str(row.get("EPS Forecast", "N/A") or "N/A")
                raw_ly = str(row.get("Last Year EPS", "N/A") or "N/A")
                eps_forecast = raw_est.replace("$", r"\$")
                last_year_eps = raw_ly.replace("$", r"\$")
                
                eps_line = f"Est: {eps_forecast}  |  LY: {last_year_eps}"
                fig.text(sx + 0.012, cy + 0.012, eps_line, fontsize=7.5, color="#3a6b8c", family="DejaVu Sans", ha="left", va="bottom")

                card_y = cy - card_gap - card_gap

    # ---- serialize -------------------------------------------------------
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        meta = pdf.infodict()
        meta["Title"] = page_title
        meta["Creator"] = "Valuation Master Pro"
        pdf.savefig(fig)
    buf.seek(0)
    return buf.getvalue()
