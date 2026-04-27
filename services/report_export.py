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
        self.figure.text(
            x,
            y,
            text,
            fontsize=size,
            fontweight=weight,
            color=color,
            family="DejaVu Sans",
            ha=ha,
            va=va,
        )

    def _start_page(self) -> None:
        self.page_number += 1
        self._rect(0, 0, 1, 1, _PAPER)
        self._polygon([(0.78, 1.0), (1.0, 1.0), (1.0, 0.74), (0.88, 0.82)], _CREAM, alpha=0.75)
        self._polygon([(0.0, 0.0), (0.16, 0.0), (0.0, 0.16)], "#d8f3f2", alpha=0.7)
        self._rect(0, 0.966, 1, 0.034, _NAVY)
        self._rect(0, 0.966, 0.22, 0.034, _TEAL, alpha=0.95)
        self._text(_LEFT, 0.983, "Valuation Master Pro", size=8.5, weight="bold", color="white", va="center")
        self._text(1 - _RIGHT, 0.983, f"PAGE {self.page_number}", size=7.5, color="#d4e7f7", ha="right", va="center")
        self._text(_LEFT, 0.045, "Educational use only. Not financial advice.", size=7, color=_MUTED)
        self._text(1 - _RIGHT, 0.045, self.title, size=7, color=_MUTED, ha="right")
        self.y = _TOP

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
        hero_height = 0.255
        y = self.y - hero_height
        self._patch(_LEFT, y, _CONTENT_WIDTH, hero_height, facecolor=_NAVY, radius=0.028)
        self._polygon(
            [(_LEFT + 0.55, self.y), (_LEFT + _CONTENT_WIDTH, self.y), (_LEFT + _CONTENT_WIDTH, y), (_LEFT + 0.72, y)],
            "#123d55",
            alpha=0.92,
        )
        self._polygon(
            [(_LEFT + 0.68, self.y), (_LEFT + _CONTENT_WIDTH, self.y), (_LEFT + _CONTENT_WIDTH, y + 0.085)],
            _TEAL,
            alpha=0.45,
        )
        self._rect(_LEFT, y, 0.018, hero_height, _ORANGE)
        ticker = _report_ticker(report.title)
        self._patch(_LEFT + 0.038, self.y - 0.075, 0.12, 0.075, facecolor=_CREAM, radius=0.019)
        self._text(_LEFT + 0.098, self.y - 0.036, ticker, size=14, weight="bold", color=_NAVY, ha="center", va="center")
        self._text(_LEFT + 0.038, self.y - 0.106, "EQUITY VALUATION REPORT", size=8, weight="bold", color=_ORANGE)
        self._text(_LEFT + 0.038, self.y - 0.142, report.title, size=25, weight="bold", color="white")
        self._text(_LEFT + 0.038, self.y - 0.183, "Concise investment snapshot with guardrails, technical context and sector lens.", size=9.5, color="#b9d8ee")
        if report.meta_lines:
            meta = "  |  ".join(report.meta_lines[:3])
            self._text(_LEFT + 0.038, self.y - 0.22, meta, size=8, color="#d8ecfb")
        self._patch(0.69, self.y - 0.19, 0.19, 0.072, facecolor="#07111d", edgecolor="#2b5f77", linewidth=0.8)
        self._text(0.785, self.y - 0.148, "PDF MEMO", size=12, weight="bold", color=_ORANGE, ha="center", va="center")
        self.y -= hero_height + 0.045

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
            if title not in {"Snapshot", "Sector lens", "Core metrics", "Technicals", "Caveats"}
        ]
        for title, lines in remaining_sections:
            canvas.section_card(title, lines)

        canvas.paragraph("Educational use only. This report is not financial advice.", color=_MUTED)
        canvas.finish()

    buffer.seek(0)
    return buffer.getvalue()
