"""Export helpers for polished valuation reports."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle


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
_PAPER = "#f7fbff"
_CARD = "#ffffff"
_NAVY = "#091624"
_NAVY_SOFT = "#10243a"
_TEAL = "#16a3a3"
_BLUE = "#2b78d4"
_ORANGE = "#ff9f1c"
_GREEN = "#168f52"


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
        self._rect(0, 0.955, 1, 0.045, _NAVY)
        self._rect(0, 0.955, 0.34, 0.045, _TEAL, alpha=0.85)
        self._text(_LEFT, 0.982, "Valuation Master Pro", size=9, weight="bold", color="white", va="center")
        self._text(1 - _RIGHT, 0.982, f"Page {self.page_number}", size=8, color="#d4e7f7", ha="right", va="center")
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
        self._text(_LEFT, self.y, label, size=15, weight="bold", color=_NAVY)
        self._rect(_LEFT, self.y - 0.031, 0.05, 0.004, _ORANGE)
        self.y -= 0.06

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
        self._patch(_LEFT, self.y - 0.19, _CONTENT_WIDTH, 0.18, facecolor=_NAVY, radius=0.022)
        self._rect(_LEFT, self.y - 0.19, 0.018, 0.18, _TEAL)
        self._text(_LEFT + 0.035, self.y - 0.035, report.title, size=24, weight="bold", color="white")
        self._text(_LEFT + 0.035, self.y - 0.082, "Professional valuation export", size=11, color="#b9d8ee")
        if report.meta_lines:
            meta = "  |  ".join(report.meta_lines[:3])
            self._text(_LEFT + 0.035, self.y - 0.13, meta, size=8.5, color="#d8ecfb")
        self._patch(0.705, self.y - 0.145, 0.19, 0.06, facecolor=_NAVY_SOFT, edgecolor="#244c67", linewidth=0.8)
        self._text(0.8, self.y - 0.111, "Export PDF", size=11, weight="bold", color=_ORANGE, ha="center", va="center")
        self.y -= 0.225

    def metric_cards(self, snapshot_lines: list[str]) -> None:
        metrics = [_split_bullet(line) for line in snapshot_lines if line.startswith("- ")]
        if not metrics:
            return

        card_gap = 0.014
        card_width = (_CONTENT_WIDTH - card_gap * 2) / 3
        card_height = 0.092

        for index, (label, value) in enumerate(metrics[:6]):
            if index % 3 == 0:
                self.ensure_space(card_height + 0.02)
                row_y = self.y - card_height
                self.y -= card_height + 0.018
            col = index % 3
            x = _LEFT + col * (card_width + card_gap)
            self._patch(x, row_y, card_width, card_height, facecolor=_CARD, edgecolor=_LINE, linewidth=0.8)
            self._text(x + 0.016, row_y + card_height - 0.026, label.upper(), size=7.4, weight="bold", color=_MUTED)
            self._text(x + 0.016, row_y + card_height - 0.056, value or "N/A", size=15, weight="bold", color=_INK)

    def section_card(self, title: str, lines: list[str], *, accent: str = _BLUE) -> None:
        if not lines:
            return

        rendered_lines: list[str] = []
        for line in lines:
            clean_line = line[2:].strip() if line.startswith("- ") else line.strip()
            rendered_lines.extend(_wrap_text(clean_line, 92))

        height = min(0.33, 0.08 + 0.023 * len(rendered_lines))
        self.ensure_space(height + 0.018)
        top = self.y
        y = top - height

        self._patch(_LEFT, y, _CONTENT_WIDTH, height, facecolor=_CARD, edgecolor=_LINE, linewidth=0.8, radius=0.017)
        self._rect(_LEFT, top - 0.041, _CONTENT_WIDTH, 0.0035, accent)
        self._text(_LEFT + 0.018, top - 0.022, title, size=12, weight="bold", color=_NAVY)

        line_y = top - 0.063
        for line in rendered_lines[:10]:
            self._text(_LEFT + 0.022, line_y, line, size=8.8, color=_INK)
            line_y -= 0.023
        if len(rendered_lines) > 10:
            self._text(_LEFT + 0.022, line_y, "...", size=9, color=_MUTED)

        self.y -= height + 0.018

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
        canvas.section_card("Sector lens", report.sections.get("Sector lens", []), accent=_TEAL)
        canvas.section_card("Core metrics", report.sections.get("Core metrics", []), accent=_ORANGE)
        canvas.section_card("Technical read", report.sections.get("Technicals", []), accent=_BLUE)
        canvas.section_card("Caveats and guardrails", report.sections.get("Caveats", []), accent=_GREEN)

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
