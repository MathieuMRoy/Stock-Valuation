"""Export helpers for valuation reports."""

from __future__ import annotations

from io import BytesIO
import re
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


_PAGE_WIDTH = 8.27
_PAGE_HEIGHT = 11.69
_LEFT_MARGIN = 0.08
_RIGHT_MARGIN = 0.08
_TOP_MARGIN = 0.94
_BOTTOM_MARGIN = 0.06


def _new_pdf_page() -> Figure:
    """Create a clean A4 figure used as one PDF page."""
    figure = Figure(figsize=(_PAGE_WIDTH, _PAGE_HEIGHT), facecolor="white")
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


def _line_style(raw_line: str) -> tuple[str, int, str, float, float]:
    """Return text, font size, weight, x-position and line spacing."""
    stripped = raw_line.strip()
    if stripped.startswith("# "):
        return _clean_markdown_text(stripped[2:]), 20, "bold", _LEFT_MARGIN, 0.035
    if stripped.startswith("## "):
        return _clean_markdown_text(stripped[3:]), 15, "bold", _LEFT_MARGIN, 0.03
    if stripped.startswith("- "):
        return f"- {_clean_markdown_text(stripped[2:])}", 10, "normal", _LEFT_MARGIN + 0.02, 0.021
    return _clean_markdown_text(stripped), 10, "normal", _LEFT_MARGIN, 0.021


def _wrap_width(font_size: int, x_position: float) -> int:
    """Approximate wrapped characters per line based on font and margin."""
    available_width = 1.0 - x_position - _RIGHT_MARGIN
    return max(44, int(100 * available_width * (10 / max(font_size, 1))))


def markdown_report_to_pdf_bytes(report_markdown: str, *, document_title: str = "Valuation report") -> bytes:
    """Render a lightweight Markdown valuation report as a paginated PDF."""
    buffer = BytesIO()
    figure = _new_pdf_page()
    y_position = _TOP_MARGIN

    with PdfPages(buffer) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = document_title
        metadata["Subject"] = "Valuation Master Pro export"
        metadata["Creator"] = "Valuation Master Pro"

        for raw_line in report_markdown.splitlines():
            if not raw_line.strip():
                y_position -= 0.014
                continue

            text, font_size, weight, x_position, line_spacing = _line_style(raw_line)
            wrapped_lines = textwrap.wrap(
                text,
                width=_wrap_width(font_size, x_position),
                break_long_words=False,
                replace_whitespace=False,
            ) or [text]

            for wrapped_line in wrapped_lines:
                if y_position < _BOTTOM_MARGIN:
                    pdf.savefig(figure)
                    figure = _new_pdf_page()
                    y_position = _TOP_MARGIN

                figure.text(
                    x_position,
                    y_position,
                    wrapped_line,
                    fontsize=font_size,
                    fontweight=weight,
                    color="#172033",
                    family="DejaVu Sans",
                )
                y_position -= line_spacing

        pdf.savefig(figure)

    buffer.seek(0)
    return buffer.getvalue()
