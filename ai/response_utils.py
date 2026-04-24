"""Text hygiene and response formatting helpers for AI specialist outputs."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from .router import normalize_intent_text


def text_from_parts(parts: list[Any]) -> str:
    chunks = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def normalize_finish_reason(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name).upper()
    return str(value).upper()


def merge_continuation_text(base_text: str, continuation_text: str) -> str:
    if not base_text:
        return continuation_text.strip()
    if not continuation_text:
        return base_text.strip()

    base = base_text.rstrip()
    continuation = continuation_text.strip()
    if not continuation:
        return base

    max_overlap = min(len(base), len(continuation), 200)
    for overlap in range(max_overlap, 20, -1):
        if base[-overlap:] == continuation[:overlap]:
            return f"{base}{continuation[overlap:]}".strip()

    if continuation.startswith(base[-80:]):
        return f"{base}{continuation[len(base[-80:]):]}".strip()

    return f"{base} {continuation}".strip()


def has_terminal_sentence_ending(text: str) -> bool:
    candidate = (text or "").rstrip()
    if not candidate:
        return False

    closers = ('"', "'", "`", ")", "]", "}", "Â»")
    while candidate and candidate[-1] in closers:
        candidate = candidate[:-1].rstrip()

    return candidate.endswith((".", "!", "?", "â€¦"))


def sentence_signature(text: str) -> str:
    normalized = normalize_intent_text(text or "")
    normalized = re.sub(r"`[^`]+`", " ", normalized)
    normalized = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", normalized)
    normalized = re.sub(r"[*_#>\-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def sentence_similarity(first: str, second: str) -> float:
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


def dedupe_repetitive_sentences(text: str) -> str:
    if not text:
        return ""

    paragraphs = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    cleaned_blocks: list[str] = []

    for block in paragraphs:
        if block.startswith("**Sources**"):
            cleaned_blocks.append(block)
            continue

        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        heading = ""
        body_lines = lines
        if lines[0].startswith("**") and len(lines) > 1:
            heading = lines[0]
            body_lines = lines[1:]

        body = " ".join(body_lines).strip()
        if not body:
            cleaned_blocks.append(heading or block)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", body)
        kept_sentences: list[str] = []
        seen_signatures: list[str] = []
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue
            signature = sentence_signature(clean_sentence)
            if signature and any(
                signature == previous
                or signature in previous
                or previous in signature
                or sentence_similarity(signature, previous) >= 0.9
                for previous in seen_signatures[-3:]
            ):
                continue
            kept_sentences.append(clean_sentence)
            if signature:
                seen_signatures.append(signature)

        rebuilt_body = " ".join(kept_sentences).strip() or body
        if heading:
            cleaned_blocks.append(f"{heading}\n{rebuilt_body}")
        else:
            cleaned_blocks.append(rebuilt_body)

    return "\n\n".join(block for block in cleaned_blocks if block.strip())


def looks_repetitive_specialist_answer(text: str, specialist_name: str) -> bool:
    lowered_name = (specialist_name or "").lower()
    if lowered_name not in {"comparison_agent", "peer_agent", "supervisor_agent"}:
        return False

    signatures: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        signature = sentence_signature(sentence)
        if len(signature) < 35:
            continue
        if any(
            signature == previous
            or signature in previous
            or previous in signature
            or sentence_similarity(signature, previous) >= 0.92
            for previous in signatures
        ):
            return True
        signatures.append(signature)

    return False


def looks_truncated_text(text: str | None) -> bool:
    if not text:
        return False

    stripped = text.rstrip()
    if len(stripped) < 80:
        return False

    if has_terminal_sentence_ending(stripped):
        return False

    if stripped.endswith((",", ";", ":", "-", "/", "(", "[", "{")):
        return True

    lowered = stripped.lower()
    dangling_endings = (
        " l'",
        " d'",
        " et",
        " ou",
        " de",
        " du",
        " des",
        " la",
        " le",
        " les",
        " pour",
        " avec",
        " sur",
        " dans",
        " par",
        " versus",
        " vs",
    )
    if any(lowered.endswith(ending) for ending in dangling_endings):
        return True

    if len(stripped) >= 60:
        return True

    return False


def is_specialist_answer_usable(text: str | None, specialist_name: str) -> bool:
    if not text or not text.strip():
        return False

    stripped = text.strip()
    if looks_truncated_text(stripped):
        return False
    if looks_repetitive_specialist_answer(stripped, specialist_name):
        return False

    lowered_name = (specialist_name or "").lower()
    if lowered_name in {"comparison_agent", "peer_agent", "news_agent", "market_signal_agent", "filings_agent"}:
        if len(stripped) < 90:
            return False
        if not any(punct in stripped for punct in [".", "!", "?", "\n", ":"]):
            return False

    return True


def is_chat_answer_usable(text: str | None) -> bool:
    if not text or not text.strip():
        return False

    stripped = text.strip()
    last_line = next((line.strip() for line in reversed(stripped.splitlines()) if line.strip()), "")

    if re.search(r"(?i)\b[ld]'$", last_line):
        return False

    if last_line.startswith(("- ", "* ", "â€¢ ")) or re.match(r"^\d+\.\s", last_line):
        return True

    if looks_truncated_text(stripped):
        return False

    if len(stripped) >= 80 and not has_terminal_sentence_ending(stripped):
        return False

    return True


def extract_model_text_and_finish_reason(response: Any) -> tuple[str | None, str]:
    text = getattr(response, "text", None)
    if text and str(text).strip():
        candidates = getattr(response, "candidates", None) or []
        finish_reason = ""
        if candidates:
            finish_reason = normalize_finish_reason(getattr(candidates[0], "finish_reason", None))
        return str(text).strip(), finish_reason

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        candidate_text = text_from_parts(getattr(getattr(candidate, "content", None), "parts", None) or [])
        if candidate_text:
            finish_reason = normalize_finish_reason(getattr(candidate, "finish_reason", None))
            return candidate_text.strip(), finish_reason

    return None, ""


def markdown_section(title: str, body: str | None) -> str:
    clean_body = (body or "").strip()
    if not clean_body:
        return ""
    return f"**{title}**\n{clean_body}"


def source_lines_from_refs(source_refs: dict[str, Any] | None, limit: int = 4) -> list[str]:
    refs = source_refs or {}
    lines: list[str] = []

    quote_page = refs.get("quote_page") or refs.get("current_quote_page") or {}
    if quote_page.get("url") and quote_page.get("label"):
        lines.append(f"- [{quote_page['label']}]({quote_page['url']})")

    sec_page = refs.get("sec_filings_page") or {}
    if sec_page.get("url") and sec_page.get("label"):
        lines.append(f"- [{sec_page['label']}]({sec_page['url']})")

    for item in (refs.get("market_news_links") or []) + (refs.get("press_release_links") or []):
        if item.get("title") and item.get("url"):
            lines.append(f"- [{item['title']}]({item['url']})")
        if len(lines) >= limit:
            break

    return lines[:limit]


def compose_professional_response(
    opening: str,
    sections: list[tuple[str, str | None]],
    *,
    source_refs: dict[str, Any] | None = None,
) -> str:
    blocks = [opening.strip()]
    for title, body in sections:
        section = markdown_section(title, body)
        if section:
            blocks.append(section)

    source_lines = source_lines_from_refs(source_refs)
    if source_lines:
        blocks.append("**Sources**\n" + "\n".join(source_lines))

    return "\n\n".join(block for block in blocks if block.strip())


def bullet_lines(items: list[str]) -> str:
    clean_items = [item.strip() for item in items if item and item.strip()]
    return "\n".join(f"- {item}" for item in clean_items)


def specialist_system_instruction(
    base_instruction: str,
    *,
    include_sources: bool = False,
) -> str:
    suffix = (
        "Reponds comme un analyste financier clair et professionnel. "
        "Commence par repondre directement a la question en une phrase. "
        "Ensuite, organise la suite en 2 a 4 sections courtes avec des titres markdown simples. "
        "Utilise seulement les chiffres qui aident vraiment a la decision. "
        "Explique ce que cela implique concretement pour l'investisseur. "
        "Evite les listes generiques, les preambules vagues et le jargon inutile. "
        "Ne repete jamais deux fois la meme idee sous deux formulations voisines. "
        "Si une donnee cle manque, signale-le clairement en une phrase. "
        "N'utilise pas de backticks autour des tickers, des chiffres ou des ratios."
    )
    if include_sources:
        suffix += " Si des URLs sont disponibles, termine par une section `Sources` avec quelques puces markdown."
    return f"{base_instruction} {suffix}"
