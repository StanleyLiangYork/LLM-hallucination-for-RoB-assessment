"""Character-overlap chunking and Okapi BM25 retrieval for trial reports."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from rank_bm25 import BM25Okapi


_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    text: str
    page_start: int
    page_end: int


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text or "")]


def chunk_pdf(path: str | Path, chunk_size: int = 1400, overlap: int = 200) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    page_texts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_texts.append(normalize_whitespace(page.extract_text() or ""))

    combined_parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    cursor = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        if not page_text:
            continue
        if combined_parts:
            combined_parts.append("\n")
            cursor += 1
        start = cursor
        combined_parts.append(page_text)
        cursor += len(page_text)
        page_spans.append((start, cursor, page_number))
    text = "".join(combined_parts)
    if not text:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        page_numbers = [
            number for page_start, page_end, number in page_spans
            if page_end > start and page_start < end
        ]
        chunks.append(
            TextChunk(
                chunk_id=len(chunks),
                text=text[start:end],
                page_start=min(page_numbers) if page_numbers else 0,
                page_end=max(page_numbers) if page_numbers else 0,
            )
        )
        if end >= len(text):
            break
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty corpus")
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        results: list[dict[str, Any]] = []
        for rank, index in enumerate(order[: min(top_k, len(order))], start=1):
            chunk = self.chunks[index]
            row = asdict(chunk)
            row.pop("text")
            row.update({"rank": rank, "score": float(scores[index]), "snippet": chunk.text})
            results.append(row)
        return results
