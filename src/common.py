"""Tiện ích dùng chung cho Day 7: đọc/ghi JSON, chuẩn hóa text và định nghĩa "hit".

Định nghĩa hit (dùng thống nhất cho mọi phương pháp retrieval và mọi metric):
  một chunk "hit" một phần tử gold khi
    1. chunk.source_id == gold.source_id
    2. gold.page nằm trong [chunk.page, chunk.metadata.page_end]  (chunk theo Điều có thể vắt trang)
    3. MỌI đoạn trong gold.evidence_all đều xuất hiện trong chunk.text sau khi chuẩn hóa
       NFC và bỏ toàn bộ khoảng trắng (lớp text PDF Công báo chen khoảng trắng giữa âm tiết).
  Các phần tử trong `gold` là evidence THAY THẾ: hit bất kỳ phần tử nào là đủ.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

_WS = re.compile(r"\s+")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def squash(text: str) -> str:
    """NFC + casefold + bỏ mọi khoảng trắng: dùng để so khớp evidence."""
    return _WS.sub("", unicodedata.normalize("NFC", text)).casefold()


def find_exact_span(haystack: str, needle: str) -> str | None:
    """Tìm `needle` trong `haystack` bỏ qua khác biệt khoảng trắng; trả về đoạn NGUYÊN VĂN trong haystack."""
    text = unicodedata.normalize("NFC", haystack)
    keep = [(ch.casefold(), i) for i, ch in enumerate(text) if not ch.isspace()]
    flat = "".join(c for c, _ in keep)
    target = squash(needle)
    pos = flat.find(target)
    if not target or pos < 0:
        return None
    start, end = keep[pos][1], keep[pos + len(target) - 1][1] + 1
    return text[start:end]


def page_range(chunk: dict) -> tuple[int, int]:
    start = int(chunk["page"])
    end = chunk.get("metadata", {}).get("page_end") or start
    return start, int(end)


def gold_hit(chunk: dict, g: dict) -> bool:
    if chunk.get("source_id") != g["source_id"]:
        return False
    lo, hi = page_range(chunk)
    if not lo <= int(g["page"]) <= hi:
        return False
    body = squash(chunk["text"])
    quotes = g.get("evidence_all") or [g["evidence"]]
    return all(squash(q) in body for q in quotes)


def hit(chunk: dict, gold: list[dict]) -> bool:
    return any(gold_hit(chunk, g) for g in gold)
