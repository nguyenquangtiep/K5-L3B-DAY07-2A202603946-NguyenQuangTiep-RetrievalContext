"""Bước 1: chuyển chunk + benchmark Day 6 sang định dạng Day 7 (data/README.md).

Đầu vào (repo Day 6):
  starter/results/chunks/B.jsonl   356 chunk theo Điều — đúng tập chunk của cấu hình C đã chọn ở Day 6,
                                   cũng là tập vector strategy='B' trong pgvector.
  benchmark/questions.json         20 câu có đáp án (15 dev / 5 test), giữ nguyên split.
  data/manifest.json               thông tin nguồn (tên luật, số hiệu, URL tải, giấy phép).

Đầu ra (repo Day 7):
  data/chunks.json      [{id, text, source_id, source, page, metadata}]
  data/questions.json   [{id, split, question, expected_answer, gold:[{source_id, page, evidence, evidence_all, article}], filter, type, answerable}]

Chỉ đổi tên trường và lấy lại evidence NGUYÊN VĂN từ chunk (bỏ qua khác biệt khoảng trắng);
không sửa câu hỏi, đáp án, split, nguồn hay trang. Gold chỉ dùng để chấm, không đưa vào agent/retrieval.

Chạy từ thư mục gốc repo Day 7:
  python -m src.export_day6 --day6 "<đường dẫn repo Day 6>"
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.common import DATA, find_exact_span, hit, save_json

DEFAULT_DAY6 = Path(__file__).resolve().parents[3] / "Lab06" / "K5-L3B-DAY06-2A202603946-DataIngestion"


def convert_chunks(day6: Path, sources: dict[str, dict]) -> list[dict]:
    out = []
    for line in (day6 / "starter/results/chunks/B.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        meta = dict(c["metadata"])
        src = sources[c["document_id"]]
        meta.update({"pdf_path_day6": c["source"], "download_url": src.get("download_url"),
                     "gazette": src.get("gazette")})
        out.append({
            "id": c["chunk_id"],                 # = chunk_id trong bảng pgvector của Day 6
            "text": c["text"],
            "source_id": c["document_id"],
            "source": f"{src['title']} {src['law_number']}",
            "page": c["page"],
            "metadata": meta,
        })
    return out


def convert_questions(day6: Path, chunks: list[dict]) -> tuple[list[dict], list[str]]:
    raw = json.loads((day6 / "benchmark/questions.json").read_text(encoding="utf-8"))["questions"]
    out, problems = [], []
    for q in raw:
        gold = []
        for g in q["gold"]:
            # Lấy lại đoạn nguyên văn trong chunk đúng nguồn chứa đoạn trích (để starter so khớp chuỗi cũng chạy được).
            exact = []
            for quote in g["evidence_quote"]:
                span = next((s for c in chunks if c["source_id"] == g["source_id"]
                             and (s := find_exact_span(c["text"], quote))), None)
                exact.append(span or quote)
                if span is None:
                    problems.append(f"{q['id']}: không tìm thấy evidence trong chunk {g['source_id']}: {quote[:60]}")
            gold.append({"source_id": g["source_id"], "page": g["page"], "article": g.get("article"),
                         "evidence": exact[0], "evidence_all": exact})
        item = {"id": q["id"], "split": q["split"], "question": q["question"],
                "expected_answer": q["expected_answer"], "gold": gold, "answerable": True,
                "expected_behavior": "search_and_answer", "type": q.get("type"),
                "filter": q.get("filter")}
        if not any(hit(c, gold) for c in chunks):
            problems.append(f"{q['id']}: không chunk nào hit gold (nguồn/trang/evidence)")
        out.append(item)
    return out, problems


def check_unanswerable(items: list[dict], chunks: list[dict]) -> list[str]:
    """Câu thiếu evidence do học viên soạn (data/unanswerable.json): kiểm lại từ khóa cấm không xuất hiện."""
    import re
    import unicodedata
    from src.common import squash
    problems = []
    for u in items:
        for key in ("id", "split", "question", "absence_reason"):
            if not u.get(key):
                problems.append(f"{u.get('id')}: thiếu {key}")
        v = u.get("verification", {})
        for kw in v.get("must_not_contain", []):
            found = [c["id"] for c in chunks if squash(kw) in squash(c["text"])]
            if found:
                problems.append(f"{u['id']}: từ khóa '{kw}' xuất hiện trong {found[:3]}")
        for pat in v.get("regex_must_not_match", []):
            found = [c["id"] for c in chunks if re.search(pat, unicodedata.normalize("NFC", c["text"]))]
            if found:
                problems.append(f"{u['id']}: regex {pat} khớp {found[:3]}")
        for rule in v.get("cooccur_must_not_match", []):
            found = [c["id"] for c in chunks if squash(rule["if_contains"]) in squash(c["text"])
                     and re.search(rule["regex"], unicodedata.normalize("NFC", c["text"]))]
            if found:
                problems.append(f"{u['id']}: chunk có '{rule['if_contains']}' khớp {rule['regex']}: {found[:3]}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day6", type=Path, default=DEFAULT_DAY6)
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()

    manifest = json.loads((args.day6 / "data/manifest.json").read_text(encoding="utf-8"))
    sources = {s["source_id"]: s for s in manifest["sources"]}

    chunks = convert_chunks(args.day6, sources)
    questions, problems = convert_questions(args.day6, chunks)
    if len({c["id"] for c in chunks}) != len(chunks):
        problems.append("chunk id bị trùng")

    unanswerable_path = args.out / "unanswerable.json"
    unanswerable = json.loads(unanswerable_path.read_text(encoding="utf-8")) if unanswerable_path.exists() else []
    problems += check_unanswerable(unanswerable, chunks)
    all_q = questions + unanswerable
    if len({q["id"] for q in all_q}) != len(all_q):
        problems.append("question id bị trùng")

    save_json(args.out / "chunks.json", chunks)
    save_json(args.out / "questions.json", all_q)

    print(f"chunks: {len(chunks)} | theo nguồn: {dict(Counter(c['source_id'] for c in chunks))}")
    table = Counter((q["split"], q.get("answerable", True)) for q in all_q)
    for split in ("dev", "test"):
        print(f"{split}: {table[(split, True)]} có đáp án + {table[(split, False)]} thiếu bằng chứng")
    print(f"tổng: {len(all_q)} câu | có filter: {sum(bool(q.get('filter')) for q in all_q)}")
    spans = sum(1 for c in chunks if c["metadata"].get("page_end", c["page"]) != c["page"])
    print(f"chunk vắt trang: {spans}")
    print("Kiểm tra:", "OK — mọi câu có đáp án có chunk hit gold; câu thiếu bằng chứng không chứa từ khóa cấm"
          if not problems else "")
    for p in problems:
        print("  LỖI:", p)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
