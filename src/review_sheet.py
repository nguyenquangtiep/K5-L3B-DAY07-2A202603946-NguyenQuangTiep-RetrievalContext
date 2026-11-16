"""In bảng hỗ trợ chấm tay cho một lần chạy agent: câu hỏi, đáp án chuẩn, evidence gold, câu trả lời agent,
và NGUYÊN VĂN các chunk được trích dẫn để người chấm đối chiếu. Không tự chấm điểm.

  python -m src.review_sheet --run results/agent/dev/A0_dense_s3 > review.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common import DATA, hit, load_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--max-chars", type=int, default=1500)
    args = ap.parse_args()
    qs = {q["id"]: q for q in load_json(DATA / "questions.json")}
    chunks = {c["id"]: c for c in load_json(DATA / "chunks.json")}
    for t in load_json(args.run / "traces.json"):
        q = qs[t["query_id"]]
        print(f"\n## {q['id']} — {'có đáp án' if q.get('answerable', True) else 'THIẾU BẰNG CHỨNG'}  (status={t['status']}, reason={t.get('reason')})")
        print(f"**Câu hỏi:** {q['question']}")
        if q.get("filter"):
            print(f"**Thông tin người dùng:** {q['filter']}")
        if q.get("answerable", True):
            print(f"**Đáp án chuẩn:** {q['expected_answer']}")
            for g in q["gold"]:
                print(f"**Evidence gold:** {g['source_id']} trang {g['page']} {g.get('article') or ''}: “{' … '.join(g['evidence_all'])}”")
        else:
            print(f"**Lý do thiếu bằng chứng:** {q['absence_reason']}")
        searches = [e for e in t.get("events", []) if e.get("event") == "search"]
        for e in searches:
            print(f"- search: `{e['query']}` filter={e['filter']} → " +
                  ", ".join(f"{r['article']} {r['source_id']} p{r['page']}" for r in e["results"]))
        print(f"\n**Agent trả lời:**\n> {(t.get('answer') or '').replace(chr(10), chr(10) + '> ')}")
        for cid in t.get("citations", []):
            c = chunks[cid]
            tag = "HIT gold" if q.get("answerable", True) and hit(c, q["gold"]) else ""
            print(f"\n<details><summary>[{cid}] {c['source']} trang {c['page']}-{c['metadata'].get('page_end')} {tag}</summary>\n")
            print(c["text"][:args.max_chars] + ("…" if len(c["text"]) > args.max_chars else ""))
            print("\n</details>")


if __name__ == "__main__":
    main()
