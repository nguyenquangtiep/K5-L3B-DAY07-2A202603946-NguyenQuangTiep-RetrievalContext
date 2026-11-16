"""Gom mọi kết quả đã lưu thành bảng Markdown: results/summary_tables.md.

- Retrieval: đọc results/retrieval/<split>/<config>/summary.json
- Agent: đọc results/agent/<split>/<config>/{traces.json, agent_summary.json (nhãn chấm tay đã tổng hợp)}
Chỉ tổng hợp số đã có; không chạy model, không tự chấm.

  python -m src.summarize
"""
from __future__ import annotations

import statistics as st

from src.common import RESULTS, load_json


def frac(n, d):
    return f"{n}/{d} ({n / d:.0%})" if d else "0/0 (không áp dụng)"


def retrieval_tables() -> list[str]:
    out = []
    for split in ("dev", "test"):
        rows = sorted((RESULTS / "retrieval" / split).glob("*/summary.json"))
        if not rows:
            continue
        out += [f"\n### Retrieval — {split}\n",
                "| Cấu hình | top-k | Budget (ký tự) | Filter | Hit@1 | Evidence Hit@3 | Context Evidence | MRR@10 | Trung vị ms/query | Context TB (ký tự) | Chi phí |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in rows:
            s = load_json(p)
            n = s["n_answerable"]
            out.append(f"| {p.parent.name} | {s['top_k']} | {s['budget_chars']} | {'bật' if s['use_filter'] else 'tắt'} | "
                       f"{frac(s['hit1'], n)} | {frac(s['hit3'], n)} | {frac(s['context_hit'], n)} | {s['mrr10']} | "
                       f"{s['median_ms']} | {s['avg_context_chars']} | {s['cost_usd']} USD |")
    return out


def agent_tables() -> list[str]:
    out = []
    for split in ("dev", "test"):
        dirs = sorted(d for d in (RESULTS / "agent" / split).glob("*") if (d / "traces.json").exists())
        if not dirs:
            continue
        out += [f"\n### Agent — {split}\n",
                "| Cấu hình | Model | Tool use | Số lượt tìm (phân bố) | LLM call/câu | Grounded 0/1/2 (TB/2) | Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Context có gold (tự động) | Token vào/ra TB | Trung vị s/câu | Lỗi | Chi phí |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in dirs:
            tr = load_json(d / "traces.json")
            ok = [t for t in tr if t["status"] != "error"]
            models = ", ".join(sorted({t.get("model") or "?" for t in tr}))
            dist = {}
            for t in ok:
                dist[t.get("n_searches")] = dist.get(t.get("n_searches"), 0) + 1
            ans = [t for t in ok if t["answerable"]]
            ctx_gold = sum(t.get("auto_check", {}).get("context_has_gold", False) for t in ans)
            tok_in = round(st.mean(t["usage"]["prompt_tokens"] for t in ok)) if ok else 0
            tok_out = round(st.mean(t["usage"]["completion_tokens"] for t in ok)) if ok else 0
            med = st.median(t["elapsed_s"] for t in ok) if ok else 0
            calls = round(sum(t.get("n_llm_calls") or 0 for t in ok) / len(ok), 2) if ok else 0
            cost = sum((t.get("usage") or {}).get("cost_usd") or 0 for t in tr)
            summ = d / "agent_summary.json"
            if summ.exists():
                a = load_json(summ)
                g = a["grounded_answer"]
                grade = f"{g['0']}/{g['1']}/{g['2']} ({g['mean']:.2f})" if g["N"] else "-"
                cit = frac(a["citation_validity"]["n"], a["citation_validity"]["N"])
                cov = frac(a["answer_citation_coverage"]["n"], a["answer_citation_coverage"]["N"])
                abst = frac(a["abstention"]["n"], a["abstention"]["N"])
                fabs = frac(a["false_abstention"]["n"], a["false_abstention"]["N"])
                tool = frac(a["tool_use"]["n"], a["tool_use"]["N"])
            else:
                grade = cit = cov = abst = fabs = "chưa chấm"
                tool = frac(sum(any(e.get("event") == "search" for e in t["events"]) for t in ok), len(ok))
            out.append(f"| {d.name} | {models} | {tool} | {dist} | {calls} | {grade} | {cit} | {cov} | {abst} | {fabs} | "
                       f"{frac(ctx_gold, len(ans))} | {tok_in}/{tok_out} | {med} | {len(tr) - len(ok)} | {cost:.4f} USD |")
    return out


def main() -> None:
    lines = ["# Bảng tổng hợp kết quả (sinh tự động bởi `python -m src.summarize`)",
             "", "Grounded/citation/abstention lấy từ nhãn chấm tay (`reviews.json`) qua `starter/agent_metrics.py`. "
             "'Context có gold (tự động)' chỉ là kiểm tra khớp chuỗi, không thay chấm tay."]
    lines += retrieval_tables() + agent_tables()
    path = RESULTS / "summary_tables.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
