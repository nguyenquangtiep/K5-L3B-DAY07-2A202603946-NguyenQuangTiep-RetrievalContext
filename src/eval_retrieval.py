"""Đánh giá retrieval (không dùng LLM) trên một split.

Metric (mẫu số = số câu CÓ đáp án trong split; câu thiếu evidence không vào mẫu số):
  Hit@1                  chunk hạng 1 hit gold
  Evidence Hit@3         có chunk hit gold trong top-3
  MRR@10                 1/rank của chunk hit đầu tiên trong top-10
  Context Evidence Rate  context cuối (pack_context từ top-k, budget ký tự) chứa chunk hit gold
Hit theo src.common.hit (đúng nguồn, trang gold trong [page, page_end], chứa đủ mọi đoạn evidence).
Query = câu hỏi gốc; filter = trường `filter` do người dùng cung cấp trong benchmark (không lấy từ gold).
Câu thiếu evidence vẫn được chạy và lưu trace (để xem retriever trả gì), nhưng không tính metric.

Chạy:  python -m src.eval_retrieval --split dev --configs dense bm25_syllable bm25_bigram hybrid
"""
from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime

from src.common import DATA, RESULTS, hit, load_json, save_json
from src.retrievers import BM25Retriever, DenseRetriever, HybridRRF, RerankRetriever, pack_context

EVAL_K = 10


def build(names: list[str]) -> dict:
    cache: dict = {}

    def dense():
        if "dense" not in cache:
            cache["dense"] = DenseRetriever()
        return cache["dense"]

    def bm25(tok):
        key = f"bm25_{tok}"
        if key not in cache:
            cache[key] = BM25Retriever(tokenizer=tok)
        return cache[key]

    factories = {
        "dense": dense,
        "bm25_syllable": lambda: bm25("syllable"),
        "bm25_bigram": lambda: bm25("bigram"),
        "hybrid": lambda: HybridRRF([dense(), bm25("bigram")]),
        "hybrid_syllable": lambda: HybridRRF([dense(), bm25("syllable")]),
        "rerank_hybrid": lambda: RerankRetriever(HybridRRF([dense(), bm25("bigram")]), candidates=20),
        "rerank_dense": lambda: RerankRetriever(dense(), candidates=20),
    }
    out = {}
    for n in names:
        t = time.perf_counter()
        out[n] = factories[n]()
        print(f"[init] {n}: {time.perf_counter() - t:.1f}s")
    return out


def evaluate(retriever, questions, *, top_k: int, budget: int, use_filter: bool) -> tuple[list, dict]:
    traces = []
    for q in questions:
        filters = q.get("filter") if use_filter else None
        t = time.perf_counter()
        error = None
        try:
            results = retriever.search(q["question"], filters=filters, top_k=max(EVAL_K, top_k))
        except Exception as e:  # lỗi vector store/model: ghi lại, không thay bằng kết quả giả
            results, error = [], f"{type(e).__name__}: {e}"
        ms = (time.perf_counter() - t) * 1000
        context, ctx_ids = pack_context(results[:top_k], budget)
        answerable = q.get("answerable", True)
        gold = q.get("gold", [])
        flags = [hit(c, gold) for c in results] if answerable else [False] * len(results)
        first = next((i for i, f in enumerate(flags, 1) if f), None)
        traces.append({
            "query_id": q["id"], "split": q["split"], "answerable": answerable, "question": q["question"],
            "filters": filters, "method": retriever.name, "top_k": top_k, "budget_chars": budget,
            "results": [{"rank": c["rank"], "id": c["id"], "score": round(c["score"], 6), "source_id": c["source_id"],
                         "page": c["page"], "page_end": c["metadata"].get("page_end"),
                         "article": c["metadata"].get("article"), "hit": f,
                         **({"component_ranks": c["component_ranks"]} if "component_ranks" in c else {}),
                         **({"base_rank": c["base_rank"]} if "base_rank" in c else {}),
                         "preview": c["text"][:160]} for c, f in zip(results, flags)],
            "context_ids": ctx_ids, "context_chars": len(context), "elapsed_ms": round(ms, 1), "error": error,
            **({"hit1": bool(flags[:1] and flags[0]), "hit3": any(flags[:3]),
                "first_hit_rank": first,
                "context_hit": any(f and c["id"] in ctx_ids for c, f in zip(results, flags))} if answerable else {}),
        })
    ans = [t for t in traces if t["answerable"]]
    n = len(ans)
    summary = {
        "method": retriever.name, "split": questions[0]["split"] if questions else None, "top_k": top_k,
        "budget_chars": budget, "use_filter": use_filter, "n_answerable": n,
        "n_unanswerable_run": len(traces) - n, "errors": sum(bool(t["error"]) for t in traces),
        "hit1": sum(t["hit1"] for t in ans), "hit3": sum(t["hit3"] for t in ans),
        "context_hit": sum(t["context_hit"] for t in ans),
        "mrr10": round(sum(1 / t["first_hit_rank"] for t in ans if t["first_hit_rank"]) / n, 4) if n else None,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in ans), 1) if n else None,
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
        "cost_usd": 0.0,
        "misses_hit3": [t["query_id"] for t in ans if not t["hit3"]],
        "misses_hit1": [t["query_id"] for t in ans if not t["hit1"]],
    }
    return traces, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--configs", nargs="+", default=["dense", "bm25_syllable", "bm25_bigram", "hybrid"])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=3000, help="context budget (ký tự)")
    ap.add_argument("--no-filter", action="store_true", help="bỏ filter người dùng cung cấp")
    ap.add_argument("--tag", default="", help="hậu tố tên thư mục kết quả")
    args = ap.parse_args()

    questions = [q for q in load_json(DATA / "questions.json") if q["split"] == args.split]
    retrievers = build(args.configs)
    rows = []
    for name, r in retrievers.items():
        r.search("khởi động", top_k=1)  # warm-up để thời gian đo không gồm lần chạy đầu
        traces, summary = evaluate(r, questions, top_k=args.top_k, budget=args.budget, use_filter=not args.no_filter)
        summary.update({"config": name, "run_at": datetime.now().isoformat(timespec="seconds")})
        out = RESULTS / "retrieval" / args.split / f"{name}{args.tag}"
        save_json(out / "traces.json", traces)
        save_json(out / "summary.json", summary)
        rows.append(summary)

    n = rows[0]["n_answerable"]
    print(f"\nsplit={args.split} | top_k={args.top_k} | budget={args.budget} ký tự | filter={'tắt' if args.no_filter else 'bật'}"
          f" | N={n} câu có đáp án")
    print(f"{'config':18} {'Hit@1':>10} {'Hit@3':>10} {'Context':>10} {'MRR@10':>7} {'ms':>7}  trượt Hit@3")
    for s in rows:
        f = lambda k: f"{s[k]}/{n} {s[k] / n:.0%}"
        print(f"{s['config']:18} {f('hit1'):>10} {f('hit3'):>10} {f('context_hit'):>10} {s['mrr10']:>7} "
              f"{s['median_ms']:>7}  {s['misses_hit3']}")
    save_json(RESULTS / "retrieval" / args.split / f"comparison{args.tag}.json", rows)


if __name__ == "__main__":
    main()
