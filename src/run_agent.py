"""Chạy agent thật trên benchmark và lưu trace từng câu.

Ví dụ:
  python -m src.run_agent --split dev --config dense_k3_s3 --retriever dense --ids q01 u01
  python -m src.run_agent --split dev --config dense_k3_s3 --retriever dense          # cả split, tự bỏ qua câu đã xong

- Chỉ đưa vào agent: câu hỏi + `filter` (thông tin người dùng cung cấp trong benchmark). Không đưa gold/đáp án.
- Lưu sau MỖI câu vào results/agent/<split>/<config>/traces.json, nên hết hạn mức giữa chừng vẫn giữ được kết quả;
  chạy lại sẽ bỏ qua câu đã có status answered/abstain (câu lỗi sẽ được chạy lại). Dùng --rerun để chạy lại tất cả.
- In thêm bảng tóm tắt tự động (KHÔNG phải điểm chấm: chấm đúng/sai phải đọc tay ở bước review).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from src.agent import run_agent
from src.common import DATA, RESULTS, hit, load_json, save_json
from src.llm import OpenRouterLLM


def make_retriever(name: str):
    from src.retrievers import BM25Retriever, DenseRetriever, HybridRRF, RerankRetriever
    if name == "dense":
        return DenseRetriever()
    if name == "bm25":
        return BM25Retriever(tokenizer="bigram")
    if name == "hybrid":
        return HybridRRF([DenseRetriever(), BM25Retriever(tokenizer="bigram")])
    if name == "rerank_dense":
        return RerankRetriever(DenseRetriever())
    if name == "rerank_hybrid":
        return RerankRetriever(HybridRRF([DenseRetriever(), BM25Retriever(tokenizer="bigram")]))
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--config", required=True, help="tên cấu hình (thư mục kết quả)")
    ap.add_argument("--retriever", default="dense", choices=("dense", "bm25", "hybrid", "rerank_dense", "rerank_hybrid"))
    ap.add_argument("--max-searches", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=6000, help="tổng context budget (ký tự)")
    ap.add_argument("--prompt", default="base", choices=("base", "strict_refs"), help="phiên bản system prompt")
    ap.add_argument("--model", default=None, help="ghi đè LLM_MODEL trong .env")
    ap.add_argument("--ids", nargs="*", help="chỉ chạy các ID này")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--sleep", type=float, default=3.0, help="nghỉ giữa các câu (giây) để tránh rate limit")
    args = ap.parse_args()

    questions = [q for q in load_json(DATA / "questions.json") if q["split"] == args.split]
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    out_dir = RESULTS / "agent" / args.split / args.config
    path = out_dir / "traces.json"
    traces = {t["query_id"]: t for t in (load_json(path) if path.exists() else [])}

    llm = OpenRouterLLM(model=args.model)
    retriever = make_retriever(args.retriever)
    retriever.search("khởi động", top_k=1)
    print(f"model={llm.model} retriever={retriever.name} max_searches={args.max_searches} top_k={args.top_k} "
          f"budget={args.budget} prompt={args.prompt} | {len(questions)} câu")

    for i, q in enumerate(questions):
        old = traces.get(q["id"])
        if old and old["status"] != "error" and not args.rerun:
            print(f"  {q['id']}: bỏ qua (đã có {old['status']})")
            continue
        user_info = q.get("filter") or None
        try:
            r = run_agent(q["question"], retriever, llm, user_info=user_info, max_searches=args.max_searches,
                          top_k=args.top_k, budget_chars=args.budget, prompt_version=args.prompt)
        except Exception as e:  # lỗi ngoài dự kiến: ghi lại, không bịa kết quả
            r = {"status": "error", "answer": "", "citations": [], "events": [], "reason": "exception",
                 "error": {"type": type(e).__name__, "message": str(e)}}
        # Thông tin tự động để hỗ trợ người chấm (không đưa vào agent): context có chứa chunk gold không?
        ctx_ids = set(r.get("context_ids", []))
        auto = {}
        if q.get("answerable", True) and r.get("status") != "error":
            chunks = {e_id: None for e_id in ctx_ids}
            ctx_chunks = [c for c in load_json(DATA / "chunks.json") if c["id"] in chunks] if ctx_ids else []
            auto = {"context_has_gold": any(hit(c, q["gold"]) for c in ctx_chunks),
                    "cited_gold": any(hit(c, q["gold"]) for c in ctx_chunks if c["id"] in r.get("citations", []))}
        traces[q["id"]] = {"query_id": q["id"], "split": q["split"], "answerable": q.get("answerable", True),
                           "question": q["question"], "config": args.config,
                           "run_at": datetime.now().isoformat(timespec="seconds"), **r, "auto_check": auto}
        save_json(path, list(traces.values()))
        ans = (r.get("answer") or "").replace("\n", " ")
        print(f"  {q['id']} [{'có đáp án' if q.get('answerable', True) else 'thiếu evidence'}] "
              f"status={r['status']} searches={r.get('n_searches')} llm={r.get('n_llm_calls')} "
              f"{r.get('elapsed_s')}s auto={auto} | {ans[:160]}")
        if r["status"] == "error":
            print(f"      lỗi: {r.get('error')}")
        if i < len(questions) - 1:
            time.sleep(args.sleep)

    done = [t for t in traces.values() if t["query_id"] in {q["id"] for q in questions}]
    save_json(out_dir / "run_summary.json", {
        "config": args.config, "split": args.split, "model": llm.model, "retriever": retriever.name,
        "max_searches": args.max_searches, "top_k": args.top_k, "budget_chars": args.budget, "prompt": args.prompt,
        "n": len(done), "status": {s: sum(t["status"] == s for t in done) for s in ("answered", "abstain", "error")},
        "llm_calls": sum(t.get("n_llm_calls") or 0 for t in done),
        "cost_usd": round(sum((t.get("usage") or {}).get("cost_usd") or 0 for t in done), 6),
        "note": "Tóm tắt tự động; điểm câu trả lời/citation/abstention phải chấm tay."})


if __name__ == "__main__":
    main()
