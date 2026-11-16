"""Các phương pháp retrieval của Day 7. Mọi retriever có cùng giao diện:

    search(query: str, *, filters: dict | None, top_k: int) -> list[dict]

Mỗi kết quả là chunk (id, text, source_id, source, page, metadata) + score, rank, method.
`filters` chỉ nhận thông tin người dùng cung cấp hoặc agent suy ra từ câu hỏi (vd {"year": 2025});
KHÔNG bao giờ lấy từ gold. Hiện hỗ trợ: year, source_id.

1. DenseRetriever   — bge-m3 (giống Day 6) + PostgreSQL/pgvector Day 6, cosine, exact search.
2. BM25Retriever    — Okapi BM25 tự cài trên data/chunks.json; tách từ theo âm tiết hoặc âm tiết + bigram.
3. HybridRRF        — hợp nhất ranking của dense và BM25 bằng Reciprocal Rank Fusion (không cộng score thô).
4. RerankRetriever  — lấy nhiều candidate từ một retriever rồi xếp lại bằng cross-encoder bge-reranker-v2-m3.
"""
from __future__ import annotations

import math
import os
import re
import time
import unicodedata
from collections import Counter
from typing import Any, Protocol

from src.common import DATA, ROOT, load_json

ALLOWED_FILTERS = {"year": int, "source_id": str}


class Retriever(Protocol):
    name: str

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]: ...


def check_filters(filters: dict | None) -> dict:
    filters = dict(filters or {})
    for k in filters:
        if k not in ALLOWED_FILTERS:
            raise ValueError(f"filter '{k}' không được hỗ trợ; chỉ nhận {sorted(ALLOWED_FILTERS)}")
        filters[k] = ALLOWED_FILTERS[k](filters[k])
    return filters


def match_filters(chunk: dict, filters: dict) -> bool:
    for k, v in filters.items():
        actual = chunk["source_id"] if k == "source_id" else chunk["metadata"].get(k)
        if actual != v:
            return False
    return True


def load_env() -> dict[str, str]:
    from dotenv import dotenv_values
    cfg = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None}
    return {**cfg, **{k: os.environ[k] for k in cfg if k in os.environ}}


# --------------------------------------------------------------------------- dense
class DenseRetriever:
    """Embedding query bằng đúng model đã index ở Day 6, truy vấn pgvector (strategy B)."""

    def __init__(self, database_url: str | None = None, model_name: str | None = None, strategy: str | None = None):
        import psycopg
        import torch
        from sentence_transformers import SentenceTransformer

        env = load_env()
        self.model_name = model_name or env.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.strategy = strategy or env.get("VECTOR_STRATEGY", "B")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if self.device == "cuda" else {}
        self.model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.conn = psycopg.connect(database_url or env["DATABASE_URL"])
        with self.conn.cursor() as cur:  # kiểm tra index tương thích với model query
            cur.execute("SELECT count(*), min(embedding_model), max(embedding_model), max(vector_dims(embedding)) "
                        "FROM chunks WHERE strategy = %s", (self.strategy,))
            n, m1, m2, dims = cur.fetchone()
        self.conn.rollback()
        if not n or m1 != self.model_name or m2 != self.model_name or dims != self.dim:
            raise RuntimeError(f"Index không tương thích: n={n}, model={m1}/{m2}, dims={dims}; "
                               f"query model={self.model_name} dims={self.dim}")
        self.n_chunks = n
        self.chunks = {c["id"]: c for c in load_json(DATA / "chunks.json")}
        self.name = "dense"

    def embed(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32").tolist()

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        filters = check_filters(filters)
        vec = "[" + ",".join(f"{x:.7f}" for x in self.embed(query)) + "]"
        where, params = ["strategy = %s", "embedding_model = %s"], [self.strategy, self.model_name]
        if "year" in filters:
            where.append("year = %s"); params.append(filters["year"])
        if "source_id" in filters:
            where.append("document_id = %s"); params.append(filters["source_id"])
        sql = (f"SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score FROM chunks "
               f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        with self.conn.cursor() as cur:
            cur.execute(sql, [vec, *params, vec, top_k])
            rows = cur.fetchall()
        self.conn.rollback()
        out = []
        for rank, (cid, score) in enumerate(rows, 1):
            if cid not in self.chunks:
                raise RuntimeError(f"chunk {cid} có trong pgvector nhưng không có trong data/chunks.json")
            out.append({**self.chunks[cid], "score": float(score), "rank": rank, "method": self.name})
        return out


# --------------------------------------------------------------------------- BM25
_TOKEN = re.compile(r"\w+")


def tokenize(text: str, mode: str = "syllable") -> list[str]:
    """syllable: mỗi âm tiết là 1 token. bigram: âm tiết + cặp âm tiết liền kề (xấp xỉ từ ghép tiếng Việt,
    vd 'thử việc' -> 'thử', 'việc', 'thử_việc'), không cần thư viện tách từ."""
    syl = _TOKEN.findall(unicodedata.normalize("NFC", text).casefold())
    if mode == "syllable":
        return syl
    if mode == "bigram":
        return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]
    raise ValueError(mode)


class BM25Retriever:
    def __init__(self, chunks: list[dict] | None = None, *, tokenizer: str = "bigram", k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks if chunks is not None else load_json(DATA / "chunks.json")
        self.tokenizer, self.k1, self.b = tokenizer, k1, b
        self.tf = [Counter(tokenize(c["text"], tokenizer)) for c in self.chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.len) / len(self.len)
        df = Counter(term for t in self.tf for term in t)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.name = f"bm25_{tokenizer}"

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        filters = check_filters(filters)
        q = Counter(tokenize(query, self.tokenizer))
        scored = []
        for i, c in enumerate(self.chunks):
            if filters and not match_filters(c, filters):
                continue
            tf, norm = self.tf[i], self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
            if s > 0:
                scored.append((s, c["id"], c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{**c, "score": s, "rank": r, "method": self.name} for r, (s, _, c) in enumerate(scored[:top_k], 1)]


# --------------------------------------------------------------------------- hybrid
class HybridRRF:
    """score_RRF(d) = Σ_r 1 / (k + rank_r(d)). Dùng thứ hạng nên không cần chuẩn hóa thang điểm cosine/BM25."""

    def __init__(self, retrievers: list[Retriever], *, k: int = 60, candidates: int = 20):
        self.retrievers, self.k, self.candidates = retrievers, k, candidates
        self.name = "hybrid_rrf(" + "+".join(r.name for r in retrievers) + ")"

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        fused: dict[str, dict] = {}
        for r in self.retrievers:
            for item in r.search(query, filters=filters, top_k=self.candidates):
                f = fused.setdefault(item["id"], {**item, "score": 0.0, "component_ranks": {}})
                f["score"] += 1.0 / (self.k + item["rank"])
                f["component_ranks"][r.name] = item["rank"]
        ranked = sorted(fused.values(), key=lambda x: (-x["score"], x["id"]))[:top_k]
        return [{**x, "rank": i, "method": self.name} for i, x in enumerate(ranked, 1)]


# --------------------------------------------------------------------------- rerank
class RerankRetriever:
    """Lấy `candidates` kết quả từ retriever gốc rồi xếp lại bằng cross-encoder (đọc cặp query–chunk)."""

    def __init__(self, base: Retriever, *, model_name: str = "BAAI/bge-reranker-v2-m3", candidates: int = 20):
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if device == "cuda" else {}
        self.model = CrossEncoder(model_name, device=device, max_length=1024, **kwargs)
        self.base, self.candidates, self.model_name = base, candidates, model_name
        self.name = f"rerank({base.name})"

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        cands = self.base.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], batch_size=8, show_progress_bar=False)
        order = sorted(zip(scores, cands), key=lambda x: (-float(x[0]), x[1]["id"]))[:top_k]
        return [{**c, "base_rank": c["rank"], "score": float(s), "rank": i, "method": self.name}
                for i, (s, c) in enumerate(order, 1)]


# --------------------------------------------------------------------------- context
def pack_context(results: list[dict], budget_chars: int = 3000) -> tuple[str, list[str]]:
    """Chọn context: bỏ trùng ID, giữ thứ tự rank, mỗi đoạn kèm nhãn nguồn/trang; đoạn không vừa budget thì bỏ qua
    và thử đoạn tiếp theo (không cắt giữa chừng để không mất điều kiện của điều luật)."""
    blocks, ids, seen = [], [], set()
    for c in results:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        lo, hi = c["page"], c["metadata"].get("page_end") or c["page"]
        pages = f"trang {lo}" if lo == hi else f"trang {lo}-{hi}"
        block = f"[{c['id']}] {c['source']} — {pages}\n{c['text']}"
        if len("\n\n".join(blocks + [block])) <= budget_chars:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids
