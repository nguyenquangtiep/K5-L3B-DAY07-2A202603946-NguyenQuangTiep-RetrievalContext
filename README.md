# Day 7 — Agentic RAG

## Bài làm Day 7

> **Nguyễn Quang Tiệp · 2A202603946 · L3B** — Báo cáo: [report/individual/2A202603946_NguyenQuangTiep.md](report/individual/2A202603946_NguyenQuangTiep.md) · Bảng số liệu: [results/summary_tables.md](results/summary_tables.md)
>
> **Corpus:** 3 văn bản luật lao động – việc làm (140 trang, public domain), 356 chunk từ Day 6. **Vector store:** PostgreSQL 16 + pgvector 0.8.7 của Day 6. **Embedding:** `BAAI/bge-m3` (1024 chiều, local). **Agent model:** `gemini-3.1-flash-lite` (Google AI Studio, free tier). **Retrieval:** dense, BM25 (âm tiết / bigram), hybrid RRF, rerank `bge-reranker-v2-m3`. **Cấu hình đã chốt:** `A1_dense_k2` (dense, top-2, budget 4.000 ký tự). **Chi phí:** 0 USD.
>
> ### Cài đặt (Windows PowerShell)
> 1. Bật vector store Day 6 (repo `K5-L3B-DAY06-2A202603946-DataIngestion`, thư mục `starter/`): `docker compose -p day6_student_starter up -d --wait`. Dữ liệu index đã có trong volume Docker; nếu chưa có, chạy `python -m src.run prepare` rồi `python -m src.run index` theo README Day 6.
> 2. Môi trường Python 3.12: `python -m venv .venv ; .venv\Scripts\Activate.ps1`, cài torch (xem [requirements.txt](requirements.txt)) rồi `python -m pip install -r requirements.txt`.
> 3. `Copy-Item .env.example .env`, điền `GEMINI_API_KEY` (tạo ở aistudio.google.com). **Không commit `.env`.**
> 4. Trên Windows đặt `$env:PYTHONUTF8="1"` trước khi chạy (một test của starter đọc file không chỉ định UTF-8).
>
> ### Chạy (từ thư mục gốc repo)
> | Lệnh | Việc làm | Đầu ra |
> |---|---|---|
> | `python -m unittest discover -s tests -v` | 22 test (15 starter + 7 vòng lặp agent với LLM giả) | — |
> | `python -m src.export_day6 --day6 "<repo Day 6>"` | Chuyển chunk + benchmark Day 6, gộp câu thiếu bằng chứng, tự kiểm tra evidence/từ khóa cấm | `data/chunks.json`, `data/questions.json` |
> | `python -m src.eval_retrieval --split dev --configs dense bm25_syllable bm25_bigram hybrid hybrid_syllable rerank_dense rerank_hybrid` | So sánh retrieval (không dùng LLM): Hit@1, Hit@3, Context Evidence, MRR, thời gian | `results/retrieval/dev/*/` |
> | `python -m src.run_agent --split dev --config A0_dense_s3 --retriever dense --top-k 3 --budget 6000` | Agent baseline (lưu sau mỗi câu, chạy lại tự bỏ qua câu đã xong) | `results/agent/dev/A0_dense_s3/traces.json` |
> | `python -m src.run_agent --split dev --config A1_dense_k2 --retriever dense --top-k 2 --budget 4000` | Cải thiện A1 (cấu hình đã chốt) | `results/agent/dev/A1_dense_k2/` |
> | `python -m src.run_agent --split dev --config A2_dense_strict --retriever dense --top-k 3 --budget 6000 --prompt strict_refs` | Cải thiện A2 (quy tắc grounding 7) | `results/agent/dev/A2_dense_strict/` |
> | `python -m src.review_sheet --run results/agent/dev/A0_dense_s3` | Bảng đối chiếu nguyên văn chunk để chấm tay | `review_sheet.md` |
> | `python starter/agent_metrics.py --cases data/questions.json --traces <run>/traces.json --reviews <run>/reviews.json --split dev --output <run>/agent_summary.json` | Tổng hợp nhãn chấm tay | `agent_summary.json` |
> | `--split test` cho các lệnh trên (chỉ sau khi chốt, xem [experiments/selection.md](experiments/selection.md)) | Đánh giá test một lần | `results/*/test/` |
> | `python -m src.summarize` | Gom mọi kết quả thành bảng | `results/summary_tables.md` |
>
> Code: [src/](src/) (`common.py` định nghĩa hit, `retrievers.py`, `agent.py`, `llm.py`, `run_agent.py`, `eval_retrieval.py`). Benchmark: [data/questions.json](data/questions.json) (24 câu), câu thiếu bằng chứng soạn tay: [data/unanswerable.json](data/unanswerable.json). Tiêu chí chọn công bố trước: [experiments/selection_criteria.md](experiments/selection_criteria.md). `results/agent/dev/smoke_dense/` là lần chạy thử code bằng model OpenRouter, không dùng để đánh giá.

---

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
