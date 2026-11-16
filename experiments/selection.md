# Chốt cấu hình (trước khi chạy test)

Ngày 2026-11-16. Học viên xác nhận nhãn chấm tay dev của A0, A1, A2 và chốt **A1_dense_k2** theo tiêu chí trong [selection_criteria.md](selection_criteria.md) (ghi trước khi chạy agent trên dev). Chưa chạy bất kỳ cấu hình nào trên test tại thời điểm này.

## Cấu hình đã chốt — A1_dense_k2
- Model agent: `gemini-3.1-flash-lite` (Google AI Studio, free tier), temperature 0.
- Retriever: dense — `BAAI/bge-m3` + PostgreSQL/pgvector Day 6 (strategy B, 356 chunk), cosine, exact search.
- Tool `search_law(query, year?)`; `max_searches=3`, `top_k=2`, context budget 4.000 ký tự; prompt `base`.
- Thông tin người dùng (`filter` năm luật) được đưa vào câu hỏi; không đưa gold.

## Kết quả dev (15 câu có đáp án + 2 câu thiếu bằng chứng)

| Cấu hình | Grounded 0/1/2 (TB) | Citation | Từ chối đúng | Từ chối nhầm | Context evidence | LLM/câu | Token vào TB | Trung vị |
|---|---|---|---|---|---|---|---|---|
| A0 baseline (top-3, 6.000) | 0/1/14 (1,93) | 20/20 | 2/2 | 0/15 | 15/15 | 2,0 | 2.212 | 4,08 s |
| A1 (top-2, 4.000) | 0/1/14 (1,93) | 19/19 | 2/2 | 0/15 | 15/15 | 2,0 | 1.899 | 3,54 s |
| A2 (+ quy tắc 7) | 0/0/15 (2,00) | 21/21 | 2/2 | 0/15 | 15/15 | 2,0 | 2.358 | 4,91 s |

## Áp tiêu chí
1. Grounded answer: chênh tối đa 1 câu (0,067 ≤ 0,07) → hòa cả ba.
2. Từ chối đúng / nhầm: 2/2, 0/15 cả ba → hòa.
3. Context Evidence Rate: 15/15 cả ba → hòa.
4. LLM/câu bằng nhau (2,0); thời gian trung vị thấp nhất: **A1** → chọn A1.

## Ghi chú trung thực
- A2 là cấu hình duy nhất sửa lỗi grounding q07 (không tự diễn giải điều luật được dẫn chiếu) nhưng hơn baseline đúng 1 câu, nằm trong ngưỡng hòa đã công bố trước nên không được chọn. Ngưỡng 1 câu trên 15 câu dev là thô; đây là giới hạn của tiêu chí, ghi trong báo cáo.
- Thời gian phản hồi Gemini dao động lớn (2,5–48 s/câu), nên chênh lệch trung vị giữa các cấu hình có thể một phần do nhiễu.

## Kế hoạch test (chạy một lần)
- Retrieval: `eval_retrieval --split test` cho dense với cấu hình của A0 (top-3, 6.000) và A1 (top-2, 4.000).
- Agent: A1 (đã chốt) và A0 (baseline) trên toàn bộ 7 câu test (5 có đáp án + 2 thiếu bằng chứng), cùng model.
- Chấm tay toàn bộ 7 câu cho cả hai cấu hình; chấm context 0/1/2 cho 5 câu có đáp án.
