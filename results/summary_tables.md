# Bảng tổng hợp kết quả (sinh tự động bởi `python -m src.summarize`)

Grounded/citation/abstention lấy từ nhãn chấm tay (`reviews.json`) qua `starter/agent_metrics.py`. 'Context có gold (tự động)' chỉ là kiểm tra khớp chuỗi, không thay chấm tay.

### Retrieval — dev

| Cấu hình | top-k | Budget (ký tự) | Filter | Hit@1 | Evidence Hit@3 | Context Evidence | MRR@10 | Trung vị ms/query | Context TB (ký tự) | Chi phí |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25_bigram | 3 | 3000 | bật | 12/15 (80%) | 14/15 (93%) | 14/15 (93%) | 0.8667 | 3.0 | 2461.5 | 0.0 USD |
| bm25_syllable | 3 | 3000 | bật | 10/15 (67%) | 14/15 (93%) | 14/15 (93%) | 0.8 | 3.2 | 2334.5 | 0.0 USD |
| dense | 3 | 3000 | bật | 14/15 (93%) | 15/15 (100%) | 15/15 (100%) | 0.9667 | 37.8 | 2279.1 | 0.0 USD |
| hybrid | 3 | 3000 | bật | 12/15 (80%) | 14/15 (93%) | 14/15 (93%) | 0.8741 | 30.3 | 2344.3 | 0.0 USD |
| hybrid_syllable | 3 | 3000 | bật | 12/15 (80%) | 14/15 (93%) | 14/15 (93%) | 0.875 | 32.7 | 2508.5 | 0.0 USD |
| rerank_dense | 3 | 3000 | bật | 13/15 (87%) | 15/15 (100%) | 15/15 (100%) | 0.9333 | 488.0 | 2440.5 | 0.0 USD |
| rerank_hybrid | 3 | 3000 | bật | 13/15 (87%) | 15/15 (100%) | 15/15 (100%) | 0.9333 | 490.5 | 2440.5 | 0.0 USD |

### Retrieval — test

| Cấu hình | top-k | Budget (ký tự) | Filter | Hit@1 | Evidence Hit@3 | Context Evidence | MRR@10 | Trung vị ms/query | Context TB (ký tự) | Chi phí |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25_bigram_k3 | 3 | 6000 | bật | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 1.0 | 3.0 | 4006.2 | 0.0 USD |
| dense_k2 | 2 | 4000 | bật | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 1.0 | 24.3 | 2702 | 0.0 USD |
| dense_k3 | 3 | 6000 | bật | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 1.0 | 27.8 | 3860.4 | 0.0 USD |
| hybrid_k3 | 3 | 6000 | bật | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 1.0 | 24.9 | 4006.2 | 0.0 USD |

### Agent — dev

| Cấu hình | Model | Tool use | Số lượt tìm (phân bố) | LLM call/câu | Grounded 0/1/2 (TB/2) | Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Context có gold (tự động) | Token vào/ra TB | Trung vị s/câu | Lỗi | Chi phí |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_dense_s3 | gemini-3.1-flash-lite | 17/17 (100%) | {1: 17} | 2.0 | 0/1/14 (1.93) | 20/20 (100%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0%) | 15/15 (100%) | 2212/132 | 4.08 | 0 | 0.0000 USD |
| A1_dense_k2 | gemini-3.1-flash-lite | 17/17 (100%) | {1: 17} | 2.0 | 0/1/14 (1.93) | 19/19 (100%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0%) | 15/15 (100%) | 1899/132 | 3.54 | 0 | 0.0000 USD |
| A2_dense_strict | gemini-3.1-flash-lite | 17/17 (100%) | {1: 17} | 2.0 | 0/0/15 (2.00) | 21/21 (100%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0%) | 15/15 (100%) | 2358/135 | 4.91 | 0 | 0.0000 USD |
| smoke_dense | nvidia/nemotron-3.5-super-vl:free | 2/2 (100%) | {1: 1, 2: 1} | 2.5 | chưa chấm | chưa chấm | chưa chấm | chưa chấm | chưa chấm | 1/1 (100%) | 3788/662 | 7.02 | 0 | 0.0000 USD |

### Agent — test

| Cấu hình | Model | Tool use | Số lượt tìm (phân bố) | LLM call/câu | Grounded 0/1/2 (TB/2) | Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Context có gold (tự động) | Token vào/ra TB | Trung vị s/câu | Lỗi | Chi phí |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_dense_s3 | gemini-3.1-flash-lite | 7/7 (100%) | {1: 7} | 2.0 | 0/0/5 (2.00) | 11/11 (100%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0%) | 5/5 (100%) | 2371/173 | 4.33 | 0 | 0.0000 USD |
| A1_dense_k2 | gemini-3.1-flash-lite | 7/7 (100%) | {1: 7} | 2.0 | 0/0/5 (2.00) | 8/8 (100%) | 5/5 (100%) | 2/2 (100%) | 0/5 (0%) | 5/5 (100%) | 2027/184 | 4.69 | 0 | 0.0000 USD |
