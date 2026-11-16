# Báo cáo Day 7 — Agentic RAG

**Học viên:** Nguyễn Quang Tiệp · **MSSV:** 2A202603946 · **Lớp:** L3B · **Ngày chạy:** 2026-11-16
**Chủ đề:** Pháp luật lao động – việc làm Việt Nam (tiếp nối Day 6) · **Cấu hình đã chốt:** `A1_dense_k2`

Bảng số liệu đầy đủ (sinh tự động từ kết quả đã lưu): [results/summary_tables.md](../../results/summary_tables.md). Tiêu chí chọn ghi trước khi chạy: [experiments/selection_criteria.md](../../experiments/selection_criteria.md); biên bản chốt cấu hình trước test: [experiments/selection.md](../../experiments/selection.md).

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Kho tài liệu giữ nguyên từ Day 6 gồm 3 văn bản trích từ PDF *Công báo Chính phủ*. Tổng cộng 140 trang nội dung, đếm theo quy tắc trong `data/manifest.json` của repo Day 6 (`K5-L3B-DAY06-2A202603946-DataIngestion`):
    - Bộ luật Lao động 45/2019/QH14 (`BLLD2019`, 93 trang)
    - Luật Việc làm 38/2013/QH13 (`LVL2013`, 20 trang)
    - Luật Việc làm 74/2025/QH15 (`LVL2025`, 27 trang)
  - **Quyền sử dụng:** văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả (Luật SHTT, Điều 15). File PDF trên Wikimedia Commons gắn nhãn Public domain. URL tải về được lưu trong `metadata.download_url` của từng chunk.
  - **Chunk:** dùng **356 chunk của cấu hình B ở Day 6** (mỗi Điều là một chunk; Điều dài hơn 2.000 ký tự được tách theo khoản). Đây là tập chunk mà cấu hình C đã chọn ở Day 6 sử dụng. Phân bố: BLLD2019 231, LVL2013 63, LVL2025 62. Có 102 chunk vắt sang trang sau, được ghi bằng `page`–`page_end`.
  - **Chuyển dữ liệu:** [src/export_day6.py](../../src/export_day6.py) chỉ đổi tên trường và lấy lại evidence nguyên văn từ chunk. Script kiểm tra **356/356 chunk có ID và text trùng khớp** với bảng `chunks` trong pgvector.

- **Vector store, embedding model, phiên bản và số chiều:**
  - **Vector store:** PostgreSQL 16.16 + pgvector 0.8.7 (Docker Compose của Day 6, cổng local 55433), bảng `chunks`, `strategy='B'`. Đo bằng cosine (`<=>`, score = 1 − khoảng cách), **tìm kiếm chính xác** (không dùng index xấp xỉ). Khi điểm bằng nhau thì xếp theo `chunk_id`.
  - **Embedding:** `BAAI/bge-m3`, 1024 chiều, chuẩn hóa L2, không dùng tiền tố. Chạy local bằng sentence-transformers 6.3.0 và torch 2.14.1+cu126, fp16 trên RTX 3050 Laptop 4 GB.
  - Query được embed **bằng đúng model đã dùng để index**. `DenseRetriever` kiểm tra tên model và số chiều trong DB khi khởi tạo, khác thì báo lỗi.
  - **Không index lại**, nên không phát sinh chi phí embedding mới.
  - **Kiểm tra chéo:** dense trên dev cho Hit@1 = 14/15 và Hit@3 = 15/15, **trùng với cấu hình C ở Day 6**.

- **Agent model, cách chạy và phân chia dev/test:**
  - **Model:** `gemini-3.1-flash-lite`, gọi qua endpoint tương thích OpenAI của Google AI Studio, free tier. Theo trang quota của tài khoản: 15 request/phút, 500 request/ngày. temperature = 0, `max_tokens` = 1200.
  - Mọi lần chạy dùng để so sánh và đánh giá (dev A0/A1/A2, test A0/A1) đều dùng **cùng model này**.
  - Lần chạy thử ban đầu bằng `nvidia/nemotron-3.5-super-vl:free` qua OpenRouter (thư mục `results/agent/dev/smoke_dense`) **chỉ để kiểm tra code**. Lần này không chấm và không đưa vào so sánh. Lý do đổi model: hạn mức free của OpenRouter là 50 request/ngày, không đủ để chạy hết benchmark trong buổi lab.
  - **Chia split:** giữ nguyên chia dev/test của Day 6 (15/5 câu có đáp án, chia theo chủ đề). Thêm 4 câu thiếu bằng chứng: 2 câu dev (u01, u02) và 2 câu test (u03, u04). **Tổng 24 câu: dev 17, test 7.**
  - Câu test của Day 6 chỉ được chạy một lần ở Day 6 và không dùng để chỉnh hệ thống, nên vẫn dùng làm test cho Day 7.

- **Cách xác định câu có hoặc thiếu evidence, và tiêu chí chấm:**
  - **Câu có đáp án:** mỗi câu có `gold = [{source_id, page, evidence_all}]`. Các phần tử của `gold` là evidence thay thế cho nhau.
  - **Định nghĩa "hit"** ([src/common.py](../../src/common.py)), dùng chung cho mọi phương pháp. Một chunk hit gold khi thỏa đồng thời:
    - đúng `source_id`;
    - trang gold nằm trong khoảng `[page, page_end]` của chunk;
    - chứa **đủ mọi đoạn** evidence, so sau khi chuẩn hóa NFC và bỏ khoảng trắng.
  - Hàm `hit()` của starter chỉ so `page` bằng đúng một trang, nên chấm trượt oan 4 câu có chunk vắt trang (q06, q09, q11, q20). Vì vậy mình không dùng hàm đó.
  - **Câu thiếu bằng chứng** ([data/unanswerable.json](../../data/unanswerable.json)) được chọn **gần chủ đề**: corpus có nhắc đến vấn đề nhưng không có con số hoặc quy định cụ thể. Mỗi câu ghi `absence_reason` và cách kiểm tra:
    - đọc toàn văn điều luật gần nhất;
    - dò từ khóa và regex trên **toàn bộ 356 chunk**, không chỉ trong top-k;
    - các quy tắc này được kiểm tra lại tự động mỗi lần chạy `export_day6`.

    | ID | Câu hỏi | Vì sao gần chủ đề | Vì sao thiếu |
    |---|---|---|---|
    | u01 (dev) | Lương tối thiểu vùng I bao nhiêu đồng/tháng? | Điều 91 BLLĐ nói về lương tối thiểu theo vùng | Điều 91 giao Chính phủ công bố; toàn corpus không có số tiền nào |
    | u02 (dev) | Tỷ lệ đóng BHXH vào quỹ hưu trí – tử tuất? | Luật Việc làm có "đóng 1%" | 1% là bảo hiểm thất nghiệp; tỷ lệ BHXH thuộc Luật BHXH |
    | u03 (test) | Lao động nam nghỉ bao nhiêu ngày khi vợ sinh? | Điều 139 khoản 5 nhắc lao động nam khi vợ sinh con | Khoản 5 chỉ dẫn chiếu pháp luật BHXH, không có số ngày |
    | u04 (test) | Không đóng BHTN bị phạt bao nhiêu tiền? | Corpus nhắc "xử phạt vi phạm hành chính" | Không có mức tiền phạt nào trong corpus |

    Có 1 ứng viên bị loại khi kiểm tra: "điều kiện hưởng lương hưu". Trang 90–92 của BLLĐ có phần sửa đổi Luật BHXH ("đủ 20 năm đóng BHXH"), nên câu này thực ra **có** đáp án.
  - **Chấm tay theo [BENCHMARK.md](../../BENCHMARK.md):**
    - Grounded answer: 0 = sai hoặc thiếu evidence, 1 = đúng một phần, 2 = đúng và đủ. Từ chối câu có đáp án chấm 0.
    - Citation: hợp lệ khi có trong context **và** nguồn thật sự hỗ trợ câu trả lời; đã đọc nguyên văn chunk để xác nhận.
    - Abstention: từ chối mà không thêm khẳng định vô căn cứ.
    - Context (test): 0, 1 hoặc 2.
  - **Quy tắc chấm học viên đã chọn:** câu trả lời đúng nhưng **có khẳng định không nằm trong context** chấm 1 điểm (xem q07).
  - **Quy trình chấm:** nhãn được soạn với sự hỗ trợ của AI, dựa trên bảng đối chiếu nguyên văn chunk ([src/review_sheet.py](../../src/review_sheet.py)). Học viên đã duyệt và quyết định các trường hợp biên (q07, q20).

## Retrieval và thử nghiệm

Năm phương pháp đã triển khai trong [src/retrievers.py](../../src/retrievers.py). Mọi phương pháp đều trả về: ID, text, score, rank, nguồn, trang và metadata.
1. **Dense:** bge-m3 + pgvector Day 6.
2. **BM25:** Okapi BM25 tự cài (k1 = 1,5, b = 0,75). Thử 2 cách tách từ tiếng Việt:
   - theo âm tiết;
   - âm tiết + bigram âm tiết, để xấp xỉ từ ghép như `thử_việc`.
3. **Hybrid RRF:** gộp thứ hạng của dense và BM25 bigram, k = 60, mỗi bên lấy 20 ứng viên. Dùng thứ hạng thay cho score vì score cosine và score BM25 khác thang đo.
4. **Rerank:** lấy 20 ứng viên rồi xếp lại bằng cross-encoder `BAAI/bge-reranker-v2-m3` (fp16).

**Filter:** chỉ dùng `year`, lấy từ trường `filter` mà **người dùng cung cấp** trong benchmark (q12, q13, q14, q20). Filter không bao giờ lấy từ gold. Trong agent, thông tin này được ghi vào câu hỏi ("hồ sơ áp dụng Luật Việc làm 2025"), và **model tự quyết định** có truyền `year` vào tool hay không; trace ghi lại lựa chọn đó.

**Đo lường:** Hit@1, Evidence Hit@3 và Context Evidence Rate có mẫu số là số câu có đáp án (dev 15, test 5). Câu thiếu bằng chứng vẫn được chạy và lưu trace nhưng không vào mẫu số. Context budget tính bằng **ký tự**, bao gồm nhãn nguồn.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense bge-m3 + pgvector, top-3, filter năm người dùng cung cấp, không rerank, 3.000 ký tự | Phương pháp bắt buộc | dev | 14/15 (93%) | 15/15 (100%) | 15/15 (100%) | 38 ms/query; 0 USD |
| Phương pháp 2 | BM25 âm tiết, top-3, cùng filter, 3.000 ký tự | Lexical; xem ảnh hưởng tách từ | dev | 10/15 (67%) | 14/15 (93%) | 14/15 (93%) | 3 ms; 0 USD |
| Phương pháp 2b | BM25 âm tiết + bigram | Bigram xấp xỉ từ ghép tiếng Việt, sẽ tăng độ chính xác | dev | 12/15 (80%) | 14/15 (93%) | 14/15 (93%) | 3 ms; 0 USD |
| Phương pháp 3 | Hybrid RRF (dense + BM25 bigram), k = 60, 20 ứng viên mỗi bên | Kết hợp nghĩa và từ khóa sẽ tốt hơn từng phương pháp | dev | 12/15 (80%) | 14/15 (93%) | 14/15 (93%) | 30 ms; 0 USD |
| Cải thiện 1a (phương pháp 4) | Rerank bge-reranker-v2-m3 trên 20 ứng viên dense | Cross-encoder phân biệt được Điều 57/58 (q14) và Điều 35/36 (q05) | dev | 13/15 (87%) | 15/15 (100%) | 15/15 (100%) | 488 ms; 0 USD |
| Cải thiện 1b | Rerank trên 20 ứng viên hybrid | Rerank sửa được thứ hạng xấu của RRF (q03) | dev | 13/15 (87%) | 15/15 (100%) | 15/15 (100%) | 491 ms; 0 USD |
| Cải thiện 2 (agent A1) | Dense, **top-2**, 4.000 ký tự | Chunk đúng luôn ở hạng 1–2 trên dev, nên giảm token mà giữ chất lượng | dev | 14/15 (93%)* | 15/15 (100%)* | 15/15 agent | xem bảng agent |
| Cải thiện 3 (agent A2) | Dense, top-3, 6.000 ký tự + quy tắc prompt 7 | Không tự diễn giải điều luật được dẫn chiếu mà chưa đọc (lỗi q07) | dev | 14/15* | 15/15* | 15/15 agent | xem bảng agent |
| Cấu hình đã chọn (A1) | Dense, top-2, 4.000 ký tự | Chốt theo tiêu chí công bố trước | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 24 ms; 0 USD |
| Baseline (A0) | Dense, top-3, 6.000 ký tự | | test | 5/5 (100%) | 5/5 (100%) | 5/5 (100%) | 28 ms; 0 USD |

\* A1 và A2 dùng cùng retriever dense, nên ranking giống baseline. Hit@1 và Hit@3 tính trên ranking của retriever với câu hỏi gốc. "Context Evidence" của agent là context cuối cùng gửi model (tự kiểm tra khớp chuỗi, 15/15 ở cả ba cấu hình).

**Trên test** (5 câu có đáp án), BM25 bigram và hybrid cũng đạt 5/5 ở cả ba chỉ số. Test quá nhỏ và dễ để phân biệt các phương pháp.

**Lỗi dev #1: retrieval (hybrid và rerank).**
- **q03** là câu diễn đạt lại: "hợp đồng **có kỳ hạn** được **kéo dài** tối đa…".
  - BM25 bắt nhầm các chữ "thời hạn", "kéo dài", nên trả về Điều 155 (thời hạn giấy phép lao động) và Điều 126.
  - Hybrid RRF kéo chunk đúng (Điều 20) **từ hạng 1 của dense xuống hạng 9**. Lý do: Điều 20 không có trong top-20 của BM25, nên thua các chunk xuất hiện ở cả hai danh sách dù xếp thấp.
- **q05:** trong RRF, Điều 35 và Điều 36 **bằng điểm tuyệt đối** (một bên dense 1 + BM25 2, bên kia dense 2 + BM25 1). Phá hòa theo ID nên chọn sai.
- **Rerank** sửa được q14 (Điều 57 lên hạng 1) và q05, **khớp giả thuyết**. Nhưng rerank làm hỏng hai câu khác:
  - q03: cross-encoder đẩy Điều 155 ("thời hạn… tối đa 02 năm") lên trên Điều 20, vì bị khớp bề mặt với cụm "thời hạn… tối đa" trong câu hỏi.
  - q13: Điều 35 và Điều 39 có điểm rerank bằng nhau (0,929).
  - Kết quả ròng: Hit@1 giảm từ 14 xuống 13, và chậm hơn khoảng 13 lần.
- **Quyết định** (theo tiêu chí retrieval đã ghi trước: Hit@3, rồi Hit@1, rồi tốc độ): giữ **dense** làm retriever cho agent.

**Lỗi dev #2: grounding của agent (q07).**
- Agent baseline trả lời đúng mức trợ cấp thôi việc ("một nửa tháng tiền lương", Điều 46).
- Nhưng agent tự diễn giải cụm "trường hợp quy định tại điểm e khoản 1 Điều 36" thành *"tự ý bỏ việc không có lý do chính đáng từ 05 ngày làm việc liên tục"*.
- Điều 36 **không có trong context**: đã kiểm tra, không chunk nào trong context chứa "05 ngày". Đây là kiến thức sẵn có của model. Nội dung đúng với luật thật nhưng không có nguồn hỗ trợ, nên học viên chấm 1 điểm.
- **Giả thuyết:** thêm quy tắc 7 vào prompt. Ví dụ trong quy tắc **cố ý không lấy từ benchmark** để tránh "dạy tủ".
- **Kết quả A2:** q07 chỉ còn nêu nguyên văn "điểm e khoản 1 Điều 36", không diễn giải thêm; không câu nào xấu đi. Grounded tăng từ 1,93 lên 2,00.
- A1 (chỉ đổi top-k) **lặp lại đúng lỗi này**, khớp dự đoán vì A1 không nhằm sửa lỗi grounding.

**Lý do chọn cấu hình**, áp đúng [selection_criteria.md](../../experiments/selection_criteria.md) đã ghi trước khi chạy agent trên dev:
1. Grounded answer: A0 và A1 đạt 1,93, A2 đạt 2,00. Chênh **1 câu** (0,067 ≤ 0,07), nên theo quy tắc hòa đã công bố, ba cấu hình **ngang nhau**.
2. Từ chối đúng 2/2 và từ chối nhầm 0/15 ở cả ba, nên ngang nhau.
3. Context Evidence 15/15 ở cả ba, nên ngang nhau.
4. Số lượt gọi LLM bằng nhau (2,0/câu). Thời gian trung vị: A1 3,54 s, A0 4,08 s, A2 4,91 s. Vậy **chọn A1**.

**Ghi chú trung thực:** A2 là cấu hình duy nhất sửa được lỗi grounding, nhưng không vượt ngưỡng hòa đã đặt trước. Ngưỡng "1 câu" trên dev 15 câu là thô. Chênh lệch thời gian giữa các cấu hình cũng có thể một phần do nhiễu: độ trễ của Gemini dao động 2,5–48 s/câu. Không đổi tiêu chí sau khi xem kết quả.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** ([src/agent.py](../../src/agent.py)):
  - **Tool:** một tool `search_law(query, year ∈ {2013, 2019, 2025}?)`. Model tự viết query (ví dụ q03 tự đổi "có kỳ hạn" thành "xác định thời hạn") và tự quyết định có lọc theo năm hay không.
  - **Điều kiện dừng:**
    - model trả lời bằng text; hoặc
    - đạt `max_searches = 3`, khi đó gọi lần cuối với `tool_choice = "none"` để buộc trả lời; nếu model vẫn gọi tool thì ghi `abstain`, lý do `search_limit`.
    - Lý do chọn 3 lượt: đủ cho truy vấn viết lại hoặc truy vấn đa bước mà vẫn chặn vòng lặp. Thực tế mọi câu dev và test đều **chỉ tìm 1 lần**.
  - **Chọn context:**
    - bỏ chunk đã có từ lượt trước;
    - giữ thứ tự rank;
    - tổng context không vượt budget (6.000 ký tự cho A0/A2, 4.000 cho A1);
    - **không cắt giữa chunk**, để không làm mất điều kiện hay ngoại lệ của điều luật;
    - mỗi đoạn gắn nhãn `[chunk-id] tên luật — trang`.
  - **Chính sách trả lời:**
    - mỗi ý phải có `[chunk-id]`;
    - không đủ bằng chứng thì câu trả lời phải bắt đầu bằng "KHÔNG ĐỦ BẰNG CHỨNG", và được ghi `abstain`;
    - trích ID không có trong context, hoặc không trích gì, thì chuyển thành `abstain` (lý do `invalid_or_missing_citations`); câu trả lời gốc vẫn được giữ trong trace. Không câu nào chạm tới quy tắc này.
    - Nội dung tài liệu được khai báo là **dữ liệu, không phải chỉ dẫn**.
  - **Xử lý lỗi:**
    - retry có backoff (5/15/30/60 s) cho lỗi 429, 5xx và timeout;
    - hết retry thì ghi `status: "error"` kèm chi tiết, **không thay bằng kết quả giả**;
    - với OpenRouter, code chặn model trả phí để tránh phát sinh nợ.
  - **Lỗi hệ thống đã gặp và cách xử lý:**
    1. **Gemini 3 trả lỗi 400 "missing thought_signature"** ở 5 câu đầu của lần chạy A0 đầu tiên. Nguyên nhân: code dựng lại lời gọi tool bằng tay nên làm mất trường `extra_content.google.thought_signature`. Đã sửa bằng cách giữ nguyên toàn bộ dữ liệu provider trả về (`_tool_call_payload`), rồi chạy lại 5 câu đó. Trace cuối không còn lỗi.
    2. Các model free trên OpenRouter (Gemma) trả 429 do quá tải.
    3. `gemini-2.5-flash` trả 404 (không còn cấp cho người dùng mới); `gemini-3.8-flash` timeout quá 120 s.
  - **Test mock** ([tests/test_agent_loop.py](../../tests/test_agent_loop.py), 7 test) kiểm tra: citation bịa, từ chối, giới hạn lượt tìm, budget và bỏ trùng, lỗi API, và việc gold không bị gửi đi. Toàn bộ 22/22 test đạt. Test mock chỉ kiểm tra code, không phải benchmark.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A0 baseline / dev | 17/17 | 1 lượt tìm ×17; 2 LLM call/câu | 0/1/14 — 1,93 | 20/20 | 15/15 | 2/2 | 0/15 |
| A1 (top-2) / dev | 17/17 | 1 lượt tìm ×17; 2 LLM call/câu | 0/1/14 — 1,93 | 19/19 | 15/15 | 2/2 | 0/15 |
| A2 (quy tắc 7) / dev | 17/17 | 1 lượt tìm ×17; 2 LLM call/câu | 0/0/15 — 2,00 | 21/21 | 15/15 | 2/2 | 0/15 |
| **A1 đã chốt / test** | 7/7 | 1 lượt tìm ×7; 2 LLM call/câu | 0/0/5 — 2,00 | 8/8 | 5/5 | 2/2 | 0/5 |
| A0 baseline / test | 7/7 | 1 lượt tìm ×7; 2 LLM call/câu | 0/0/5 — 2,00 | 11/11 | 5/5 | 2/2 | 0/5 |

**Chấm context trên test** (5 câu có đáp án, cả A0 và A1): **5 câu đạt mức 2** (đủ evidence và nguồn), 0 câu mức 1, 0 câu mức 0. Lý do từng câu nằm trong `reviews.json` (trường `context_grade`).

**Token và thời gian:**
- Dev: A0 dùng trung bình 2.212 token vào và 132 token ra, trung vị 4,08 s. A1: 1.899/132, 3,54 s (**−14% token vào**). A2: 2.358/135, 4,91 s.
- Test: A0 2.371/173, 4,33 s. A1 2.027/184, 4,69 s (**−15% token vào**).

**Ba ví dụ có trace:**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q13 (dev, A1) | Có đáp án, có filter do người dùng cung cấp (năm 2025) | `search_law("thời điểm bắt đầu hưởng trợ cấp thất nghiệp…", year=2025)` → Điều 39 LVL2025 (trang 46) hạng 1 | "Theo Luật Việc làm 74/2025/QH15, thời điểm hưởng trợ cấp thất nghiệp là ngày làm việc thứ 11 kể từ ngày nộp đủ hồ sơ… [B-LVL2025-bfcab6f7e936f50f]" | 2 điểm. Đúng nguyên văn Điều 39 khoản 3, và phân biệt đúng với q12 (luật 2013: "ngày thứ 16"), cùng câu hỏi nhưng khác năm |
| u02 (dev, A0) | Thiếu bằng chứng (câu bẫy) | `search_law("mức đóng bảo hiểm xã hội vào quỹ hưu trí và tử tuất…")` → Điều 57 LVL2013, Điều 33 LVL2025, Điều 168 BLLĐ | "KHÔNG ĐỦ BẰNG CHỨNG: … chỉ quy định về trách nhiệm tham gia BHXH [B-BLLD2019-45c2…] và mức đóng bảo hiểm thất nghiệp [B-LVL2013-defa…], [B-LVL2025-8a3f…]… không nêu cụ thể tỷ lệ… quỹ hưu trí và tử tuất." | Từ chối đúng. Không bị lừa bởi con số 1% (bảo hiểm thất nghiệp); 3 citation hỗ trợ đúng các ý nêu |
| q07 (dev, A0) | Có đáp án, **lỗi** | `search_law("trợ cấp thôi việc mỗi năm làm việc…")` → Điều 46, Điều 47 BLLĐ, Điều 50 LVL2013 | "…mỗi năm làm việc được trợ cấp một nửa tháng tiền lương [B-BLLD2019-7ddb…]… (trừ trường hợp… người lao động tự ý bỏ việc mà không có lý do chính đáng từ 05 ngày làm việc liên tục trở lên)" | 1 điểm. Phần "05 ngày" lấy từ Điều 36, không có trong context. A2 sửa được lỗi này |

Ngoài ra, ở lần chạy thử bằng Nemotron (u01), model tự chèn con số "4.420.000 đồng" (kiến thức sẵn có) vào **query** lượt tìm thứ 2, nhưng không đưa vào câu trả lời. Đây là dấu hiệu model có kiến thức ngoài tài liệu; quy tắc grounding là cần thiết.

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace agent: `results/agent/<split>/<config>/traces.json`. Mỗi câu ghi:
    - từng lượt gọi LLM: model thực tế phục vụ, thời gian, token, số retry, tool call;
    - từng lượt tìm: query, filter, top-k, kết quả kèm rank, score, trang, chunk nào được thêm vào context, chunk nào trùng, chunk nào bị bỏ do vượt budget;
    - context cuối, câu trả lời, citation, status và lý do.
  - Nhãn chấm tay: `reviews.json` cùng thư mục. Tổng hợp bằng `starter/agent_metrics.py` ra `agent_summary.json`. Bảng đối chiếu nguyên văn: `review_sheet.md`.
  - Trace retrieval: `results/retrieval/<split>/<config>/traces.json` (ghi rank, score, hit, `component_ranks` cho hybrid, `base_rank` cho rerank).
  - Lệnh chạy: xem [README.md](../../README.md#bài-làm-day-7).
  - **Chi phí: 0 USD.** Embedding và rerank chạy local; LLM dùng free tier.
    - Số lượt gọi LLM trong các trace cuối: dev 3 × 34 = 102, test 2 × 14 = 28.
    - Cộng thêm khoảng 5 request lỗi 400 (đã chạy lại), khoảng 6 request thử model, và 5 request chạy thử Nemotron.

- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau:**
    - Rerank: +q14, +q05 nhưng −q03, −q13; Hit@1 từ 14 xuống 13; chậm khoảng 13 lần. Không chọn.
    - Quy tắc 7 (A2): q07 từ 1 lên 2 điểm, không câu nào giảm, nhưng tốn thêm khoảng 7% token. Không được chọn vì nằm trong ngưỡng hòa.
    - top-2 (A1): giữ nguyên chất lượng, giảm khoảng 14–15% token. Được chọn.
  - **Đánh đổi:** chất lượng gần như ở mức trần. Khác biệt thật nằm ở chi phí token (A1) và độ chặt grounding (A2). Với bài toán pháp lý, thêm một ý ngoài tài liệu là rủi ro thật, nên nếu triển khai thực tế, mình đề xuất kết hợp top-2 với quy tắc 7. Cấu hình kết hợp này **chưa được đo**, nên không đưa vào kết luận.
  - **Giới hạn:**
    - **Benchmark nhỏ** (dev 15 + 2, test 5 + 2) và **gần mức trần**: dense đã đạt 15/15 Hit@3 trên dev, mọi agent đạt 2,00 trên test. Test **không phân biệt được** A0 với A1 về chất lượng; mỗi câu chiếm 6,7% (dev) hoặc 20% (test).
    - Câu hỏi đa số là tra cứu một điều luật, nên chưa kiểm tra được khả năng tìm nhiều lượt (mọi câu chỉ tìm 1 lần).
    - **Một người chấm** (học viên, có AI hỗ trợ soạn nhãn), không đo độ đồng thuận giữa người chấm.
    - Corpus là **nguyên văn khi ban hành**: BLLĐ 2019 chưa gồm các sửa đổi năm 2025. Câu trả lời "đúng theo corpus" có thể khác luật hiện hành. Còn sót lỗi tách âm tiết do lớp text PDF (ví dụ "1 2 tháng").
    - Model free tier có thể bị đổi hoặc gỡ; độ trễ dao động lớn. Google có thể dùng prompt của free tier để cải thiện sản phẩm; corpus là public domain nên không có dữ liệu cá nhân.
    - Filter năm lấy từ trường `filter` mà benchmark giả định người dùng cung cấp trước; chưa đánh giá trường hợp người dùng không nói năm mà câu hỏi phụ thuộc phiên bản luật.

## Kết luận
- Agent tìm đúng điều luật ngay lượt đầu ở mọi câu, trả lời có citation hợp lệ 100%, từ chối đúng 4/4 câu thiếu bằng chứng (dev và test) và không từ chối nhầm câu nào.
- Dense trên pgvector Day 6 là phương pháp retrieval tốt nhất trên dev. BM25, hybrid RRF và rerank không vượt dense; nguyên nhân đã được phân tích bằng trace.
- Cấu hình chốt theo tiêu chí công bố trước là **A1 (dense, top-2, budget 4.000 ký tự)**: giữ chất lượng, giảm khoảng 15% token. Phát hiện quan trọng nhất về chất lượng là lỗi grounding khi điều luật dẫn chiếu sang điều khác (q07); quy tắc prompt 7 sửa được lỗi này trên dev.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
