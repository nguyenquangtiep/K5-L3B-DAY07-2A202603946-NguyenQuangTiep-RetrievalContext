# Tiêu chí chọn cấu hình (chốt trước khi chạy cấu hình agent nào trên dev)

Ghi ngày 2026-11-16, trước khi chạy agent baseline trên toàn bộ dev. Test chỉ chạy một lần sau khi đã chốt cấu hình theo tiêu chí này.

## Phạm vi so sánh
- Cùng corpus (356 chunk Day 6, strategy B), cùng benchmark dev (15 câu có đáp án + 2 câu thiếu bằng chứng), cùng định nghĩa hit (`src/common.py`).
- Cùng model agent trong một lượt so sánh; nếu buộc phải đổi model (vd model free bị gỡ), ghi rõ và chạy lại các cấu hình cần so sánh.
- Lần chạy lỗi hệ thống (429/402/timeout) được chạy lại; nếu vẫn lỗi thì báo riêng, không tính là trả lời sai cũng không loại khỏi bảng.

## Thứ tự ưu tiên khi chọn
1. **Grounded answer trung bình /2** trên 15 câu dev có đáp án (chấm tay: 0 sai/thiếu evidence, 1 đúng một phần, 2 đúng và đủ; từ chối câu có đáp án = 0).
2. **An toàn trả lời:** số câu thiếu bằng chứng được từ chối đúng (u01, u02) — ưu tiên cao hơn; sau đó số câu có đáp án bị từ chối nhầm — ít hơn tốt hơn.
3. **Context Evidence Rate** của agent (context cuối có chunk chứa evidence gold).
4. **Chi phí/thời gian:** số lượt gọi LLM trung bình mỗi câu, rồi thời gian trung vị mỗi câu.

Quy tắc hòa: ở mỗi tiêu chí 1–3, các cấu hình chênh nhau **không quá 1 câu** (hoặc ≤ 0,07 điểm trung bình ở tiêu chí 1) được coi là ngang nhau và xét tiếp tiêu chí sau. Nếu vẫn hòa sau tiêu chí 4, chọn cấu hình đơn giản hơn (ít thành phần hơn).

## Retrieval (không dùng LLM)
So sánh trên dev bằng Hit@1, Evidence Hit@3, Context Evidence Rate, MRR@10 và thời gian (`src/eval_retrieval.py`). Retriever dùng cho cấu hình agent "cải thiện 1" là retriever có Evidence Hit@3 cao nhất, hòa thì Hit@1, hòa tiếp thì nhanh hơn.
