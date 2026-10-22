# Trạng thái Review 2 — Task 2

Trong 24 mục checklist: **DONE 24, IN PROGRESS 0, NOT DONE 0**. Paper-vs-Group đã hoàn tất **kiểm toán nguồn**, không có số liệu bài báo phù hợp để tính chênh lệch; đây không phải tuyên bố tái lập paper thành công.

Các mục DONE: 1–24. Cụ thể, mục 10 DONE nghĩa là bảng đối chiếu và giới hạn so sánh đã được chứng minh bằng nguồn; cả ba baseline đều **NOT DIRECTLY COMPARABLE** với một kết quả paper đã xác minh. Mục 15 DONE theo nghĩa các vai trò quan sát được của từng thành phần truy hồi/an toàn đã thể hiện trong bảng và hình; **không** có ước lượng nhân quả cho việc tắt riêng evidence packet, clause validator hay fallback. Mục 21 DONE: RAG V3 serving được kiểm chứng; image `pipeline` được rebuild và lệnh mặc định kết thúc thành công (exit 0), không chạy ingestion. Mục 22 DONE: CI run `39729302994`, release run `39728569052`, và GHCR manifest `v1.0.0` được xác minh. Các sửa đổi local hiện tại chưa nằm trong những run đó.

Ngoài 24 mục checklist: huấn luyện LLM từ đầu là **NOT DONE / OUT OF SCOPE**; dự án dùng mô hình mở đã tiền huấn luyện. Không xem đây là thiếu sót cần giải quyết trước Review 2.

**Trả lời đề xuất — “Nhóm hiện đang ở hạng mục nào và hạng mục nào khó nhất?”** Nhóm đã hoàn tất 24/24 mục. Phần Docker RAG V3 đã chạy được trên CPU với ba container phục vụ độc lập và truy vấn có dẫn chứng; container pipeline lịch sử đã nạp dependency và kết thúc lệnh mặc định thành công. Bảng ablation làm rõ vai trò quan sát được của từng thành phần nhưng không tuyên bố tác động nhân quả nếu tắt từng lớp an toàn.

Giới hạn chính xác: không có published macro-F1 trên đúng tập test 887 hàng của nhóm; số paper-group gap để trống. Retrieval Hit@k/MRR là **known-positive** trên sáu câu, không phải full recall. Latency V2 và V3 đến từ hai run khác nhau. Số unsupported accepted là số lỗi được validator phát hiện, không phải đánh giá thủ công trên quần thể. CI/CD đã xác minh chỉ áp dụng cho các commit/tag đã đẩy lên GitHub, không bao gồm sửa đổi local hiện tại. Không thay đổi mô hình, dữ liệu, nhãn, Chroma hay RAG V3 trong lần chốt minh chứng này.

Kiểm tra tại máy: Docker daemon PASS; Compose config PASS; serving-image và pipeline-image build PASS; pipeline mặc định exit 0; Qwen 3B, API, Streamlit HTTP health PASS; truy vấn RAG V3 và event LLMOps PASS. `pytest` **288 passed, 1 warning**; Ruff PASS; `compileall -q src app scripts` PASS.
