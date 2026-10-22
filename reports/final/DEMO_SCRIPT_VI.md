# Kịch bản demo 7–10 phút

## Trước khi vào lớp

Từ thư mục dự án, chạy `.\scripts\start_demo.cmd`. Đợi bảng `demo-preflight` có chín dòng `PASS`, sau đó mở địa chỉ Streamlit được in ra. Chuẩn bị sẵn `KEY_NUMBERS.json`, `MODEL_COMPARISON.csv`, `reports/final/RAG_V3_ACCEPTANCE_REPORT.md`, và `reports/rag_v3/health_report.json` để đối chiếu nếu máy chạy chậm. Không chạy lại huấn luyện, crawl hay xây index trong buổi demo.

## 0:00–0:50 — Bài toán

“Thưa thầy/cô, nhóm bọn em xây dựng hệ thống đọc đánh giá nhà hàng tiếng Việt. Mục tiêu là thấy được xu hướng ý kiến khách hàng và hỏi đáp dựa trên đánh giá có thể kiểm tra được. Phần nghiên cứu dùng VN-MSASar có nhãn để so sánh mô hình; phần sản phẩm dùng dữ liệu Foody thật để phân tích nhà hàng. Bọn em không coi dự đoán trên Foody là độ chính xác đã được kiểm chứng.”

## 0:50–1:40 — Dữ liệu và luồng xử lý

Mở `FINAL_ARCHITECTURE.md` hoặc slide kiến trúc. Chỉ đường đi: Foody CSV → làm sạch → phân loại cảm xúc và khía cạnh → luật phát hiện vấn đề → mart nhà hàng → Chroma/BM25 → RAG V3 → giao diện. Nêu 33.946 dòng nguồn, 31.694 lượt suy luận AI hợp lệ, 2.110 tên nhà hàng, 32.457 tài liệu truy xuất. “Khi mở demo, hệ thống chỉ đọc các artifact đã đóng băng, không thu thập hay huấn luyện lại.”

## 1:40–2:50 — Câu chuyện thực nghiệm

Mở `MODEL_COMPARISON.csv` hoặc slide Task 2. “Tập nghiên cứu gồm 5.869 dòng tiếng Việt về nhà hàng, chia 4.140/842/887 cho train/validation/test và kiểm tra trùng review/text giữa các phần. Lớp trung tính rất ít, chỉ 43 dòng test. Rule-based đạt macro F1 0,431; TF-IDF logistic regression đạt 0,683; Qwen V1 đạt 0,622; V2-A đạt 0,614 nhưng phép so V2-A chỉ mang tính thăm dò vì test cũ đã được xem. Vì vậy nhóm bọn em giữ TF-IDF làm mô hình sentiment cho demo. Mô hình aspect riêng đạt micro F1 0,832 và macro F1 0,634.” Không gọi các con số này là độ chính xác trên Foody.

## 2:50–4:10 — Restaurant Intelligence

Trong sidebar chọn **Restaurant Intelligence**. Ở hộp **Restaurant**, chọn mặc định **Hanuri - Quán Ăn Hàn Quốc - Nguyễn Đình Chiểu**. Chỉ các thẻ đầu trang: số review, điểm rating trung bình, số review AI hợp lệ, và tỷ lệ dự đoán positive/neutral/negative. Cuộn xuống biểu đồ tỷ lệ negative theo thời gian, phân bố aspect, issue heuristic, và historical rule alerts. Nói rõ rating là dữ liệu nguồn; sentiment/aspect là dự đoán; alert là luật lịch sử, không phải kết luận hiện tại. Có thể nhấn **Generate Qwen health report** nếu server phản hồi ổn; chỉ từng mục và nhãn fallback nếu có.

## 4:10–4:50 — Review Explorer

Trong sidebar chọn **Review Explorer**. Chỉ hai vùng **SOURCE** (review, rating, ngày) và **AI** (sentiment, confidence, aspect, issue). Thử lọc `negative` hoặc `service`, rồi cho thấy cùng `review_id` nối dữ liệu nguồn với dự đoán. “Confidence ở đây chưa được hiệu chỉnh và không chứng minh dự đoán đúng.”

## 4:50–6:40 — AI Review Analyst và chứng cứ

Trong sidebar chọn **AI Review Analyst**, giữ Hanuri. Hộp **Question** mặc định hỏi “Nhân viên phục vụ có vấn đề gì không?”; nhấn **Ask**. Đọc **Answer**, **Generation mode**, tên Qwen và phương pháp retrieval. Mở **Supporting review evidence**: chỉ review ID, trích văn bản, rating, predicted sentiment, aspects. “Khi nhãn là Qwen synthesis, câu được tạo đã đi qua bộ kiểm tra từng mệnh đề; khi nhãn là Evidence-based fallback, hệ thống trích dẫn đánh giá gốc vì bản tạo không đạt. Nhóm bọn em không giấu fallback.” Kết quả live có thể chuyển sang fallback; hãy đọc nhãn thực tế thay vì hứa trước một chế độ.

Thay câu hỏi thành “Khách nói gì về chất lượng món ăn?” và nhấn **Ask**; trong đánh giá cố định, trường hợp này có chứng cứ và trả lời bằng extractive fallback. Chỉ trích dẫn và nguồn. Nếu thời gian ngắn, bỏ lượt hỏi thứ hai và mở `RAG_V3_ACCEPTANCE_REPORT.md` để chứng minh chế độ fallback đã được đo.

## 6:40–7:30 — Từ chối đúng và nhân quả

Hỏi “Khách có phàn nàn về giá không?” rồi nhấn **Ask**. Chỉ dòng **Không đủ bằng chứng để kết luận.** vì review khen giá rẻ không thể chứng minh phàn nàn giá cao. Có thể hỏi tiếp “Tại sao đánh giá của nhà hàng này giảm?”; hệ thống không suy nguyên nhân chỉ từ review tiêu cực gần thời điểm. Đây là cơ chế bảo vệ, không phải lỗi thiếu mô hình.

## 7:30–8:30 — LLMOps và kết luận

Trong sidebar chọn **LLMOps Monitor**. Chỉ mục **PRODUCTION / DEMO MONITORING**: RAG V3, Qwen 3B, prompt/model version và các request log không chứa câu hỏi/văn bản review thô. Kéo xuống **RESEARCH METRICS** để nhắc rằng điểm benchmark VN-MSASar tách biệt khỏi phân bố suy luận Foody. “Đánh giá RAG V3 cố định có sáu trường hợp: một câu tổng hợp Qwen được chấp nhận, hai fallback có nguồn, ba từ chối đúng và không có câu được chấp nhận mà thiếu hỗ trợ. Tập này nhỏ; nhóm bọn em chỉ kết luận hệ thống demo có thể kiểm tra chứng cứ và từ chối thận trọng, chưa khẳng định độ tin cậy tổng quát.”
