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

Trong sidebar chọn **AI Review Analyst**, giữ Hanuri. Hộp **Question** mặc định hỏi “Nhân viên phục vụ có vấn đề gì không?”; nhấn **Ask**. Đọc **AI Analysis**, nhãn **Generation mode**, tên Qwen và phương pháp retrieval. Vùng **Evidence used** hiển thị E1/E2, sentiment, aspect, rating và đoạn review đã cung cấp cho Qwen. Review ID gốc, ánh xạ claim–review, prompt hash và trạng thái validator nằm trong **Technical details**. “Khi nhãn là AI synthesis, từng claim của Qwen có ref và đã qua kiểm tra hỗ trợ; khi nhãn là Evidence fallback, hệ thống trích đánh giá gốc vì hai lần tạo không đạt.” Câu phục vụ trong lượt nghiệm thu mới vẫn fallback do Qwen đưa khẳng định tần suất không có hỗ trợ; đọc đúng nhãn live thay vì hứa một chế độ.

Thay câu hỏi thành “Khách nói gì về chất lượng món ăn?” và nhấn **Ask**. Lượt kiểm tra sản phẩm mới đã cho AI synthesis ngắn, trích [E1][E2] từ hai review; đây là kết quả demo riêng, không thay số trong benchmark RAG V3 đóng băng. Mở **Technical details** nếu cần giải thích ánh xạ E-ref sang review ID. Nếu kết quả live khác, đọc đúng nhãn thực tế và dùng `AI_ANALYST_SYNTHESIS_ACCEPTANCE.md` để trình bày lượt kiểm tra đã lưu.

## 6:40–7:30 — Từ chối đúng và nhân quả

Hỏi “Khách có phàn nàn về giá không?” rồi nhấn **Ask**. Chỉ dòng **Không đủ bằng chứng để kết luận.** vì review khen giá rẻ không thể chứng minh phàn nàn giá cao. Có thể hỏi tiếp “Tại sao đánh giá của nhà hàng này giảm?”; hệ thống không suy nguyên nhân chỉ từ review tiêu cực gần thời điểm. Đây là cơ chế bảo vệ, không phải lỗi thiếu mô hình.

## 7:30–8:30 — LLMOps và kết luận

Trong sidebar chọn **LLMOps Monitor**. Chỉ mục **PRODUCTION / DEMO MONITORING**: RAG V3, Qwen 3B, prompt/model version và các request log không chứa câu hỏi/văn bản review thô. Kéo xuống **RESEARCH METRICS** để nhắc rằng điểm benchmark VN-MSASar tách biệt khỏi phân bố suy luận Foody. “Đánh giá RAG V3 cố định có sáu trường hợp: một câu tổng hợp Qwen được chấp nhận, hai fallback có nguồn, ba từ chối đúng và không có câu được chấp nhận mà thiếu hỗ trợ. Tập này nhỏ; nhóm bọn em chỉ kết luận hệ thống demo có thể kiểm tra chứng cứ và từ chối thận trọng, chưa khẳng định độ tin cậy tổng quát.”

## Phần mở rộng Sales / Review / Cross (2–3 phút, khi PostgreSQL sẵn sàng)

Trong **AI Review Analyst**, giữ câu hỏi review mặc định về phục vụ Hanuri và nhấn **Ask**: chỉ câu trả lời RAG V3 cùng trích dẫn `[E1]`; Hanuri mặc định không được coi là cùng ID với một chi nhánh Hanuri trong dữ liệu đơn hàng.

Đổi câu hỏi thành **“Doanh thu tháng 2025-05 là bao nhiêu?”** và chọn **Thai Market Restaurant - 95 Cao Thắng [ID 1853]** trong hộp Sales restaurant ID. Nhấn **Ask**: nguồn **SQL Analytics**, doanh thu Delivered **492.320 VND**, **4 đơn**, tăng **8,25%** so với tháng 2025-04. Đây là số từ PostgreSQL, không do Qwen tính.

Đổi câu hỏi thành **“Doanh thu tháng 2025-05 và khách phàn nàn điều gì?”**, giữ đúng Sales ID 1853. Nhấn **Ask**: nguồn **Hybrid SQL + RAG**; chỉ số doanh thu vẫn đến từ SQL, phần phàn nàn phải có review cùng nhà hàng/cùng tháng và citation `[E1]`. Đọc câu giới hạn: sự trùng thời gian **không chứng minh quan hệ nhân quả**. Nếu chọn một ID không có ánh xạ xác minh, hệ thống phải từ chối ghép hai nguồn. Các đơn có ngày tương lai so với ngày hiện tại ở Việt Nam không được trình bày như doanh thu đã quan sát.
