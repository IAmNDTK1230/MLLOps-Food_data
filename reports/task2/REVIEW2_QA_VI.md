# Câu trả lời ngắn cho Review 2

**Dữ liệu có bao nhiêu mẫu?** VN-MSASar v3 có 76.785 dòng nguồn; lọc tiếng Việt, nhà hàng còn 5.869 dòng câu–khía cạnh, chia cố định 4.140/842/887 train/validation/test. Foody có 33.946 review thô, 32.457 review ID hợp lệ riêng biệt, 2.110 nhà hàng. Foody dùng cho suy luận/RAG; không có nhãn cảm xúc/khía cạnh gold đáng tin cậy.

**Mất cân bằng lớp ra sao?** Positive 3.714 (63,28%), negative 1.861 (31,71%), neutral 294 (5,01%); tỷ lệ lớp nhiều nhất/ít nhất là 12,63:1. Đây là mất cân bằng đáng kể.

**Rò rỉ dữ liệu?** Chia theo thành phần liên thông của review ID và văn bản chuẩn hóa; số ID và văn bản chuẩn hóa xuất hiện ở nhiều split đều bằng 0. Chưa chứng minh không có near-duplicate mờ.

**Vì sao `class_weight="balanced"`?** Neutral chỉ có 208 dòng train. Trọng số có hiệu lực là âm 1,056; trung tính 6,635; dương 0,526. F1 neutral trên validation tăng từ 0,1702 lên 0,4719, recall từ 0,0930 lên 0,4884. Số dòng không đổi; validation/test không được cân lại.

**Vì sao không SMOTE?** TF-IDF là biểu diễn sparse; SMOTE tạo véc-tơ nội suy chứ không tạo câu tiếng Việt tự nhiên. Trọng số xử lý trực tiếp hàm mất mát, giữ nguyên văn bản và split.

**Vì sao mỗi review là một chunk/document?** Giữ quan hệ review ID–vector 1:1, trích dẫn và lọc nhà hàng rõ ràng. Tuy nhiên 11.235 review (34,6%) vượt giới hạn 128 content token của embedding runtime nên phần đuôi có thể bị cắt khi nhúng. Thử nghiệm tách theo token mới chỉ mô phỏng workload, chưa đo cải thiện retrieval; chưa đổi index đã được chấp nhận.

**Tổng token RAG?** Trên 32.457 review: 3.750.416 content token theo tokenizer MiniLM, 4.105.360 theo tokenizer Qwen2.5-3B. Đây là hai cách mã hóa khác nhau, không cộng thành một tổng token duy nhất; đã tắt truncation chỉ khi đếm.

**Thời gian embedding?** Đã đo 1.054,63 giây, 30,78 document/giây, MiniLM 384 chiều chạy CPU ONNX. Thời gian chính sách chunk khác là ước tính tỷ lệ, không phải chạy lại index.

**Vì sao MiniLM?** Mô hình embedding đa ngôn ngữ 384 chiều là lựa chọn đã được kiểm tra và dùng cho index 32.457 review. Chưa có chứng cứ rằng nó tối ưu; giới hạn 128 token là điểm yếu cần thử ở Task 3.

**Vì sao Chroma/BM25?** Chroma lưu vector/các trường lọc, BM25 bổ sung truy xuất từ khóa; RAG V3 định tuyến theo ý định và kiểm tra bằng chứng. Trên sáu truy vấn có nhãn dương chưa đầy đủ, BM25 và V3 cùng MRR 0,1667, dense là 0,0185. Tập nhỏ này không chứng minh ưu thế tổng quát.

**Vì sao Qwen/Llama/DeepSeek?** Đây là ba họ mô hình open-weight khác nhau; `llama.cpp` là runtime chung để so sánh suy luận trên cùng sáu evidence packet. DeepSeek bản distill là 1,5B, nhỏ hơn Qwen/Llama 3B; kết quả không tách được tác động của kích thước khỏi kiến trúc.

**Vì sao chọn Qwen cho demo?** Trong đánh giá RAG V3 đã chấp nhận, Qwen 3B giảm fallback từ 3 xuống 2 so với Qwen 1,5B trên cùng sáu ca và không có câu chấp nhận bị validator phát hiện thiếu căn cứ. Đây là lựa chọn demo địa phương, không chứng minh Qwen thắng mọi mô hình/họ khác.

**Vì sao fallback?** Khi câu sinh không qua kiểm tra schema, trích dẫn, chủ đề hoặc mệnh đề, hệ thống dùng đoạn bằng chứng trích xuất hoặc từ chối. Qwen 3B cần fallback 2/6 ca; 3 ca thiếu bằng chứng được từ chối đúng.

**Vì sao V1 không vượt baseline?** Macro F1 Qwen V1 = 0,621789 < TF-IDF = 0,682854; neutral F1 = 0,257778 < 0,413043. Lớp neutral nhỏ và 81 văn bản chuẩn hóa có nhãn khác nhau theo aspect (11 trong test) làm bài toán text-only mơ hồ; các yếu tố khác chưa được kiểm định nhân quả.

**V2/V3 sửa gì?** V2-A thử đầu vào có aspect nhưng kết quả test 0,614487 là exploratory vì đã xem test trước. RAG V3 giải quyết truy xuất theo ý định, kiểm tra trích dẫn/mệnh đề và fallback cho demo; không thay đổi điểm sentiment test.

**Chỉ số retrieval?** V3 Hit@5 = 0,0833, Hit@10 = 0,0833, MRR known-positive = 0,1667, latency trung bình 140,97 ms trên sáu truy vấn. Các known-positive chưa đầy đủ nên không xem đây là recall đầy đủ.

**Vì sao chỉ sáu câu hỏi RAG?** Đây là bộ ca chấp nhận cố định cho các tình huống đủ/thiếu bằng chứng, không phải benchmark đại diện. Cần mở rộng câu hỏi, nhãn relevance và đánh giá người dùng ở Task 3.

**Docker/CI/CD?** `docker compose config --quiet` qua; Docker Desktop Linux daemon không chạy nên build/start local chưa kiểm tra. [GitHub CI run 39620310721](https://github.com/IAmNDTK1230/MLLOps-Food_data/actions/runs/39620310721) thành công ở commit `d3a8393`; chưa xác minh release GHCR hay deploy production.

**Giới hạn và Task 3?** Ưu tiên bộ relevance nhiều câu hỏi hơn, thử index tách review dài để khắc phục truncation, đánh giá faithfulness bằng người, kiểm tra robustness neutral/aspect ambiguity và xác minh Docker/GHCR triển khai. Không thay đổi nhãn hoặc coi benchmark sáu ca là kết luận tổng quát.
