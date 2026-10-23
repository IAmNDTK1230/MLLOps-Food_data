# Review 2 — DevOps, kho phân tích và quan sát Foody

## Kiến trúc và trạng thái kiểm chứng

Demo Streamlit/RAG V3/Qwen 3B hiện tại được giữ nguyên. Tệp `docker-compose.devops.yml` chỉ bổ sung PostgreSQL, một Kafka KRaft broker, consumer, Prometheus và một Grafana. Terraform tạo network và bốn named volume; Compose chạy service. Backend `/ask-v3` và Streamlit AI Review Analyst dùng chung helper ghi Prometheus sau khi gọi cùng hàm RAG V3; thuật toán và đầu ra không đổi. Streamlit vẫn gọi RAG V3 trực tiếp, ghi SQLite LLMOps như trước và xuất metric riêng ở cổng nội bộ 8002. Prometheus scrape cả backend lẫn Streamlit; một yêu cầu chỉ được tính ở nơi thực sự thực thi.

Luồng batch: Parquet Foody đã làm giàu → loader → PostgreSQL warehouse. Luồng minh họa streaming: 20 review thật từ cùng Parquet → `foody.review.events` → consumer → `fact_review_events`. Consumer dùng sentiment/aspect **đã dự đoán** trong `fact_review` nếu review có sẵn, không chạy suy luận lại. Prometheus scrape backend, Streamlit và consumer; Grafana tự nhận datasource/dashboard. Kafka không thay thế pipeline batch.

## A. Data Warehouse

PostgreSQL 17 Alpine lưu dữ liệu trong volume `foody_postgres_data`, không ghi database vào Git. Migration SQL `001` tạo `schema_migrations`, `dim_restaurant`, `dim_date`, `fact_review`, `fact_restaurant_daily`, `fact_restaurant_monthly`, `fact_aspect`, `fact_issue`, `fact_review_events`; `002` tạo index. `SELECT version FROM schema_migrations` đã trả `001`, `002`. Loader dùng một transaction để thay thế đúng bảy bảng batch; bảng event độc lập được giữ lại. Lần chạy thứ hai ghi nhận không có migration mới và số hàng không đổi.

| Bảng | Hàng đo thực tế |
| --- | ---: |
| `dim_restaurant` | 2.110 |
| `dim_date` | 4.029 |
| `fact_review` | 32.465 |
| `fact_restaurant_daily` | 27.994 |
| `fact_restaurant_monthly` | 16.603 |
| `fact_aspect` | 6.010 |
| `fact_issue` | 1.758 |
| `fact_review_events` | 20 sau replay |

Enriched Parquet có 33.946 observation; warehouse đếm 32.465 `review_id` duy nhất. RAG index có 32.457 document — ít hơn tám review chưa có sentiment/không thuộc tập indexed. Đây là các mẫu số khác nhau, không phải lỗi đếm. Truy vấn thực tế: Hanuri đứng đầu với 133 review duy nhất; tháng 10/2026 có rating trung bình 6,00 trên các review có rating; sentiment gồm 20.985 positive, 10.813 negative, 659 neutral và 8 chưa có dự đoán; aspect `food` xuất hiện 19.872 lượt dự đoán. SQL mẫu và chính sách tải lại ở `warehouse/README.md`.

## B. Kafka

Kafka 3.9.1 chạy một broker KRaft, không ZooKeeper. Topic duy nhất `foody.review.events` dùng một partition, retention 6 giờ, giới hạn 128 MiB và segment 16 MiB. Broker đặt một partition cho topic offset nội bộ và index tối đa 1 MiB để giảm file thưa; volume `foody_kafka_data` giữ log ngoài Git. Producer mặc định chỉ phát 20 review thật và tạo `event_id` SHA-256 ổn định từ `review_id`; JSON có `schema_version=1.0`, thời gian/rating/comment từ nguồn. Consumer kiểm tra schema và ghi một bản ghi theo khóa `event_id`. Nếu broker restart làm mất offset sau commit database, replay vẫn an toàn vì `ON CONFLICT DO NOTHING`; lỗi database tạm thời không bị commit offset.

Ở cấu hình cuối, phát 20 event: consumer ghi 20; phát lại cùng 20 event: warehouse vẫn 20 hàng. Bộ đếm consumer ghi đúng **40 consumed / 20 inserted / 20 duplicate / 0 failed**. Đây là mô phỏng extension real-time, không phải tuyên bố đã có nguồn review trực tuyến.

## C. CI/CD và version

CI hiện có chạy pytest, Ruff, compileall, import app và Docker build. Bản mở rộng kiểm tra Compose base/DevOps, JSON dashboard, migration và `terraform fmt/init/validate`; không đưa GGUF hay index lên Actions. CD hiện có build và đẩy image gắn release tag/commit identity lên GHCR; **chưa** chứng minh triển khai production từ xa. Thay đổi mới chưa được push, nên chỉ có kết quả local (305 tests, Ruff, compileall, Compose, Terraform) và bằng chứng CI cũ mà nhóm đã xác nhận trước đó. Lần thử pytest đầu tiên gặp quyền ghi thư mục tạm; lần khác bị sandbox chặn socket asyncio của Windows. Chạy lại toàn bộ với thư mục tạm trong repository và quyền socket cục bộ đã hoàn tất 305/305, không có lỗi assertion của ứng dụng.

`config/platform_versions.yaml` ghi project/event/schema version; Git SHA đi vào backend qua Compose. `/version` trả Git SHA, RAG/prompt ID, SHA-256 dataset, sentiment/aspect, embedding revision/hash và Qwen revision/hash, không trả secret hoặc đường dẫn máy cá nhân. Schema thực tế vẫn truy vấn được qua `schema_migrations`.

## D. IaC, network, cổng

Terraform provider Docker được pin và chỉ quản lý `foody_devops_network`, `foody_postgres_data`, `foody_kafka_data`, `foody_prometheus_data`, `foody_grafana_data`. Plan đầu tiên: 5 add, 0 change, 0 destroy. Lần chạy lại: 0/0/0. Compose kết nối consumer → Kafka/PostgreSQL, Prometheus → backend/Streamlit/consumer, Grafana → Prometheus. Qwen/backend/Streamlit tiếp tục cùng network mặc định của demo và cũng tham gia network DevOps. Host mở Streamlit 8501, backend 8000, Qwen 8193 như trước; cổng mới chỉ bind loopback: Grafana 3000, Prometheus 9090, Kafka 9092, PostgreSQL 5433. Cổng metric Streamlit 8002 chỉ ở network Docker. `stop_review2.ps1` chỉ dừng DevOps containers; volume/network và demo thường không bị xóa. `iac_destroy.ps1` cần cờ xác nhận riêng và không nằm trong luồng dừng.

## E. Monitoring và Grafana

SQLite LLMOps giữ sự kiện chi tiết theo request; Prometheus giữ metric vận hành theo thời gian; Grafana trực quan hóa metric Prometheus. Metric chỉ dùng label giới hạn (`route`, `status`, `generation_mode`), không dùng query, prompt, restaurant hoặc review ID. Backend có request count/latency, RAG count/route/mode/evidence, LLM latency/tokens. Consumer có consumed/inserted/duplicate/failed và processing histogram. Prometheus retention 6 giờ/256 MB; Grafana dùng một volume, không plugin. Datasource Prometheus báo `OK`; dashboard `Foody DevOps & LLMOps` được provision tự động với 10 panel. Chín panel trả giá trị số ở lần kiểm tra cuối: API request, RAG request, generation mode, retrieval route, token và Kafka event đều khác 0; error/failure = 0. Panel LLM latency P95 trả giá trị rỗng vì không có đủ mẫu trong cửa sổ rate 5 phút tại thời điểm kiểm tra; metric latency vẫn được thu thập. Không triển khai Loki/Tempo/JMX.

Kiểm tra bổ sung ngày 23/10/2026: ba câu hỏi thật được gửi qua nút **Ask** ở Streamlit AI Review Analyst (dịch vụ, món ăn, giá). Prometheus target `foody-streamlit` ở trạng thái `up=1`; bộ đếm yêu cầu = 3, generation mode `extractive_fallback/qwen/none` mỗi loại 1, route `SERVICE_COMPLAINT/GENERAL_ASPECT/PRICE_COMPLAINT` mỗi loại 1, tổng evidence = 3, LLM latency tích lũy = 49,544 giây, input/output token = 2.273/231. Grafana panel **RAG requests** hiển thị đường số liệu từ 1 lên 3 thay vì `No data`. Ba hàng SQLite `rag_v3_request` có ID 485, 487, 488 và cùng tổng token 2.273/231. Không gọi lại backend `/ask-v3` cho các yêu cầu này, nên tổng trên Grafana vẫn bằng 3, không bị đếm hai lần. Sau khi Docker Desktop dừng và khởi động lại, bộ đếm trong tiến trình Streamlit bắt đầu lại từ 0; dữ liệu lịch sử Prometheus vẫn lưu 3 yêu cầu, 2 lần quan sát LLM latency và 2.273/231 token trong cửa sổ 6 giờ. Cả ba scrape target đều `up`, và dashboard **Last 6 hours** tiếp tục hiển thị API, RAG, generation mode, route, latency, token và Kafka. Không gửi thêm câu hỏi Qwen trong lần kiểm tra sau restart.

## F. Cách demo ngắn

```powershell
.\scripts\devops\start_review2.ps1
python -m src.cli kafka-produce --limit 20
docker compose -f docker-compose.yml -f docker-compose.devops.yml exec -T postgres psql -U foody -d foody -c "SELECT COUNT(*) FROM fact_review_events;"
```

Mở Streamlit `http://localhost:8501`, Grafana `http://localhost:3000`, Prometheus `http://localhost:9090`, backend `/version` và `/metrics` ở `http://localhost:8000`. Gửi lại lệnh producer để chứng minh `event_id` không nhân đôi. Dùng `-ProduceEvents` với `start_review2.ps1` nếu muốn script tự phát 20 event. Dừng riêng phần DevOps bằng `.\scripts\devops\stop_review2.ps1`.

## G. Lưu trữ và giới hạn

Không sao chép dataset, GGUF, embedding index hay Chroma/BM25. Build context loại `release/`, `data/`, `models/`, `tmp/` và Terraform state. Image DevOps dùng layer nhỏ trên image backend hiện hữu; PostgreSQL/Kafka/Prometheus/Grafana dùng một named volume mỗi dịch vụ. Kafka volume sau cấu hình lại là **25,23 MB dung lượng logic** theo `docker system df -v`, dưới mục tiêu 500 MB. Không chạy prune vì image khác có thể đang được dự án khác sử dụng. `docker system df` trước/sau và kích thước mã Git mới nằm trong `DEVOPS_ACCEPTANCE.json`. Streamlit trực tiếp vẫn không đi qua `/ask-v3`, nhưng có scrape target riêng; SQLite giữ audit chi tiết. Đây là stack một máy phục vụ Review 2, không phải triển khai HA hay production.

## Q&A bảo vệ ngắn

**Kafka dùng để làm gì?** Mô phỏng ingestion review gần thời gian thực, có replay và consumer ghi warehouse.

**Đã có CSV/Parquet sao thêm Kafka?** Batch xử lý lịch sử; Kafka minh họa đường event cho review mới. Demo dùng 20 review cũ để tái hiện có kiểm soát.

**Kafka có thay PostgreSQL không?** Không. Kafka giữ event có retention ngắn; PostgreSQL phục vụ truy vấn phân tích bền hơn.

**Tại sao cần warehouse?** Cho phép SQL kết hợp review, thời gian, nhà hàng và mart với schema/version rõ ràng.

**Data mart khác warehouse thế nào?** Mart là tổng hợp phục vụ bài toán cụ thể; warehouse chứa dimension, fact review và các mart đã nạp.

**Prometheus khác Grafana?** Prometheus scrape/lưu chuỗi thời gian; Grafana truy vấn và hiển thị dashboard.

**LLMOps Monitor khác Grafana?** SQLite lưu audit chi tiết theo request, Grafana xem metric tổng hợp vận hành.

**IaC là gì? Docker Compose có phải IaC không?** Cả hai mô tả cấu hình bằng mã; ở đây Terraform quản lý network/volume, Compose quản lý service.

**Terraform quản lý gì?** Một network và bốn volume có tiền tố Foody, không quản lý tài nguyên Docker của dự án khác.

**CI khác CD thế nào?** CI kiểm tra chất lượng/build; CD phát hành image versioned lên GHCR. Chưa có deploy production từ xa.

**Version data/model/prompt thế nào?** Hash dataset/model trong manifest, revision embedding/Qwen, prompt ID, Git SHA, schema/event version và `/version`.

**Consumer chết thì sao?** `restart: unless-stopped` khởi động lại; offset chưa commit được replay, khóa `event_id` chặn insert trùng.

**Tránh duplicate event thế nào?** `event_id` deterministic và PostgreSQL primary key cộng `ON CONFLICT DO NOTHING`.

**Tại sao không Kubernetes?** Review 2 là demo một máy; Compose/Terraform đủ và nhẹ hơn.
