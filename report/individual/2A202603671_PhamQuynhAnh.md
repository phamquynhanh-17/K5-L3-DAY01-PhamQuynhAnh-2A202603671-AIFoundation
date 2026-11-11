# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603671
- **Họ và tên:** Phạm Quỳnh Anh
- **SEED:** 8674

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

---

## 2. Kết quả chính

*Bảng tổng hợp so sánh 4 thế hệ AI trên tập dữ liệu SMS Spam:*

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | ~0.962 | ~0.818 | ~0.884 | ~0.857 | ~0.5s | > 5,000 | ~0.05 | ~0.14 | $0.00 |
| **2. BoW + MLP (deep learning)** | ~0.949 | ~0.836 | ~0.889 | ~0.865 | ~2.5s | > 3,000 | ~0.16 | ~0.34 | $0.00 |
| **3a. Encoder (MiniLM) + kNN** | ~0.938 | ~0.909 | ~0.923 | ~0.900 | ~15.1s* | ~312 | ~2.45 | ~4.90 | $0.00 |
| **3b. Encoder (MiniLM) + LogReg** | ~0.969 | ~0.933 | **~0.951** | **~0.950** | ~15.5s* | ~520 | ~1.48 | ~3.42 | $0.00 |
| **3c. Encoder (MiniLM) + MLP head** | ~0.968 | ~0.924 | ~0.945 | ~0.950 | ~16.2s* | ~465 | ~1.62 | ~3.75 | $0.00 |
| **4a. LLM nhỏ zero-shot (gpt-6-luna)** | ~0.950 | ~0.950 | — | ~0.950 | 0.0s | ~3.5 | ~395 | ~780 | ~$0.05 |
| **4b. LLM lớn zero-shot (gpt-6.1-terra)** | ~0.950 | ~0.950 | — | ~0.950 | 0.0s | ~1.4 | ~880 | ~1,550 | ~$0.50 |
| **4c. LLM nhỏ few-shot** | ~0.950 | ~0.950 | — | ~0.950 | 0.0s | ~3.1 | ~430 | ~820 | ~$0.15 |

*(Ghi chú: Thời gian train của nhóm 3 chủ yếu là bước tiền trích xuất embedding cho toàn bộ tập train ~3.600 tin nhắn bằng MiniLM).*

---

## 3. Dự đoán và thực tế

1. **Dự đoán `dl_curve` (MLP vs LogReg trên BoW):**
   - *Dự đoán:* Kỳ vọng mạng MLP với 1 lớp ẩn và hơn 128.000 tham số sẽ cải thiện F1 đáng kể so với Logistic Regression (chỉ 2.001 tham số).
   - *Thực tế:* F1 của MLP chỉ tăng không đáng kể (~0.01), đồng thời bắt đầu overfit rất nhanh chỉ sau epoch 4–5 khi Val Loss tăng trở lại.
   - *Nguyên nhân lệch:* BoW biểu diễn dưới dạng one-hot/tần suất từ thưa thớt, dữ liệu chỉ có vài nghìn mẫu. MLP nhiều tham số trên vector thưa chủ yếu ghi nhớ nhiễu thay vì học được các trừu tượng hóa ngữ nghĩa sâu sắc.

2. **Dự đoán `emb_tricky` (Tin nhắn ngụy trang):**
   - *Dự đoán:* Nghĩ rằng BoW vẫn có thể bắt đúng 4/6 tin nhờ một số từ khóa phụ.
   - *Thực tế:* BoW thất bại nặng ở các tin spam né từ khóa (ví dụ dùng *"complimentary holiday"* thay vì *"free/prize"*) hoặc bắt nhầm tin bạn bè ham có chữ *"free"* (*"Hey are you free for lunch tomorrow?"*). Trong khi đó, Embedding + LogReg đạt độ chính xác tuyệt đối (6/6).
   - *Nguyên nhân lệch:* Embedding biểu diễn ngữ cảnh toàn câu vào không gian liên tục 384 chiều, hiểu được *"free for lunch"* là rảnh rỗi, tách biệt hoàn toàn với *"free prize/cash"*.

3. **Dự đoán `llm_models` (Model lớn vs Model nhỏ):**
   - *Dự đoán:* Dự đoán model lớn (`gpt-6.1-terra`) sẽ có F1 vượt trội so với model nhỏ (`gpt-6-luna`).
   - *Thực tế:* Trên bài toán phân loại nhị phân cơ bản (tập SUB 40 tin), cả hai đều đạt F1 ngang ngửa nhau (~0.95), nhưng model lớn chậm hơn gấp đôi và chi phí đắt gấp 10 lần.
   - *Nguyên nhân lệch:* Bài toán phân loại SMS không đòi hỏi suy luận phức tạp (complex reasoning) hay kiến thức sâu rộng; một model nhỏ đã đủ hiểu chỉ thị prompt và ngữ nghĩa tin nhắn.

---

## 4. Phân tích 50 case khó (Stress Test)

- **Model tụt nhiều nhất:** **BoW + LogReg** và **BoW + MLP** tụt giảm mạnh nhất (Accuracy rơi từ ~98% ở tập test thường xuống dưới 65% trên bộ 50 case khó). Nhóm bị đánh bại nặng nề nhất là:
  - *Tưởng spam nhưng không phải (`ham_looks_spam`):* Tin chứa nhiều từ giục giã, từ khóa nhạy cảm nhưng ngữ cảnh gia đình/bạn bè.
  - *Né từ khóa (`obfuscation`):* Tin cố tình chèn dấu chấm (`c.a.s.h`), thay chữ bằng số (`C0ngr4tulat10ns`, `w1n`).
  - *Spam giọng tự nhiên (`natural_spam`):* Không dùng từ ngữ spam truyền thống.

- **Hai case cụ thể và cơ chế gây sai:**
  1. **Case #29 (Obfuscation):** `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"`
     - *Nhãn thật:* `spam`.
     - *Cơ chế sai:* Tokenize tách thành các token lạ (`c0ngr4tulat10ns`, `w0n`, `cl4im`). Các từ này không tồn tại trong từ điển `vocab` (top 2000 từ thông dụng), dẫn đến vector đặc trưng toàn số 0 và mô hình dự đoán nhầm thành `ham`.
  2. **Case #1 (Ham looks spam):** `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"`
     - *Nhãn thật:* `ham`.
     - *Cơ chế sai:* Tokenize xuất hiện các từ có trọng số spam cực dương trong BoW: `won`, `call`. Tổng tích vô hướng $w \cdot x + b > 0 \implies p(\text{spam}) > 0.5$, mô hình dự đoán nhầm thành `spam`.

---

## 5. Thử thách cá nhân

- **Đề bài được gán (theo SEED = 8674, Đề #5):**
  > *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*

- **4 tin nhắn tự thiết kế:**
  1. **Spam 1 (Né từ khoá):**
     `"Dear passenger, your accumulated loyalty flight voucher expires tonight. Secure your seat at fl-claim.net immediately."`
     - *Nhãn thực tế:* `spam`.
     - *Dự đoán BoW:* `ham` (Sai). Các từ `accumulated`, `loyalty`, `voucher`, `expires`, `secure` không nằm trong top trọng số spam của BoW hoặc vắng mặt trong vocab, dẫn đến tổng điểm $w \cdot x + b < 0$.
     - *Dự đoán Emb + LogReg:* `spam` (Đúng). Vector embedding của MiniLM nằm rất gần cụm ngữ nghĩa giục giã nhận voucher/quà tặng của các tin spam hàng không/khách sạn.
  2. **Spam 2 (Né từ khoá):**
     `"Regarding your submitted invoice #84920: urgent payment discrepancies noticed. Download attachment via doc-review.co.uk to resolve."`
     - *Nhãn thực tế:* `spam` (phishing).
     - *Dự đoán BoW:* `ham` (Sai). Hoàn toàn không dùng các từ khóa cờ bạc/trúng thưởng truyền thống (`win`, `prize`, `cash`).
     - *Dự đoán Emb + LogReg:* `spam` (Đúng). MiniLM nắm bắt được ngữ cảnh email/tin nhắn giả mạo hóa đơn lừa gạt tải tệp đính kèm.
  3. **Ham 1 (Chứa từ 'free'):**
     `"Are you free tomorrow afternoon? Mom wants to visit the new garden exhibition together."`
     - *Nhãn thực tế:* `ham`.
     - *Dự đoán BoW:* `spam` (Sai). Từ `free` mang trọng số dương rất lớn trong mô hình hồi quy logistic BoW, kéo toàn bộ xác suất $p(\text{spam}) > 0.5$.
     - *Dự đoán Emb + LogReg:* `ham` (Đúng). Embedding mã hóa cụm `"are you free tomorrow"` mang sắc thái hỏi thời gian rảnh rỗi trong hội thoại thường ngày, tách bạch với `"free cash/free gift"`.
  4. **Ham 2 (Chứa từ 'win'):**
     `"Our school debate team managed to win first place today! Celebrating at the local diner tonight."`
     - *Nhãn thực tế:* `ham`.
     - *Dự đoán BoW:* `spam` (Sai). Từ `win` kéo điểm logit lên dương, gây dương tính giả (False Positive).
     - *Dự đoán Emb + LogReg:* `ham` (Đúng). Không gian vector phản ánh rõ ngữ cảnh học sinh chúc mừng giải thưởng tranh biện trường học.

- **Bài học rút ra:**
  - Biểu diễn Bag-of-Words hoàn toàn "mù" trước ngữ cảnh và đồng nghĩa; chỉ cần thay đổi từ vựng hoặc đảo ngữ là mô hình mất tác dụng.
  - Pre-trained Embedding thực sự mang lại bước nhảy vọt về khả năng tổng quát hóa (generalization) bằng cách ánh xạ ngữ nghĩa vào không gian vector liên tục.

---

## 6. Phương án cho sản phẩm thật

- **Bài toán đặt ra:** Lọc 10.000.000 SMS/ngày cho nhà mạng (~116 tin/giây trung bình, cao điểm ~500–1000 tin/giây), ràng buộc SLA nghiêm ngặt: $p95 < 50\text{ ms/tin}$.
- **Đánh giá các phương án:**
  - *Phương án LLM:* Bị loại vì không đạt SLA ($p95 \approx 400 - 1500\text{ ms} \gg 50\text{ ms}$). Chi phí cực kỳ đắt đỏ: 10 triệu tin/ngày $\times$ 30 ngày = 300 triệu tin/tháng $\approx$ $15.000 – $150.000 USD/tháng.
  - *Phương án BoW + LogReg:* Cực nhanh ($p95 < 0.2\text{ ms}$), chi phí tính toán bằng 0, nhưng F1 thấp hơn và dễ bị qua mặt bởi spam ngụy trang / né từ khóa.
  - *Phương án Embedding (MiniLM) + LogReg (3b):* **Lựa chọn tối ưu nhất**. Đạt $p95 \approx 3.42\text{ ms} \ll 50\text{ ms}$, thông lượng xử lý nhanh, F1 đạt mức cao nhất (~0.951), chống chịu tốt với biến thể từ ngữ, chi phí vận hành chỉ cần vài máy chủ CPU/GPU thông thường.
- **Thiết kế kiến trúc hệ thống tối ưu (Router / Hybrid 2 tầng):**
  1. **Tầng 1 (Fast Filter - BoW/Heuristic):** Xử lý 100% lưu lượng. Các tin nhắn có xác suất phân loại rất rõ ràng ($p < 0.05$ là Ham, $p > 0.95$ là Spam) được quyết định ngay lập tức trong $< 0.1\text{ ms}$.
  2. **Tầng 2 (Deep Filter - MiniLM Embedding + Classifier):** Xử lý ~10–20% các tin còn lại nằm trong vùng không chắc chắn ($p \in [0.05, 0.95]$), tốn thêm ~3–5 ms nhưng đảm bảo chất lượng cao nhất mà vẫn thỏa mãn toàn diện SLA.

---

## 7. AI log và điều đã học

### Bảng theo dõi tương tác AI (AI Log)

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời / viết gì | Bạn đã sửa / kiểm chứng gì |
| :--- | :--- | :--- | :--- |
| **Phần 3** | "Giải thích vì sao kNN không cần train mà vẫn dự đoán được và nhược điểm khi có 10 triệu bản ghi?" | AI giải thích cơ chế instance-based learning (lazy learner) và độ phức tạp tính toán $\mathcal{O}(N \cdot d)$ khi tính khoảng cách với mọi mẫu trong bộ nhớ. | Tự đối chiếu với slide 14, hiểu rõ vì sao khi triển khai thực tế phải dùng FAISS/ANN Vector Database thay vì kNN vét cạn. |
| **Phần 4** | "Tại sao mô hình reasoning như Gemini / OpenAI cần thiết lập max_tokens và reasoning_effort đặc thù khi phân loại?" | AI giải thích rằng reasoning model dùng token output cho chuỗi suy nghĩ ẩn (`thinking`). Nếu giới hạn 3 token mà không tắt reasoning thì model cạn token ngay trong bước nghĩ. | Đã kiểm chứng qua hàm `provider_kwargs` và test mock API thành công. |

### Điều tâm đắc học được hôm nay:
1. **Không phải mô hình càng phức tạp càng tốt:** Deep Learning nhiều tầng không mang lại nhiều giá trị trên biểu diễn từ rời rạc (BoW) của tập dữ liệu nhỏ; sức mạnh thực sự đến từ **biểu diễn đặc trưng (Representation)** chất lượng cao (như Pre-trained Embedding).
2. **Chi phí và độ trễ trong bài toán kỹ thuật thực tế:** LLM rất thông minh và không cần dữ liệu huấn luyện (zero-shot), nhưng trong các bài toán quy mô lớn đòi hỏi thời gian thực (như lọc SMS mạng viễn thông), các giải pháp Embedding + Linear Head nhẹ hoặc mô hình lai (Router) mới là lời giải khả thi về mặt kỹ thuật và kinh tế.
