# Đối chiếu nhận xét và chỉnh sửa

**Bài:** *SG-ALNS: Segment-Level Graph RL for ALNS Configuration in the VRPTW* (tên cũ: *SG-ALNS: Segment-Level Graph Reinforcement Learning to Configure ALNS for the VRPTW*)
**Bản sửa:** mẫu IEEE conference (`main.tex`, 7 trang), bản ẩn danh (`main_anonymous.tex`), phụ lục (`supplementary.tex`).
**Nguồn nhận xét:** `Phan_bien_SG-ALNS.pdf` (vòng 1) và `Nhan_xet_chi_tiet_bai_bao_SG-ALNS.docx` (vòng 2).

Mọi con số trong bài được sinh tự động từ kết quả thô (`results_paper/`) bằng `make_paper.sh`; không có số nào gõ tay.

---

## A. Nhận xét chi tiết (vòng 2)

### 3.1 / 3.2. Định vị tính mới so với DAC và DR-ALNS
- **Sửa:** Contribution 1 viết lại. MDP theo đoạn là một môi trường thử nghiệm có lợi cho RL, không phải một nguyên lý mới. Nó lấy quyết định theo khoảng từ DAC và điều khiển đồng thời toán tử/tham số từ DR-ALNS. Related Work nêu rõ các công trình trước không báo cáo hành động học được, biến thiên giữa seed hay so sánh cùng thời gian.

- **Bổ sung công trình liên quan mới:** Cherrered et al. (ROADEF 2026), Xie & Li (arXiv 2025), Yu et al. (DAC-ALNS, arXiv 2026). Bảng I thêm dòng ALNS-RL. Câu "chỉ DR-ALNS học destroy scale và nhiệt độ" được sửa vì không còn đúng.
- **Đổi tiêu đề** cho khớp với đóng góp chính, tức nghiên cứu đánh giá.

### 3.3. Ablation chưa chứng minh lợi ích của GNN
- **Sửa:** Bài nói thẳng rằng mỗi ablation là một lần huấn luyện hội tụ về một hằng số khác. Bản không dùng đặc trưng động và bản MLP **học đúng hành động của seed 1 và cho kết quả giống hệt**. Vì vậy ablation đo biến thiên giữa seed, không đo giá trị của encoder.

### 3.4 / 3.5. "Không phụ thuộc trạng thái" và phạm vi kết luận
- **Thí nghiệm mới:** kiểm tra trạng thái phản thực (`scripts/probe_state.py`, Bảng V của bài chính). Mỗi policy nhận khoảng 30 000 truy vấn, gồm: đổi ngẫu nhiên đặc trưng tiến trình/nhiệt độ/trì trệ/tỉ lệ thành công; hoán vị hoặc xóa đặc trưng động của nút; thay đồ thị bằng đồ thị của instance khác.
- **Kết quả:**
  - 9 policy bị sụp **không đổi hành động trong bất kỳ truy vấn nào (0,0%)**.
  - 3 policy dùng nhiều hành động thì có phản ứng (4,5–57%). Hai trong số đó kém hơn.
  - Policy còn lại là seed 3: nó học một lịch nhiệt độ theo thời gian (m = 3 chiếm 0% ở 1 000 vòng đầu, 85% ở 1 000 vòng cuối). Lịch này giống lịch hai pha tốt nhất, nhưng **tương đương tuned static về số xe** (TOST).
- **Sửa câu chữ:** kết luận giới hạn ở "trên các instance và ngân sách đã khảo sát". Discussion và Conclusion ghi rõ bài **không** chứng minh rằng RL điều khiển ALNS nói chung là vô hiệu.

### 4.1. Baseline cấu hình tĩnh
- **Chi phí:** phân biệt rõ số cấu hình, số lần chạy, số vòng lặp, CPU-phút đo được và thời gian thực.
  - Tuned static: 2 232 lần chạy, 25 CPU-phút.
  - Tuned ALNS-RW: 2 664 lần chạy, 37 CPU-phút.
  - Một lần huấn luyện: 8 lõi × 21–23 phút.
- **Lưới đầy đủ 285 hành động** (20 592 lần chạy, 302 CPU-phút) chọn Worst/Regret-3, q = 30%, m = 3. Cấu hình này tương đương SG-ALNS về số xe (ΔCNV = −1,0; p = 0,061).
- **Đường cong ngân sách tuning, có tách validation/test** (Bảng S14): chọn trên 12 instance, đánh giá trên 12 instance còn lại. Chất lượng held-out bão hòa ở khoảng 30 cấu hình.
- **Tách tập:** instance tuning (seed 90000–90023), instance huấn luyện và instance validation lấy từ các dòng ngẫu nhiên khác nhau. Benchmark chỉ dùng làm tập test.
- **irace:** chưa dùng; được nêu trong threats to validity.

### 4.2. Nguyên nhân policy collapse
- **Thêm 2 seed** (tổng 5 seed): 2/5 seed dùng ít xe hơn ALNS-RW có ý nghĩa thống kê, 5 seed học 4 hành động khác nhau.
- **Entropy và bias khởi tạo:** đã kiểm tra bằng policy có entropy cố định và prior trung tính.
- **Reward:** huấn luyện lại với λ = 1 và λ = 25 (Bảng S15).
- **Theo dõi xác suất hành động:** thêm vào `sgalns/ppo.py` việc ghi tỉ lệ hành động phổ biến nhất và số hành động khác nhau ở mỗi lượt cập nhật PPO (chỉ ghi log, không đổi kết quả huấn luyện).
- **Kiểm tra trực tiếp phản ứng với trạng thái:** xem mục 3.4.
- **Discussion:** tách kết quả khỏi giả thuyết nguyên nhân. Các yếu tố đã loại trừ: entropy, bias khởi tạo, trọng số reward. Các yếu tố còn lại: không gian hành động rời rạc nhỏ, reward shaping có chiết khấu, và việc biểu diễn một hằng số dễ dàng.

### 4.3. Reward so với mục tiêu phân cấp
- **Kiểm tra f:** trên 12 732 lần chạy, chênh lệch quãng đường lớn nhất giữa hai nghiệm của cùng một instance là 2 569 (< 10⁴). Vậy f giữ đúng thứ tự "số xe trước, quãng đường sau" cho mọi nghiệm, kể cả ở n = 400.
- **Giải thích reward:** reward chỉ là tín hiệu shaping; một xe tương đương 5% quãng đường ban đầu.
- **Độ nhạy λ:** cả λ = 1 lẫn λ = 25 đều sụp về một hành động (420,2 xe mỗi mô hình).
- **γ = 0,99:** câu "return telescoping" sửa thành "undiscounted return".

### 4.4. Kiểm định thống kê
- **Đơn vị kiểm định:** instance; các seed trên cùng instance được lấy trung bình trước.
- **Phép kiểm định:** phân cấp, hai phía. ΔCNV kèm CI bootstrap 95% và Wilcoxon trên NV; ΔTD trên các instance cùng số xe; TOST với ngưỡng 0,1 xe/instance và 0,2% quãng đường (Bảng III).
- **W/T/L:** dùng sign test hai phía, hiệu chỉnh Holm trong Bảng II.
- **Hiệu chỉnh nhiều phép thử ở Bảng III:** thêm cột p đã hiệu chỉnh Holm. Mức giảm số xe so với ALNS-RW chỉ có ý nghĩa khi chưa hiệu chỉnh (p = 0,012; p_Holm = 0,081). Abstract và Kết luận đã ghi rõ lợi thế này không bền vững.
- **Giải thích TOST:** một khác biệt có thể có thật nhưng vẫn nhỏ hơn ngưỡng tương đương 0,1 xe mỗi instance.
- **Thống nhất:** toàn bộ p-value trong văn bản, bảng và hình dùng cùng một quy tắc.

### 4.5. Baseline iteration-level
- **Thí nghiệm mới ở T = 200**, gần horizon huấn luyện của agent (Bảng S16): agent chỉ ngang ALNS-RW (428,3 so với 428,3 xe), còn SG-ALNS dùng 425,0.
- **Kết luận:** agent không học được điều khiển hữu ích ở cả hai chế độ. Bài không còn khẳng định quyết định theo đoạn tốt hơn nói chung.

### 5. Chi tiết phương pháp và tái lập
- **Phụ lục "Implementation Details"** mô tả:
  - chuẩn hóa đặc trưng; cách dựng đồ thị (cung láng giềng và cung tuyến); trạng thái tính trên nghiệm hiện tại;
  - cập nhật roulette wheel khi toán tử không được dùng (trọng số giảm dần về sàn 10⁻³);
  - log-probability và entropy của hành động phân rã (các đầu độc lập có điều kiện);
  - chi tiết PPO và cách tách các dòng instance.
- **Algorithm 1** nằm trong bài chính.
- **Mã nguồn:** có `requirements.txt` với phiên bản cố định, README hướng dẫn tái lập, script sinh mọi số liệu, kết quả thô và mô hình đã huấn luyện.

### 6. Chuyển giao kích thước
- **Phân tích theo nhóm ở n = 400** (Bảng S17): lợi thế của SG-ALNS-C nằm ở C1, C2, R1 và RC1, không ở R2 và RC2.
- **Diễn giải:** bài quy lợi thế này cho phân phối huấn luyện, không cho riêng kích thước n.

### 7. Abstract
- Abstract tách rõ kết quả seed 0 (418,0 xe) với trung bình 5 seed (419,3 xe), và giới hạn phạm vi kết luận.

---

## B. Nhận xét vòng 1 (tóm tắt trạng thái)

| Mục | Trạng thái |
|---|---|
| 2.1 Ẩn danh | Có bản ẩn danh riêng. **Còn lại:** thay `XXXX` bằng link repo thật (`paper/sections/04_experimental_setup.tex`). |
| 2.2 Phụ thuộc phụ lục | Bài chính (8 trang) có Bảng III (kiểm định, cùng thời gian chạy), Bảng IV (tóm tắt sáu phép kiểm tra), Bảng V (kiểm tra trạng thái), Hình 1 (framework) và Algorithm 1. |
| 2.3 / 2.4 Thống kê | Xem A.4.4. |
| 2.5 Thời gian tuning | Thay bằng số đo thật (xem A.4.1). |
| 2.6 Baseline iteration-level | Abstract và Conclusion viết "our re-implementations … trained on short episodes". |
| 3.1–3.6 | Đã xử lý: giải thích lưới, số lần chạy đạt BKS, 5 seed, hình lớn hơn, seed 0 = ALNS-RW tham số hóa lại, số trang. |

---

## C. Việc tác giả cần làm trước khi nộp
1. Thay link `XXXX` (bản ẩn danh: anonymous.4open.science; bản có tên: GitHub).
2. Kiểm tra giới hạn trang của hội thảo. Bài hiện 8 trang IEEE, đúng giới hạn tối đa 8 trang.
3. Chọn bản nộp: `main_anonymous.pdf` nếu phản biện mù đôi, `main.pdf` nếu không.
4. Bổ sung lời cảm ơn hoặc nguồn tài trợ nếu có.


---

## D. Phản biện vòng 3 (`Phan_bien_SG-ALNS.docx`)

| Mục | Mức | Đã làm |
|---|---|---|
| 3.5 Số policy không khớp | Bắt buộc | Mọi policy đã huấn luyện đều được đếm và có trong Hình 3a: **17/20 policy bị sụp** (thêm 2 seed prior trung tính và 1 lần huấn luyện dài). Abstract, Hình 3a, Bảng V và mục V-B thống nhất; 3 ngoại lệ được nêu rõ (seed 3, bản bỏ a¹, prior trung tính seed 0). |
| 3.5 Quỹ đạo tương đối | Bắt buộc | Mục III ghi rõ: huấn luyện T = 2 500, triển khai 5 000; t/T và lịch nhiệt được chuẩn hóa theo T; lợi thế chỉ xuất hiện sau 2 500 vòng lặp (Hình 2a). |
| 3.6 Ngưỡng TOST | Bắt buộc | Lý giải ngưỡng (1 xe trên mỗi 10 instance; lớn hơn độ lệch chuẩn giữa các seed, 0,9 xe) và kiểm tra ở **nửa ngưỡng**: tuned static vẫn tương đương (p = 0,002; lưới 285: p = 0,007), tuned ALNS-RW thì không (p = 0,145). |
| 3.7 Abstract, Bảng IV | Bắt buộc | Abstract rút còn khoảng 165 từ, một thông điệp và ba con số. Bảng tóm tắt sáu phép kiểm tra chuyển lên phần mở đầu (Bảng I). Bản ẩn danh: `main_anonymous.pdf`. |
| 3.1 Oracle theo lớp | Nên làm | **Đã chạy**: chọn hằng số tốt nhất riêng cho từng lớp Solomon (có biết trước kết quả) chỉ tiết kiệm 0,6 xe (417,4 so với 418,0); chọn theo lớp trên instance sinh còn kém hơn (419,4). |
| 3.1 Huấn luyện lâu hơn | Nên làm | **Đã chạy** gấp 5 lần (1 500 lượt cập nhật): policy vẫn sụp (Shaw/Regret-3, q = 30%, m = 0,3; 419,6 xe) và 0% truy vấn phản thực đổi hành động. |
| 3.2 Prior trung tính, 3 seed | Nên làm | **Đã chạy** thêm seed 1 và 2: cả hai sụp về một hành động. Tổng cộng 2/3 seed sụp, seed còn lại dùng 2 hành động nhưng kém hơn ALNS-RW. |
| 3.4 Baseline iteration-level | Nên làm | Bỏ khỏi abstract. Threats to validity ghi rõ đây là bản tái cài đặt, không phải mã gốc DR-ALNS/GRLOS. |
| 3.3 ALNS nền | Tùy chọn | Nêu trong Threats to validity: ALNS nền còn cách BKS khoảng 13 xe; ALNS nền mạnh hơn có thể thay đổi kết luận về headroom. |
