# SG-ALNS v2 — mã viết lại và chạy lại toàn bộ kịch bản thực nghiệm

Viết lại từ `BAN_CUOI_CUNG/05_Ma_nguon_va_ket_qua/sg-alns`, giữ nguyên thiết kế, tham số và kịch bản của bài
*SG-ALNS: Segment-Level Graph Reinforcement Learning to Configure ALNS for the VRPTW*.

## Cài đặt
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # Python 3.14, phiên bản cố định
```

## Cấu trúc
```
sgalns/core.py      Lõi Numba: trạng thái nghiệm, kiểm tra cửa sổ thời gian O(1) (forward time slack),
                    6 toán tử phá hủy, 3 toán tử sửa, vòng ALNS + SA theo phân đoạn, giai đoạn giảm xe, đặc trưng nút
sgalns/instance.py  Đọc/ghi Solomon & Gehring–Homberger, sinh instance kiểu Solomon, bộ kiểm tra nghiệm độc lập
sgalns/alns.py      Khởi tạo Solomon I1, lớp Engine (một lượt ALNS), baseline ALNS-Uni / ALNS-RW
sgalns/env.py       MDP cấp phân đoạn: hành động (a1, a2, a3), trạng thái đồ thị + toàn cục, phần thưởng, VecEnv
sgalns/policy.py    Chính sách GNN (3×GINE, PyTorch thuần) + 3 đầu chính sách + đầu giá trị
sgalns/ppo.py       PPO (clip, GAE), curriculum, validation so với ALNS-RW, checkpoint
sgalns/methods.py   Chạy một phương pháp trên một instance (uni, rw, rwt:, static:, sched:, sg:, sgs:)
scripts/train.py    Huấn luyện một mô hình (configs/default.yaml + --set khóa=giá_trị)
scripts/evaluate.py Đánh giá song song trên bộ benchmark, CSV có thể chạy tiếp
scripts/tune.py     Tinh chỉnh tuned-static, tuned-RW, headroom tĩnh và headroom động (lịch hai pha)
scripts/verify_bks.py  Đối chiếu bảng BKS với nghiệm CVRPLIB và với mọi lượt chạy
scripts/report.py   Sinh results/REPORT.md (mọi bảng, kèm cột số liệu trong bài) và results/figs/*.png
scripts/probe_state.py  Kiểm tra policy có phản ứng với trạng thái: hỏi policy trên các trạng thái phản thực
scripts/paper/      Sinh mọi số, bảng, hình của bài (make_paper.sh gọi theo thứ tự)
run_all.sh          Toàn bộ kịch bản (chạy tiếp được: bước đã xong sẽ bỏ qua)
run_revision.sh     Thí nghiệm theo phản biện vòng 1 (thời gian tuning, lưới 285 hành động, seed 3–4, BDQ)
run_revision2.sh    Thí nghiệm theo phản biện vòng 2 (seed 3–4, lưới 285, λ = 1 và 25, iteration-level ở T = 200)
make_paper.sh       Sinh lại generated/ từ kết quả và biên dịch bài, bản ẩn danh và phụ lục
```

## Chạy
```bash
nohup caffeinate -i ./run_all.sh > logs/run_all.log 2>&1 &     # ~8 giờ trên Apple M1 8 lõi
.venv/bin/python scripts/report.py                              # có thể chạy bất kỳ lúc nào trên kết quả dở dang
```

## Kịch bản (run_all.sh)
| Bước | Nội dung | Ra |
|---|---|---|
| 0 | Kiểm thử lõi và môi trường | `logs/test_*.log` |
| 1 | Tuned static (30 cấu hình), tuned RW (36), headroom tĩnh, headroom động (100 lịch hai pha) trên 24 instance sinh | `results/tune_*.csv`, `dyn_headroom.csv`, `headroom.csv` |
| 2 | Huấn luyện 13 mô hình, mỗi mô hình 2.400 episode: main (seed 0, 1, 2), 6 ablation, 2 baseline cấp vòng lặp (MLP, GNN – dừng sau 57 phút như bài), SG-ALNS-C, main_ent (entropy cố định + khởi tạo trung tính) | `results/<tên>/last.pt` |
| 3 | E1 Solomon T=5000 (5 seed), ablation, E4 RL cấp vòng lặp (3 seed), E5 so cùng thời gian (T=10000), E3 ngân sách (T=1000, 20000), E2 HG-200, E6 HG-400 | `results/e*.csv` |
| 4 | Đối chiếu BKS, sinh báo cáo | `results/REPORT.md`, `results/figs/` |

## Kiểm chứng tương đương với mã cũ
- Với cùng seed, mọi baseline (uni, rw, rwt, static, sched, rw_fixedq) cho NV/TD **giống hệt từng bit** mã cũ (Solomon, HG-200).
- Nạp mô hình cũ (`main`, `ctx`, `no_a2`, `iter_mlp`) vào mã mới: lượt chạy greedy cho kết quả giống hệt.
- `scripts/report.py` chạy trên CSV cũ tái tạo đúng mọi số trong bài (Bảng 2, ablation, Friedman, ngân sách, HG-200/400…).

## Khác biệt có chủ đích so với mã cũ
- `sgs:` (hành động lấy mẫu) seed lại RNG của torch theo seed của lượt chạy → tái lập được (bản cũ phụ thuộc thứ tự tác vụ trong pool).
- `ppo.max_minutes`: giới hạn thời gian huấn luyện (dùng cho baseline GNN cấp vòng lặp, bài dừng ở 57 phút).
- Headroom tĩnh dùng khung `static:` (cập nhật RW mỗi 50 vòng) thay cho script riêng.
- Bỏ đường dẫn mã chết (`--gen`, các khóa cấu hình không dùng).

## Bài báo (paper/)
Bài dùng mẫu IEEE conference (`IEEEtran.cls`, mẫu `conference_101719`): `main.tex` (có tên tác giả),
`main_anonymous.tex` (ẩn danh), `supplementary.tex` (phụ lục, một cột). Bản Springer LNCS của cùng nội dung được giữ ở
`main_lncs.tex` và `supplementary_lncs.tex`; nội dung chung nằm trong `paper/sections/`.
Mọi con số, bảng và hình trong `paper/` được sinh từ kết quả, không gõ tay:
```bash
RESULTS=$PWD/results_paper/results LOGS=$PWD/results_paper/logs ./make_paper.sh   # đúng bộ kết quả của bài
```
- `results_paper/` là bộ kết quả mà bài đang dùng (mô hình, CSV thô, log). Mã v2 cho kết quả giống hệt từng bit với
  bộ này (xem mục kiểm chứng ở trên); các mô hình mới của vòng phản biện 2 được huấn luyện bằng mã v2 trong `results/`
  và liên kết vào `results_paper/results/`.
- Đối chiếu BKS: chạy `scripts/verify_bks.py` trong thư mục có `data/` và `results/`, ghi log vào `<LOGS>/verify_bks.log`
  (make_paper.sh cảnh báo nếu còn số chưa điền `{?}`).
- Các câu đánh dấu `% RUN-DEPENDENT` trong `paper/sections/*.tex` mô tả hành vi học được; đọc lại sau mỗi lần huấn luyện lại.
- Trước khi nộp: thay `XXXX` trong `paper/sections/04_experimental_setup.tex` bằng đường dẫn repo thật.
