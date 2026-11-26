# HƯỚNG DẪN CHẠY TRÊN MÁY CÁ NHÂN (CPU) & ĐÓNG GÓI GITHUB
**Áp dụng cho:** Thư mục dự án `D:\HyTri` (Windows, Python 3.10+)  
**Mục tiêu:** Đẩy repo sạch mốc Lần 2 cho thầy và chạy phần chuẩn bị/tổng hợp cho thực nghiệm mở rộng tuần tới (195 builds + đa mô hình).

---

## Tổng quan quy trình 3 chặng (Nhìn là hiểu toàn bộ luồng)

```text
[CHẶNG 1 - Máy cá nhân]  Vá 3 file lõi -> Chạy kiểm tra Pilot 45 builds -> Push GitHub gửi thầy
          │
          ▼
[CHẶNG 2 - Máy cá nhân]  Cắt 195 builds (exp_195_builds.json) + Chạy 4 baselines trên 150 Test builds
          │
          ▼  (Upload exp_195_builds.json lên Kaggle — xem file 02_HUONG_DAN_CHAY_KAGGLE_GPU.md)
[CHẶNG 3 - Kaggle GPU ]  Chạy 3 tài khoản song song sinh các file cache JSON (Qwen, DeepSeek, e5-large, Reasoning)
          │
          ▼  (Tải các file cache JSON về bỏ vào D:\HyTri\data\)
[CHẶNG 4 - Máy cá nhân]  Chạy 4 lệnh CPU (~10 giây/lệnh) xuất toàn bộ bảng RQ1, RQ2, RQ3 & Wilcoxon
```

---

## Chuẩn bị chung (Làm 1 lần duy nhất)

1. Giải nén gói **`HyTri_Nhom6KH_CodePack_v2.zip`** trực tiếp vào thư mục gốc **`D:\HyTri`** (chọn ghi đè các file trùng tên).
2. Mở **Command Prompt (CMD)** tại `D:\HyTri` và kích hoạt môi trường ảo:
   ```bat
   cd /d D:\HyTri
   .venv\Scripts\activate
   ```
3. Kiểm tra trong thư mục `D:\HyTri\scripts\` đã có đủ 4 file sau:
   - `apply_hytri_patches.py` (vá lỗi mã nguồn gốc)
   - `export_dataset_split.py` (cắt mẫu 195 builds)
   - `check_cache.py` (kiểm tra chất lượng file cache từ Kaggle)
   - `run_rqs_clean.py` (chạy tính điểm RQ1–RQ3 và kiểm định Wilcoxon)

---

## CHẶNG 1: Đóng gói & Đẩy GitHub mốc Báo cáo Lần 2 gửi thầy (5 phút)

### Bước 1.1: Vá trực tiếp 3 lỗi vào mã nguồn lõi `hytri/`
Chạy lệnh sau để script tự động vá `_call_openai`, thêm trọng số `diff_line = 0.18` và thêm chế độ chuẩn hóa `Min-Max` (có tự sao lưu `.bak`):
```bat
python scripts/apply_hytri_patches.py
```

### Bước 1.2: Chạy xác minh lại đúng số liệu Slide Lần 2
```bat
python scripts/check_cache.py
python scripts/run_rqs_clean.py
```
*Kết quả đạt yêu cầu:* In ra đúng mốc `HyTri+LLM (Default) = 0.567`, `Calibrated = 0.703`, `HyTri-Selective = 0.702`.

### Bước 1.3: Gom tài liệu & kiểm tra dung lượng file cache
1. Tạo thư mục `D:\HyTri\docs\` rồi copy **Slide Báo cáo Lần 2 (bản Final)** và **Chương 5 Word** vào đó.
2. Kiểm tra dung lượng file `llm_cache_qwen.json`:
   ```bat
   dir data\llm_cache_qwen.json
   ```
   - **Nếu nhỏ hơn `50.000.000` bytes (< 50 MB):** Để nguyên file `.gitignore`, đẩy thẳng lên GitHub.
   - **Nếu lớn hơn 50 MB:** Mở `D:\HyTri\.gitignore`, xóa dòng `!data/llm_cache_qwen.json`, sau đó upload riêng file đó lên Google Drive và dán link tải vào `README.md`.

### Bước 1.4: Push lên GitHub
1. Lên GitHub tạo 1 repo mới để chế độ **Public** (ví dụ: `IT5440-Nhom6KH-HyTri-Selective`), **không tích** tạo sẵn README hay `.gitignore`.
2. Nếu thư mục `D:\HyTri` có sẵn thư mục ẩn `.git` cũ (từ lúc tải code gốc), hãy xóa thư mục `.git` đó đi trước.
3. Chạy lần lượt các lệnh sau:
   ```bat
   git init
   git add .
   git status
   ```
   > **Kiểm tra bắt buộc ở `git status`:** Nhìn lướt danh sách file màu xanh — tuyệt đối **không được có** `.venv/` và **không được có** các thư mục dữ liệu gốc `~40 GB` trong `data/`. Nếu sạch rồi mới chạy tiếp:
   ```bat
   git commit -m "Add patched HyTri, Pilot RQ1-RQ3 experiments, Qwen-7B cache and docs"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan-cua-ban>/<ten-repo>.git
   git push -u origin main
   ```
4. **Kiểm tra chéo trước khi gửi thầy:** Nhờ 1 thành viên trong nhóm `git clone` repo vừa tạo về máy khác, chạy `pip install -e .` và `python scripts/run_rqs_clean.py` xem có ra đúng số liệu không.

---

## CHẶNG 2: Chuẩn bị dữ liệu mở rộng 195 builds & chạy 4 Baselines (Tuần sau)

### Bước 2.1: Cắt tập 195 builds tách biệt 0% trùng lặp
Tạo **150 Test builds** (50 Defects4J + 50 Bears + 50 BugSwarm, `seed=42`) và **45 Validation builds** (15/nguồn, `seed=123`):
```bat
python scripts/export_dataset_split.py --n-test 50 --n-val 15 --out data/exp_195_builds.json
```
*Đầu ra:* File **`D:\HyTri\data\exp_195_builds.json`** (khoảng ~18–20 MB). Gửi file này cho các thành viên phụ trách chạy Kaggle ở file `02_HUONG_DAN_CHAY_KAGGLE_GPU.md`.

### Bước 2.2: Chạy lại 4 công cụ nền (không dùng AI) trên đúng 150 Test builds
Chạy lệnh gốc `run_evaluation.py` với cùng `seed=42` và `--num-builds 50` (tương tự cách nhóm đã chạy cho 30 builds trước đây):
```bat
python scripts/run_evaluation.py --num-builds 50 --seed 42
```
*(Lưu ý: Nếu script gốc lưu ra thư mục mặc định với tên khác, hãy đổi tên / copy file CSV kết quả 4 baselines về đúng đường dẫn **`D:\HyTri\results\test_baselines_150.csv`** để Bước 4 tự đọc.)*

---

## CHẶNG 4: Tổng hợp kết quả sau khi tải cache từ Kaggle về (CPU ~10 giây/lệnh)

Sau khi các thành viên chạy xong Kaggle (theo File 02), gom toàn bộ file cache JSON đặt vào thư mục **`D:\HyTri\data\`** đúng tên như bảng dưới, rồi chạy từng lệnh tương ứng:

| Thí nghiệm | File cache cần đặt vào `D:\HyTri\data\` | Lệnh kiểm tra & chạy đánh giá trên CPU | File CSV kết quả xuất ra |
| :--- | :--- | :--- | :--- |
| **1. Mở rộng 195 builds (`Qwen2.5-Coder-7B`)** | `llm_cache_qwen_195.json` | `python scripts/check_cache.py --dataset data/exp_195_builds.json --cache data/llm_cache_qwen_195.json`<br>`python scripts/run_rqs_clean.py --dataset data/exp_195_builds.json --cache data/llm_cache_qwen_195.json --baselines results/test_baselines_150.csv --out-prefix exp195_qwen` | `results/exp195_qwen_rq3_summary.csv` |
| **2. Đa mô hình 195 builds (`DeepSeek-Coder-6.7B`)** | `llm_cache_deepseek_195.json` | `python scripts/check_cache.py --dataset data/exp_195_builds.json --cache data/llm_cache_deepseek_195.json`<br>`python scripts/run_rqs_clean.py --dataset data/exp_195_builds.json --cache data/llm_cache_deepseek_195.json --baselines results/test_baselines_150.csv --out-prefix exp195_deepseek` | `results/exp195_deepseek_rq3_summary.csv` |
| **3. Đổi Embedding (`intfloat/e5-large-v2`)** | `llm_cache_qwen_e5large.json` | `python scripts/run_rqs_clean.py --dataset data/pilot_45_builds.json --cache data/llm_cache_qwen_e5large.json --baselines results/test_baselines_30.csv --out-prefix pilot_e5large` | `results/pilot_e5large_rq3_summary.csv` |
| **4. Đối chứng prompt có `reasoning` (Bears)** | `llm_cache_qwen_bears_reasoning.json` | `python scripts/run_rqs_clean.py --dataset data/pilot_45_builds.json --cache data/llm_cache_qwen_bears_reasoning.json --baselines results/test_baselines_30.csv --only-source Bears-Real --out-prefix bears_reasoning` | `results/bears_reasoning_rq3_summary.csv` |

*(Nếu Thí nghiệm 3 chạy re-embed trên bản `llm_cache_qwen_195.json` thì thay `--dataset data/exp_195_builds.json` và `--baselines results/test_baselines_150.csv` tương ứng.)*

Sau khi chạy xong 4 lệnh trên, copy toàn bộ bảng số liệu in ra trên terminal (hoặc các file `results/*_rq3_summary.csv`) để cập nhật biểu đồ, slide và push bản cập nhật mới nhất lên GitHub.