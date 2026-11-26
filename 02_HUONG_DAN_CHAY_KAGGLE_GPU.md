# HƯỚNG DẪN CHẠY THỰC NGHIỆM TRÊN KAGGLE GPU (KÈM FULL CODE)
**Dành cho:** Thành viên chạy GPU miễn phí sinh file cache LLM & Embedding cho Nhóm 6KH.  
**Ưu điểm:** Không cần cài máy mạnh, chỉ cần tạo 1 Notebook trên Kaggle và copy-paste đúng các Cell code bên dưới. Code có tự động lưu checkpoint mỗi 5 builds (mất mạng chạy lại sẽ làm tiếp, không bị chạy lại từ đầu).

---

## 1. Phân chia 3 tài khoản Kaggle chạy song song (để xong trong 1 buổi)

| Tài khoản / Người chạy | Việc cần chạy | File cần upload lên Kaggle | Thời gian ước tính (GPU T4) | File kết quả tải về đặt vào `D:\HyTri\data\` |
| :--- | :--- | :--- | :---: | :--- |
| **Tài khoản 1** | **Luồng A:** `Qwen2.5-Coder-7B` (195 builds) | `exp_195_builds.json` | ~2,5 tiếng | `llm_cache_qwen_195.json` |
| **Tài khoản 2** | **Luồng B:** `DeepSeek-Coder-6.7B` (195 builds) | `exp_195_builds.json` | ~2,5 tiếng | `llm_cache_deepseek_195.json` |
| **Tài khoản 3** | **Luồng C:** `Qwen-7B` giữ `reasoning` trên Bears<br>**Luồng D:** Đổi Embedding sang `e5-large-v2` | `pilot_45_builds.json`<br>`llm_cache_qwen.json` | ~15 phút<br>~1 phút | `llm_cache_qwen_bears_reasoning.json`<br>`llm_cache_qwen_e5large.json` |

---

## 2. Thao tác cài đặt trên giao diện Kaggle (Làm giống nhau cho cả 3 tài khoản)

1. Đăng nhập **kaggle.com** (tài khoản cần xác minh số điện thoại để bật GPU & Internet).
2. Vào **Code** $\rightarrow$ **New Notebook**.
3. Ở thanh bên phải (**Notebook Options / Session options**), bật đúng 2 mục sau:
   - **Accelerator:** Chọn **`GPU T4 x2`** (ưu tiên) hoặc **`GPU P100`**.
   - **Internet:** Bật **`ON`** (để tải mô hình từ HuggingFace).
4. Upload file dữ liệu đầu vào:
   - Ở cột bên phải mục **Input** $\rightarrow$ bấm **Upload** $\rightarrow$ **New Dataset** $\rightarrow$ đặt tên ví dụ `hytri-data` $\rightarrow$ kéo file `.json` vào $\rightarrow$ bấm **Create**.
   - Sau khi tạo xong, di chuột vào file trong cột **Input** và bấm biểu tượng **Copy file path** để lấy đường dẫn chính xác (dạng `/kaggle/input/hytri-data/exp_195_builds.json`).

---

## 3. FULL CODE CHO LUỒNG A, B, C (Sinh cache LLM 7B + bge-large)

Cả 3 luồng (Qwen 195 builds, DeepSeek 195 builds, và Bears có `reasoning`) đều dùng **chung một bộ 5 Cell code dưới đây**, chỉ khác nhau đúng mấy dòng cấu hình ở **Cell 2**.

### Cell 1 — Cài đặt thư viện
```python
!pip install -q -U transformers accelerate sentence-transformers
```

### Cell 2 — Cấu hình thí nghiệm (CHỈ CẦN SỬA CELL NÀY THEO LUỒNG CỦA BẠN)

Chọn **1 trong 3 cấu hình** dưới đây để dán vào Cell 2 (nhớ sửa `DATASET_PATH` đúng với đường dẫn vừa copy ở Bước 2):

- **Nếu chạy Luồng A (Qwen2.5-Coder-7B trên 195 builds):**
  ```python
  DATASET_PATH      = "/kaggle/input/hytri-data/exp_195_builds.json"
  MODEL_NAME        = "Qwen/Qwen2.5-Coder-7B-Instruct"
  EMBEDDING_NAME    = "BAAI/bge-large-en-v1.5"
  INCLUDE_REASONING = False
  ONLY_SOURCE       = None
  BATCH_SIZE        = 16
  CHECKPOINT_EVERY  = 5
  OUT_CACHE_PATH    = "/kaggle/working/llm_cache_qwen_195.json"
  ```

- **Nếu chạy Luồng B (DeepSeek-Coder-6.7B trên 195 builds):**
  ```python
  DATASET_PATH      = "/kaggle/input/hytri-data/exp_195_builds.json"
  MODEL_NAME        = "deepseek-ai/deepseek-coder-6.7b-instruct"
  EMBEDDING_NAME    = "BAAI/bge-large-en-v1.5"
  INCLUDE_REASONING = False
  ONLY_SOURCE       = None
  BATCH_SIZE        = 16
  CHECKPOINT_EVERY  = 5
  OUT_CACHE_PATH    = "/kaggle/working/llm_cache_deepseek_195.json"
  ```

- **Nếu chạy Luồng C (Đối chứng prompt có `reasoning` trên tập Bears):**
  ```python
  DATASET_PATH      = "/kaggle/input/hytri-data/pilot_45_builds.json"
  MODEL_NAME        = "Qwen/Qwen2.5-Coder-7B-Instruct"
  EMBEDDING_NAME    = "BAAI/bge-large-en-v1.5"
  INCLUDE_REASONING = True
  ONLY_SOURCE       = "Bears-Real"
  BATCH_SIZE        = 8
  CHECKPOINT_EVERY  = 5
  OUT_CACHE_PATH    = "/kaggle/working/llm_cache_qwen_bears_reasoning.json"
  ```

### Cell 3 — Nạp mô hình LLM và Embedding lên GPU
```python
import json, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | GPUs: {torch.cuda.device_count()} | Model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

embedder = SentenceTransformer(EMBEDDING_NAME, device=device)
print("Nạp mô hình thành công!")
```

### Cell 4 — Các hàm tạo prompt, chạy batch song song và trích JSON
```python
def build_norm_key(fl: dict) -> str:
    return f"{fl.get('exception_type')}|{fl.get('exception_message','')[:300]}|{fl.get('stack_trace','')[:800]}"

def extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        raw = m.group(0)
        try:
            return json.loads(raw)
        except Exception:
            cleaned = re.sub(r",\s*\}", "}", raw)
            try:
                return json.loads(cleaned)
            except Exception:
                pass
    return {}

def format_chat(system_msg: str, user_msg: str) -> str:
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"

@torch.inference_mode()
def generate_batch(prompts: list[str], max_new_tokens: int) -> list[str]:
    outputs = []
    for i in range(0, len(prompts), BATCH_SIZE):
        chunk = prompts[i : i + BATCH_SIZE]
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs

def make_norm_prompt(fl: dict) -> str:
    sys_p = "You normalize CI test failures into concise JSON. Return ONLY valid JSON."
    usr_p = (
        f"Test: {fl.get('test_name','')}\n"
        f"Exception: {fl.get('exception_type','Unknown')}: {(fl.get('exception_message') or '')[:300]}\n"
        f"Stack trace:\n{(fl.get('stack_trace') or '')[:600]}\n\n"
        'Return JSON: {"category": "<short type>", "normalized_signature": "<Exception>@<Class.method> -> <concise cause>"}'
    )
    return format_chat(sys_p, usr_p)

def pick_diff_snippet(fl: dict, changed_files: list[str], file_diffs: dict) -> str:
    st = (fl.get("stack_trace") or "").lower()
    picked = []
    for cf in changed_files:
        base = cf.split("/")[-1].replace(".java", "").replace(".py", "").lower()
        if base and base in st and cf in file_diffs:
            picked.append((cf, file_diffs[cf]))
    if not picked:
        for cf in changed_files[:3]:
            if cf in file_diffs:
                picked.append((cf, file_diffs[cf]))
    parts = []
    for cf, diff in picked[:2]:
        lines = [ln for ln in (diff or "").splitlines()[:16]]
        parts.append(f"File: {cf}\n" + "\n".join(lines))
    return "\n\n".join(parts)[:1200]

def make_hyp_prompt(fl: dict, changed_files: list[str], file_diffs: dict, include_reasoning: bool) -> str:
    sys_p = "You diagnose CI test failures given changed files and diff. Return ONLY valid JSON."
    files_str = "\n".join(f"- {f}" for f in changed_files[:15])
    diff_str = pick_diff_snippet(fl, changed_files, file_diffs)
    schema = (
        '{"suspected_change": "<exact path from Changed files or null>", '
        '"mechanism": "<concise root cause mechanism>", "confidence": 0.0-1.0, '
        '"reasoning": "<brief step-by-step reasoning>"}'
        if include_reasoning
        else '{"suspected_change": "<exact path from Changed files or null>", '
             '"mechanism": "<concise root cause mechanism>", "confidence": 0.0-1.0}'
    )
    usr_p = (
        f"Changed files:\n{files_str}\n\n"
        f"Relevant diff:\n{diff_str}\n\n"
        f"Failure:\nTest: {fl.get('test_name','')}\n"
        f"Exception: {fl.get('exception_type','Unknown')}: {(fl.get('exception_message') or '')[:250]}\n"
        f"Stack:\n{(fl.get('stack_trace') or '')[:500]}\n\n"
        f"Return JSON: {schema}"
    )
    return format_chat(sys_p, usr_p)
```

### Cell 5 — Chạy vòng lặp toàn bộ build & tự lưu checkpoint
```python
builds = json.loads(Path(DATASET_PATH).read_text(encoding="utf-8"))
if ONLY_SOURCE:
    builds = [b for b in builds if b.get("source") == ONLY_SOURCE]

if Path(OUT_CACHE_PATH).exists():
    cache = json.loads(Path(OUT_CACHE_PATH).read_text(encoding="utf-8"))
    print(f"Đang chạy tiếp từ checkpoint: đã xong {len(cache.get('hypotheses', {}))} builds.")
else:
    cache = {
        "meta": {
            "model_name": MODEL_NAME,
            "embedding_name": EMBEDDING_NAME,
            "include_reasoning": INCLUDE_REASONING,
            "only_source": ONLY_SOURCE,
            "dataset_path": DATASET_PATH,
        },
        "normalizations": {},
        "hypotheses": {},
        "embeddings": {},
        "build_timings": {},
    }

norm_max_tok = 96
hyp_max_tok = 220 if INCLUDE_REASONING else 110
t_all = time.perf_counter()
done_count = 0

for idx, b in enumerate(builds, 1):
    bid = b["build_id"]
    if bid in cache["hypotheses"]:
        continue

    t0 = time.perf_counter()
    failures = b["failures"]
    changed_files = b.get("combined_changed_files") or []
    file_diffs = b.get("combined_file_diffs") or {}

    # 1. Normalization (khử trùng lặp giữa các build)
    missing_k, missing_p, missing_f, seen_k = [], [], [], set()
    for fl in failures:
        nk = build_norm_key(fl)
        if nk not in cache["normalizations"] and nk not in seen_k:
            seen_k.add(nk)
            missing_k.append(nk)
            missing_p.append(make_norm_prompt(fl))
            missing_f.append(fl)

    if missing_p:
        for nk, fl, raw in zip(missing_k, missing_f, generate_batch(missing_p, norm_max_tok)):
            js = extract_json(raw)
            sig = js.get("normalized_signature")
            if not sig or not isinstance(sig, str):
                sig = f"{fl.get('exception_type','Unknown')}@unknown -> {(fl.get('exception_message') or '')[:50]}"
            cache["normalizations"][nk] = {
                "category": str(js.get("category") or "unknown")[:60],
                "normalized_signature": sig[:240],
            }

    # 2. Hypotheses (khử trùng lặp trong cùng build)
    hyp_dedup, u_keys, u_prompts, u_fls = {}, [], [], []
    for fl in failures:
        hk = f"{fl.get('exception_type')}|{(fl.get('exception_message') or '')[:250]}|{(fl.get('stack_trace') or '')[:500]}"
        if hk not in hyp_dedup:
            hyp_dedup[hk] = None
            u_keys.append(hk)
            u_prompts.append(make_hyp_prompt(fl, changed_files, file_diffs, INCLUDE_REASONING))
            u_fls.append(fl)

    if changed_files and u_prompts:
        for hk, fl, raw in zip(u_keys, u_fls, generate_batch(u_prompts, hyp_max_tok)):
            js = extract_json(raw)
            sc = js.get("suspected_change")
            if isinstance(sc, str) and sc.strip().lower() in ("null", "none", ""):
                sc = None
            mech = js.get("mechanism")
            if not mech or not isinstance(mech, str):
                hyp_dedup[hk] = {
                    "suspected_change": sc,
                    "mechanism": f"Failure in {fl.get('exception_type','test')}",
                    "confidence": 0.1,
                    "category": "unknown",
                    "reasoning": "",
                }
            else:
                try:
                    conf = float(js.get("confidence", 0.5))
                except Exception:
                    conf = 0.5
                hyp_dedup[hk] = {
                    "suspected_change": sc,
                    "mechanism": mech[:240],
                    "confidence": max(0.0, min(1.0, conf)),
                    "category": "llm",
                    "reasoning": str(js.get("reasoning") or "")[:300],
                }
    else:
        for hk, fl in zip(u_keys, u_fls):
            hyp_dedup[hk] = {
                "suspected_change": None,
                "mechanism": f"Failure in {fl.get('exception_type','test')}",
                "confidence": 0.1,
                "category": "unknown",
                "reasoning": "",
            }

    # 3. Embeddings
    build_hyps, to_embed = [], set()
    for fl in failures:
        sig = cache["normalizations"][build_norm_key(fl)]["normalized_signature"]
        if sig not in cache["embeddings"]:
            to_embed.add(sig)
        hk = f"{fl.get('exception_type')}|{(fl.get('exception_message') or '')[:250]}|{(fl.get('stack_trace') or '')[:500]}"
        hd = hyp_dedup[hk]
        if hd["mechanism"] not in cache["embeddings"]:
            to_embed.add(hd["mechanism"])
        build_hyps.append({"failure_id": fl["failure_id"], **hd})

    if to_embed:
        t_list = list(to_embed)
        vecs = embedder.encode(t_list, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        for t_str, v in zip(t_list, vecs):
            cache["embeddings"][t_str] = [round(float(x), 5) for x in v]

    elapsed = time.perf_counter() - t0
    cache["hypotheses"][bid] = build_hyps
    cache["build_timings"][bid] = {"llm_total_sec": round(elapsed, 3), "num_failures": len(failures)}
    done_count += 1
    print(f"[{idx:3d}/{len(builds)}] {bid:<35} | failures={len(failures):3d} | {elapsed:6.2f}s")

    if done_count % CHECKPOINT_EVERY == 0:
        Path(OUT_CACHE_PATH).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"  -> Đã lưu checkpoint ({len(cache['hypotheses'])} builds)")

Path(OUT_CACHE_PATH).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
size_mb = Path(OUT_CACHE_PATH).stat().st_size / (1024 * 1024)
print(f"\nHOÀN THÀNH sau {(time.perf_counter()-t_all)/60:.1f} phút -> {OUT_CACHE_PATH} ({size_mb:.2f} MB)")
```

---

## 4. FULL CODE CHO LUỒNG D (Đổi Embedding sang `intfloat/e5-large-v2` – Chỉ chạy ~1 phút)

Tạo 1 Notebook mới (hoặc chạy tiếp sau Luồng C) để thay riêng vector nhúng trong file cache sẵn có mà **không cần chạy lại LLM 7B**.

### Cell 1 — Cài thư viện
```python
!pip install -q -U sentence-transformers
```

### Cell 2 — Tính lại toàn bộ vector nhúng bằng `e5-large-v2`
```python
import json, time
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

IN_CACHE_PATH       = "/kaggle/input/hytri-data/llm_cache_qwen.json"   # Có thể thay bằng llm_cache_qwen_195.json
NEW_EMBEDDING_MODEL = "intfloat/e5-large-v2"
OUT_CACHE_PATH      = "/kaggle/working/llm_cache_qwen_e5large.json"

t0 = time.perf_counter()
cache = json.loads(Path(IN_CACHE_PATH).read_text(encoding="utf-8"))
texts = list(cache["embeddings"].keys())
print(f"Đang nhúng lại {len(texts)} chuỗi duy nhất bằng {NEW_EMBEDDING_MODEL}...")

# Mô hình họ e5 khuyến nghị thêm tiền tố "passage: " khi mã hóa
inputs = [f"passage: {t}" for t in texts] if "e5-" in NEW_EMBEDDING_MODEL.lower() else texts

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(NEW_EMBEDDING_MODEL, device=device)
vecs = embedder.encode(inputs, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

cache["embeddings"] = {t: [round(float(x), 5) for x in v] for t, v in zip(texts, vecs)}
cache.setdefault("meta", {})["embedding_name"] = NEW_EMBEDDING_MODEL

Path(OUT_CACHE_PATH).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
size_mb = Path(OUT_CACHE_PATH).stat().st_size / (1024 * 1024)
print(f"Xong trong {time.perf_counter()-t0:.1f}s -> {OUT_CACHE_PATH} ({size_mb:.2f} MB)")
```

---

## 5. Tải file kết quả về & Xử lý 3 sự cố hay gặp trên Kaggle

1. **Cách tải file kết quả về máy:**
   - Sau khi dòng `HOÀN THÀNH` hiện ra, nhìn sang cột bên phải mục **Output** (`/kaggle/working/`) $\rightarrow$ bấm dấu ba chấm bên cạnh file `.json` $\rightarrow$ chọn **Download**.
   - Ngoài ra, nên bấm **File $\rightarrow$ Download Notebook** (`.ipynb`) để lưu vào thư mục `D:\HyTri\notebooks\` đẩy lên GitHub làm bằng chứng.
2. **Nếu gặp lỗi `CUDA Out of Memory (OOM)`:**
   - Giảm `BATCH_SIZE` ở Cell 2 từ `16` xuống `8` (hoặc `4`), chọn **Restart Session** rồi chạy lại — code sẽ tự động đọc checkpoint chạy tiếp từ build bị dừng.
3. **Nếu muốn chạy ngầm không cần giữ tab trình duyệt mở:**
   - Sau khi chạy thử 1–2 build thấy không lỗi, bấm **Save Version** (góc phải trên) $\rightarrow$ chọn **Save & Run All (Commit)** $\rightarrow$ có thể tắt máy đi ngủ, 2,5 tiếng sau quay lại tab **Output** của Version đó tải file `.json` về.
4. **Nếu gặp lỗi `FileNotFoundError` ở Cell 5:**
   - Kiểm tra lại đường dẫn `DATASET_PATH` hoặc `IN_CACHE_PATH` ở Cell 2 bằng cách bấm **Copy file path** trực tiếp trên file ở cột **Input** bên phải.