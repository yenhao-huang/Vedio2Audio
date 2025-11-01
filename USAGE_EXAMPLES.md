# Audio2Video 使用範例

快速參考指南，展示所有常見使用場景的命令。

## 🎬 Web 介面

```bash
# 啟動 Web 應用程式（使用配置檔中的模型設定）
streamlit run app.py
```

預設使用 `config/settings.yaml` 中的 `model_key` 設定。

---

## 🖥️ 命令列工具

### 基本使用

```bash
# 使用預設模型 (Wan2.1) 生成影片
python utils/text2vedio.py \
  --text_file results/transcription.json \
  --output_dir results/ \
  --duration 3.0 \
  --fps 16
```

### 選擇不同模型

```bash
# 使用 Wan2.1 (1.3B, 穩定版)
python utils/text2vedio.py \
  --model_name wan_2_1 \
  --text_file results/transcription.json \
  --duration 3.0

# 使用 Wan2.2 (5B, 實驗版)
python utils/text2vedio.py \
  --model_name wan_2_2 \
  --text_file results/transcription.json \
  --duration 3.0
```

### 使用量化節省記憶體

```bash
# Wan2.1 + 4-bit 量化 (節省 ~50% VRAM)
python utils/text2vedio.py \
  --model_name wan_2_1 \
  --quantize \
  --text_file results/transcription.json \
  --duration 5.0 \
  --fps 16
```

### 調整影片參數

```bash
# 生成更長的影片
python utils/text2vedio.py \
  --text_file results/transcription.json \
  --duration 10.0 \
  --fps 16

# 使用更高的 FPS (更流暢，但更慢)
python utils/text2vedio.py \
  --text_file results/transcription.json \
  --duration 3.0 \
  --fps 24

# 使用負面提示詞
python utils/text2vedio.py \
  --text_file results/transcription.json \
  --duration 3.0 \
  --negative_prompt "blurry, low quality, distorted"
```

### 完整範例

```bash
# 最完整的參數組合
python utils/text2vedio.py \
  --model_name wan_2_1 \
  --quantize \
  --text_file data/my_prompt.txt \
  --output_dir results/my_videos/ \
  --duration 5.0 \
  --fps 16 \
  --negative_prompt "worst quality, blurry, distorted"
```

---

## 🧪 實驗與測試

### 快速測試 (生成 1 個影片)

```bash
# 使用預設模型
python experiments/quick_test.py

# 使用 Wan2.2
python experiments/quick_test.py --model_name wan_2_2

# 使用量化
python experiments/quick_test.py --model_name wan_2_1 --quantize
```

### 完整實驗 (生成 18 個影片)

```bash
# 使用 Wan2.1 執行完整測試套件
python experiments/test_video_generation.py

# 使用 Wan2.2 執行完整測試套件
python experiments/test_video_generation.py --model_name wan_2_2

# 使用 Wan2.1 + 量化
python experiments/test_video_generation.py --model_name wan_2_1 --quantize
```

### 查看實驗結果

```bash
# 查看最新實驗報告
cat experiments/results/exp_*/REPORT.md | less

# 查看完整 JSON 結果
cat experiments/results/exp_*/full_results.json | jq '.'

# 列出所有生成的影片
ls -lh experiments/results/exp_*/videos/
```

---

## ⚙️ 修改配置檔

編輯 `config/settings.yaml`:

```yaml
# 切換到 Wan2.2
video:
  model_key: "wan_2_2"  # 從 "wan_2_1" 改為 "wan_2_2"
  quantization: false
  default_duration: 3.0
  default_fps: 16
```

然後執行任何命令（會自動使用新設定）：

```bash
streamlit run app.py
# 或
python utils/text2vedio.py --text_file your_file.json
```

---

## 🐍 Python 程式碼範例

### 基本使用

```python
from utils.text2vedio import Text2Vedio

# 初始化模型
t2v = Text2Vedio(model_name="wan_2_1", quantized=False)

# 生成影片
frames = t2v.text_to_video(
    text="A red sports car driving through a neon city at night",
    duration=3.0,
    fps=16
)

# 儲存影片
t2v.save_video(frames, "output.mp4", fps=16)

# 查看使用統計
stats = t2v.get_usage_statistics()
print(f"Generation time: {stats['total_inference_time']:.2f}s")
print(f"Peak memory: {stats['peak_memory_usage']:.2f} GB")
```

### 使用不同模型

```python
# Wan2.1 (穩定版)
t2v_21 = Text2Vedio(model_name="wan_2_1", quantized=False)

# Wan2.2 (實驗版)
t2v_22 = Text2Vedio(model_name="wan_2_2", quantized=False)

# Wan2.1 + 量化
t2v_21_quant = Text2Vedio(model_name="wan_2_1", quantized=True)
```

### 批次處理

```python
from utils.text2vedio import Text2Vedio
import json

# 初始化一次
t2v = Text2Vedio(model_name="wan_2_1", quantized=True)

# 讀取多個提示詞
prompts = [
    "A car driving in the city",
    "A bird flying in the forest",
    "An astronaut floating in space"
]

# 批次生成
for i, prompt in enumerate(prompts):
    frames = t2v.text_to_video(prompt, duration=3.0, fps=16)
    t2v.save_video(frames, f"video_{i}.mp4", fps=16)
    print(f"Generated video {i+1}/{len(prompts)}")
```

### 從 JSON 讀取轉錄結果

```python
from utils.text2vedio import Text2Vedio

t2v = Text2Vedio(model_name="wan_2_1")

# 讀取音頻轉錄結果
text = t2v.read_text("results/transcription.json")

# 生成影片
frames = t2v.text_to_video(text, duration=5.0, fps=16)
t2v.save_video(frames, "transcription_video.mp4", fps=16)
```

### 使用負面提示詞

```python
from utils.text2vedio import Text2Vedio

t2v = Text2Vedio(model_name="wan_2_1")

frames = t2v.text_to_video(
    text="A beautiful sunset over the ocean",
    duration=3.0,
    fps=16,
    negative_prompt="blurry, low quality, distorted, grainy"
)

t2v.save_video(frames, "sunset.mp4", fps=16)
```

---

## 🔍 查看幫助訊息

```bash
# 主要工具
python utils/text2vedio.py --help

# 快速測試
python experiments/quick_test.py --help

# 完整實驗
python experiments/test_video_generation.py --help
```

---

## 📊 效能比較範例

### 比較兩個模型的生成時間

```bash
# 測試 Wan2.1
time python experiments/quick_test.py --model_name wan_2_1

# 測試 Wan2.2
time python experiments/quick_test.py --model_name wan_2_2

# 測試 Wan2.1 + 量化
time python experiments/quick_test.py --model_name wan_2_1 --quantize
```

---

## 🚨 故障排除範例

### CUDA out of memory

```bash
# 解決方案 1: 使用量化
python utils/text2vedio.py \
  --model_name wan_2_1 \
  --quantize \
  --text_file your_file.json

# 解決方案 2: 降低 FPS
python utils/text2vedio.py \
  --text_file your_file.json \
  --fps 8 \
  --duration 3.0

# 解決方案 3: 縮短時長
python utils/text2vedio.py \
  --text_file your_file.json \
  --duration 2.0
```

### 清理 GPU 快取

```bash
# Python 方式
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

---

## 📖 相關文件

- [README.md](README.md) - 專案總覽
- [docs/MODEL_SELECTION_GUIDE.md](docs/MODEL_SELECTION_GUIDE.md) - 模型選擇詳細指南
- [experiments/README.md](experiments/README.md) - 實驗系統說明

---

**版本**: 1.0
**最後更新**: 2025-11-01
