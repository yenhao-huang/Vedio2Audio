# 模型選擇使用指南

本指南說明如何在 Audio2Video 專案中選擇和使用不同的影片生成模型。

## 📋 可用模型

### Wan2.1-T2V-1.3B (`wan_2_1`)
- **HuggingFace ID**: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- **參數量**: 1.3B
- **記憶體需求**: ~8GB VRAM
- **量化支援**: ✅ 完整支援 4-bit 量化
- **穩定性**: ✅ 穩定版本
- **推薦用途**: 一般使用、生產環境

### Wan2.2-TI2V-5B (`wan_2_2`)
- **HuggingFace ID**: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- **參數量**: 5B
- **記憶體需求**: ~16GB+ VRAM
- **量化支援**: ⚠️ 實驗性（不建議）
- **穩定性**: ⚠️ 實驗性版本
- **推薦用途**: 研究、實驗、高品質需求

---

## 🎯 使用方法

### 方法 1: 透過配置檔（推薦）

編輯 `config/settings.yaml`:

```yaml
video:
  # 選擇模型: "wan_2_1" 或 "wan_2_2"
  model_key: "wan_2_1"  # 改成 "wan_2_2" 切換模型
  quantization: false    # wan_2_1 支援量化
```

然後正常執行應用程式：

```bash
# Web 介面
streamlit run app.py

# 命令列工具
python utils/text2vedio.py --text_file your_text.json
```

---

### 方法 2: 命令列參數

#### 主要命令列工具 (`utils/text2vedio.py`)

```bash
# 使用 Wan2.1 (預設)
python utils/text2vedio.py \
  --text_file results/1000_transcription.json \
  --output_dir results/ \
  --duration 3.0 \
  --fps 16

# 使用 Wan2.2
python utils/text2vedio.py \
  --model_name wan_2_2 \
  --text_file results/1000_transcription.json \
  --duration 3.0

# 使用 Wan2.1 + 量化
python utils/text2vedio.py \
  --model_name wan_2_1 \
  --quantize \
  --text_file results/1000_transcription.json
```

**可用參數：**
- `--model_name`: 選擇模型 (`wan_2_1` 或 `wan_2_2`)，預設 `wan_2_1`
- `--quantize`: 啟用 4-bit 量化（僅 `wan_2_1` 支援）
- `--text_file`: 輸入文字檔案路徑
- `--output_dir`: 輸出目錄
- `--duration`: 影片時長（秒）
- `--fps`: 每秒幀數
- `--negative_prompt`: 負面提示詞（可選）

---

#### 快速測試腳本 (`experiments/quick_test.py`)

```bash
# 使用 Wan2.1 測試
python experiments/quick_test.py

# 使用 Wan2.2 測試
python experiments/quick_test.py --model_name wan_2_2

# 使用 Wan2.1 + 量化測試
python experiments/quick_test.py --model_name wan_2_1 --quantize
```

---

#### 完整實驗腳本 (`experiments/test_video_generation.py`)

```bash
# 使用 Wan2.1 執行完整實驗
python experiments/test_video_generation.py

# 使用 Wan2.2 執行完整實驗
python experiments/test_video_generation.py --model_name wan_2_2

# 使用 Wan2.1 + 量化執行完整實驗
python experiments/test_video_generation.py --model_name wan_2_1 --quantize
```

---

### 方法 3: 直接在程式碼中使用

```python
from utils.text2vedio import Text2Vedio

# 使用 Wan2.1 (預設)
t2v = Text2Vedio(model_name="wan_2_1", quantized=False)

# 使用 Wan2.2
t2v = Text2Vedio(model_name="wan_2_2", quantized=False)

# 使用 Wan2.1 + 量化
t2v = Text2Vedio(model_name="wan_2_1", quantized=True)

# 生成影片
frames = t2v.text_to_video(
    text="A red car driving through city streets",
    duration=3.0,
    fps=16
)

# 儲存影片
t2v.save_video(frames, "output.mp4", fps=16)
```

---

## 📊 模型比較

| 特性 | Wan2.1-T2V-1.3B | Wan2.2-TI2V-5B |
|------|----------------|----------------|
| **模型大小** | 1.3B 參數 | 5B 參數 |
| **VRAM 需求** | ~8GB | ~16GB+ |
| **生成速度** | 較快 | 較慢 |
| **品質** | 良好 | 可能更好（實驗性） |
| **量化支援** | ✅ 完整 | ⚠️ 實驗性 |
| **穩定性** | ✅ 穩定 | ⚠️ 實驗性 |
| **推薦場景** | 一般使用、生產 | 研究、高品質需求 |

---

## ⚙️ 量化說明

### 什麼是量化？
量化將模型權重從 16-bit 轉換為 4-bit，可以：
- ✅ 減少 ~50% GPU 記憶體使用
- ✅ 允許在較小的 GPU 上運行
- ⚠️ 可能輕微影響品質

### 何時使用量化？
- GPU VRAM 不足（< 16GB）
- 需要節省記憶體
- 可以接受些微品質下降

### 如何啟用量化？

**配置檔方式：**
```yaml
video:
  model_key: "wan_2_1"
  quantization: true  # 啟用量化
```

**命令列方式：**
```bash
python utils/text2vedio.py --model_name wan_2_1 --quantize
```

**程式碼方式：**
```python
t2v = Text2Vedio(model_name="wan_2_1", quantized=True)
```

---

## 🔍 常見問題

### Q1: 如何查看當前使用的模型？
執行時會在控制台輸出：
```
🚀 Initializing Text2Vedio pipeline...
✅ Using GPU: NVIDIA RTX 4090
📋 Selected model: wan_2_1
```

### Q2: Wan2.2 可以使用量化嗎？
技術上可以，但不建議。系統會自動顯示警告並降級到全精度模式。

### Q3: 如何切換模型而不重啟應用程式？
目前需要重啟應用程式。模型在首次載入時會被快取。

### Q4: 兩個模型可以同時載入嗎？
不建議。這會消耗大量 GPU 記憶體。

### Q5: 如何確認模型下載完成？
首次使用會自動從 HuggingFace 下載，檔案會快取在：
```
~/.cache/huggingface/hub/
```

---

## 📝 範例工作流程

### 範例 1: 測試新模型

```bash
# 1. 先用快速測試驗證
python experiments/quick_test.py --model_name wan_2_2

# 2. 如果成功，執行完整實驗
python experiments/test_video_generation.py --model_name wan_2_2

# 3. 查看結果報告
cat experiments/results/exp_*/REPORT.md
```

### 範例 2: 記憶體不足時使用量化

```bash
# 如果遇到 CUDA out of memory
python utils/text2vedio.py \
  --model_name wan_2_1 \
  --quantize \
  --text_file your_text.json \
  --duration 3.0
```

### 範例 3: 生產環境配置

編輯 `config/settings.yaml`:
```yaml
video:
  model_key: "wan_2_1"      # 使用穩定版本
  quantization: false        # 全精度以獲得最佳品質
  default_duration: 3.0
  default_fps: 16
```

---

## 🚀 效能優化建議

### 針對 Wan2.1
- ✅ 使用量化可在 8GB GPU 上運行
- ✅ FPS=16 是速度與品質的最佳平衡
- ✅ 適合批次處理

### 針對 Wan2.2
- ⚠️ 需要 16GB+ VRAM
- ⚠️ 生成時間可能更長
- ⚠️ 建議先用小時長測試（3-5秒）
- ✅ 可能提供更好的品質

---

## 📚 相關文件

- [README.md](../README.md) - 專案總覽
- [experiments/README.md](../experiments/README.md) - 實驗系統詳細說明
- [experiments/EXPERIMENT_SUMMARY.md](../experiments/EXPERIMENT_SUMMARY.md) - 實驗快速指南

---

**版本**: 1.0
**最後更新**: 2025-11-01
