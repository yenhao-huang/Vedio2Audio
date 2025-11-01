# Audio2Video Streamlit Web Application

一個基於 Streamlit 的語音轉影片 Web 應用程式，結合了 Whisper 語音辨識和 Wan2.1 文字轉影片模型。

## 功能特色

✨ **語音輸入**
- 🎤 即時錄音功能
- 📤 上傳音檔（支援 MP3, WAV, M4A, FLAC）
- 🔊 音檔預聽播放

✨ **語音轉錄**
- 🤖 使用 OpenAI Whisper Large v3 模型
- 📝 可編輯轉錄文字
- 🕐 時間戳記分段顯示
- 💾 下載 JSON 格式轉錄結果

✨ **影片生成**
- 🎬 基於 Wan2.1-T2V-1.3B 擴散模型
- ⚙️ 可調整影片時長和 FPS
- 🔧 支援 4-bit 量化（節省 GPU 記憶體）
- 📊 即時效能監控儀表板

✨ **使用者介面**
- 🖥️ 直覺式三欄布局
- 📈 即時進度顯示
- 📥 一鍵下載影片和轉錄檔
- 💻 GPU/CPU 自動偵測

## 系統需求

### 硬體需求

**最低配置：**
- CPU: 4 核心以上
- RAM: 16GB
- GPU: NVIDIA GPU with 8GB VRAM（啟用量化）
- 儲存空間: 30GB（模型 + 資料）

**建議配置：**
- CPU: 8 核心以上
- RAM: 32GB
- GPU: NVIDIA GPU with 16GB+ VRAM
- 儲存空間: 50GB+

### 軟體需求

- Python 3.8+
- CUDA 11.8+ (如使用 GPU)
- pip 或 conda

## 安裝步驟

### 1. 安裝依賴套件

```bash
# 安裝所有必要套件
pip install -r requirements.txt
```

### 2. 驗證安裝

```bash
# 檢查 PyTorch CUDA 支援
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 啟動應用程式

### 基本啟動

```bash
streamlit run app.py
```

### 自訂埠號

```bash
streamlit run app.py --server.port 8501
```

### 允許外部連線

```bash
streamlit run app.py --server.address 0.0.0.0
```

應用程式預設會在 `http://localhost:8501` 啟動。

## 使用指南

### 步驟 1: 輸入音檔

**方式 A - 上傳檔案：**
1. 選擇「Upload Audio File」
2. 點擊上傳按鈕選擇音檔
3. 支援格式：MP3, WAV, M4A, FLAC

**方式 B - 即時錄音：**
1. 選擇「Record Audio」
2. 點擊麥克風圖示開始錄音
3. 再次點擊停止錄音
4. 播放預聽確認

### 步驟 2: 轉錄音檔

1. 點擊「🚀 Start Transcription」按鈕
2. 等待轉錄完成（進度條會顯示狀態）
3. 檢視轉錄結果
4. 可選：編輯文字內容
5. 可選：展開查看時間戳記分段
6. 可選：下載 JSON 格式轉錄檔

### 步驟 3: 配置影片參數

**基本參數：**
- **Duration**: 影片長度（1-10 秒）
- **FPS**: 每秒幀數（8-30）

**進階選項：**
- **4-bit Quantization**: 啟用可減少約 50% GPU 記憶體使用

> 💡 **提示：** 幀數 = 時長 × FPS。例如：3 秒 × 16 FPS = 48 幀

### 步驟 4: 生成影片

1. 點擊「🎬 Generate Video」按鈕
2. 等待生成完成（通常需要數分鐘）
3. 預覽生成的影片
4. 查看效能指標（CPU/GPU 使用率、記憶體等）
5. 下載影片檔案

## 配置設定

編輯 `config/settings.yaml` 可自訂應用程式設定：

```yaml
# 音檔設定
audio:
  max_duration: 300  # 最大音檔長度（秒）

# 轉錄設定
transcription:
  model_name: "openai/whisper-large-v3"
  device: "auto"  # auto, cuda, cpu

# 影片設定
video:
  default_duration: 3.0
  default_fps: 16
  quantization: false  # 預設啟用量化

# 檔案管理
files:
  temp_dir: "/tmp/audio2vedio"
  output_dir: "results"
```

## 目錄結構

```
audio2vedio/
├── app.py                      # Streamlit 主程式
├── requirements.txt            # Python 依賴套件
├── config/
│   └── settings.yaml          # 應用程式設定
├── backend/
│   └── pipeline.py            # 後端整合模組
├── components/
│   ├── audio_input.py         # 音檔輸入元件
│   ├── transcription.py       # 轉錄顯示元件
│   └── video_generator.py     # 影片生成元件
├── utils/
│   ├── audio2text.py          # Whisper 轉錄模組
│   └── text2vedio.py          # 影片生成模組
├── data/                      # 輸入音檔目錄
└── results/                   # 輸出結果目錄
```

## 常見問題 (FAQ)

### Q1: 首次啟動很慢？

**A:** 首次啟動時需要下載模型檔案（約 15GB+），請耐心等待。後續啟動會使用快取的模型。

### Q2: GPU 記憶體不足？

**A:**
1. 啟用 4-bit 量化選項
2. 減少 FPS 或影片時長
3. 關閉其他 GPU 程式
4. 使用 CPU 模式（較慢）

### Q3: 生成影片需要多久？

**A:** 取決於：
- GPU 效能（GPU 比 CPU 快 10-50 倍）
- 影片參數（更多幀數 = 更長時間）
- 是否使用量化（量化稍慢但省記憶體）

典型時間：
- GPU (16GB): 3 秒影片約 2-5 分鐘
- GPU (8GB + 量化): 3 秒影片約 3-7 分鐘
- CPU: 不建議（可能需要 30+ 分鐘）

### Q4: 支援批次處理嗎？

**A:** 目前版本僅支援單個音檔處理。批次處理將在未來版本加入。

### Q5: 可以使用其他語言嗎？

**A:** 可以！Whisper 支援多語言自動偵測。上傳任何語言的音檔即可自動辨識。

## 效能優化建議

### 提升速度：
1. ✅ 使用 GPU（必要）
2. ✅ 關閉其他佔用 GPU 的程式
3. ✅ 降低 FPS（16 已是最佳平衡點）
4. ✅ 縮短影片時長

### 節省記憶體：
1. ✅ 啟用 4-bit 量化
2. ✅ 降低 FPS
3. ✅ 清理暫存檔案（點擊側邊欄「Clear All Data」）
4. ✅ 重啟應用程式釋放記憶體

## 故障排除

### 錯誤：CUDA out of memory

```bash
# 解決方案 1: 啟用量化
# 在 UI 中勾選「Enable 4-bit Quantization」

# 解決方案 2: 減少幀數
# 降低 Duration 或 FPS

# 解決方案 3: 清理 GPU 快取
python -c "import torch; torch.cuda.empty_cache()"
```

### 錯誤：Module not found

```bash
# 重新安裝依賴
pip install -r requirements.txt --upgrade
```

### 錄音功能無法使用

**瀏覽器權限：**
- Chrome/Edge: 允許麥克風存取
- Firefox: 檢查隱私設定
- Safari: 系統偏好設定 → 安全性與隱私權 → 麥克風

## 技術細節

### 模型資訊

**語音辨識：**
- 模型: OpenAI Whisper Large v3
- 大小: ~3GB
- 輸出: 文字 + 時間戳記

**影片生成：**
- 模型: Wan-AI/Wan2.1-T2V-1.3B
- 大小: ~13GB (全精度) / ~7GB (量化)
- 輸出: MP4 影片

### 資料流程

```
音檔 → Whisper → 文字轉錄 → Wan2.1 → 影片
        (3-5分鐘)  (可編輯)     (2-10分鐘)
```

## 授權與致謝

本專案整合了以下開源專案：
- [Streamlit](https://streamlit.io/) - Web 框架
- [OpenAI Whisper](https://github.com/openai/whisper) - 語音辨識
- [Wan-AI](https://huggingface.co/Wan-AI) - 文字轉影片模型

## 支援

如遇問題或有建議，請：
1. 檢查本文件的常見問題區塊
2. 查看終端機的錯誤訊息
3. 確認系統需求符合最低配置

---

**享受創作！🎬✨**
