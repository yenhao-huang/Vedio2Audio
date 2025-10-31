# Audio2Vedio

## 概述

Audio2Vedio 是一個將音頻輸入轉換成視頻輸出的項目，主要透過兩個步驟完成：
1. **音頻轉文字轉錄：** 採用 OpenAI Whisper 模型，將音頻檔案轉錄成文字。
2. **文字生成視頻：** 使用預訓練的擴散模型，根據文字提示生成視頻。

這個流程可以將音訊內容轉化為生動的視頻展示。

## 安裝

TODO

## 使用說明

### 0. 準備好音頻

1. youtube to mp3 (https://yt1s.ai/zh-tw/youtube-to-mp3/)

### 1. 音頻轉文字轉錄

使用 Whisper-large-v3 模型，將音頻檔案轉錄成 JSON 格式的文字結果。

```bash
python utils/audio2text.py --audio_file path/to/audio.mp3 --output_dir results/
```

- `--audio_file`：輸入音頻檔案路徑（預設：`data/1000.mp3`）。
- `--output_dir`：文字轉錄結果保存資料夾（預設：`results`）。

### 2. 文字生成視頻

使用 Wan2.1-T2V-1.3B 擴散模型，根據文字提示生成視頻。

```bash
python utils/text2vedio.py --text_file results/1000_transcription.json --output_dir results/ --duration 3.0 --fps 16
```

**參數說明：**

- `--text_file`：輸入文字檔案路徑（支援 JSON 或純文字格式，預設：`results/1000_transcription.json`）
- `--output_dir`：生成視頻的輸出目錄（預設：`results/`）
- `--duration`：視頻時長（秒），預設：`3.0` 秒
- `--fps`：每秒幀數（frames per second），預設：`16`

**範例：**

```bash
# 生成 3 秒視頻（16 fps = 48 frames）
python utils/text2vedio.py --text_file results/1000_transcription.json --duration 3.0

# 生成 5 秒視頻（16 fps = 80 frames）
python utils/text2vedio.py --text_file results/1000_transcription.json --duration 5.0

# 使用自訂 fps
python utils/text2vedio.py --text_file results/1000_transcription.json --duration 3.0 --fps 24
```

**計算公式：**
```
num_frames = duration (秒) × fps
```

例如：
- 3 秒 × 16 fps = 48 frames
- 5 秒 × 16 fps = 80 frames

## 目錄結構

- `data/` - 輸入資料，如音頻檔案
- `results/` - 輸出資料，包含轉錄結果和生成的視頻
- `logs/` - 日誌檔案（`text2vedio.log` 包含模型載入、推理時間、記憶體使用等詳細資訊）
- `exp/` - 實驗相關檔案
- `script/` - 額外的輔助腳本
- `utils/` - 主要工具程式：
  - `audio2text.py` - 音頻轉文字（使用 Whisper 模型）
  - `text2vedio.py` - 文字生成視頻（使用 Wan2.1-T2V 模型）
- `plans/` - 項目計劃和筆記

## 注意事項

TODO

## 範例流程

TODO