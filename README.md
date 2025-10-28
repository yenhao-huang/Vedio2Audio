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

使用 Wan2.1-T2V 擴散模型，根據文字提示生成視頻。

```bash
python utils/text2vedio.py --text "你的文字提示" --output_path output.mp4
```

- `--text`：輸入用來生成視頻的文字描述（必填）。
- `--output_path`：生成視頻的輸出路徑（預設：`output.mp4`）。

## 目錄結構

- `data/` - 輸入資料，如音頻檔案。
- `results/` - 輸出資料，如轉錄結果。
- `logs/` - 日誌和實驗相關資料。
- `exp/` - 實驗相關檔案。
- `script/` - 額外的輔助腳本。
- `utils/` - 主要工具程式，包含音頻轉文字和文字轉視頻的程式碼。
- `plans/` - 項目計劃和筆記。

## 注意事項

TODO

## 範例流程

TODO