# 影片生成實驗總結

## 📋 實驗設計

### 核心理念
每個測試提示詞包含 **物體 + 場景 + 動作** 三個元素，全面測試模型生成能力。

### 測試矩陣

| 維度 | 配置 |
|------|------|
| 時長 | 5秒、10秒 |
| FPS | 16 |
| 提示詞 | 3個（跑車、小鳥、太空人） |
| **總測試數** | **6 tests** |

## 🎯 測試提示詞

### 1. 紅色跑車在城市街道上高速行駛
```
A sleek red sports car driving fast through a neon-lit city street at night,
wet pavement reflecting colorful lights, the car accelerating and drifting around corners
```
- **物體**: 流線型紅色跑車
- **場景**: 夜晚霓虹城市街道
- **動作**: 高速行駛、漂移

### 2. 藍色小鳥在森林中飛翔
```
A vibrant blue bird flying gracefully through a misty forest at dawn,
sunlight breaking through the trees, the bird gliding between branches and leaves
```
- **物體**: 鮮豔藍色小鳥
- **場景**: 晨霧森林
- **動作**: 優雅飛翔、穿梭

### 3. 太空人在太空站外漂浮
```
An astronaut in a white spacesuit floating outside a futuristic space station,
Earth visible in the background, the astronaut slowly rotating while conducting repairs
```
- **物體**: 白色太空服太空人
- **場景**: 太空站、地球背景
- **動作**: 漂浮、旋轉、維修

## 🚀 快速開始

### 1. 快速測試（單一影片）
```bash
python experiments/quick_test.py
```
生成一個 5 秒測試影片驗證環境配置。

### 2. 完整實驗（6個影片）
```bash
python experiments/test_video_generation.py
```
執行所有測試並生成詳細報告。

## 📊 預期輸出

```
experiments/results/exp_YYYYMMDD_HHMMSS/
├── videos/
│   ├── d5_p1_car_city_driving.mp4
│   ├── d5_p2_bird_forest_flying.mp4
│   ├── d5_p3_astronaut_space_floating.mp4
│   ├── d10_p1_car_city_driving.mp4
│   ├── d10_p2_bird_forest_flying.mp4
│   └── d10_p3_astronaut_space_floating.mp4
├── logs/ (每個測試的 JSON 詳細記錄)
├── full_results.json
└── REPORT.md (自動生成的總結報告)
```

## ⚙️ 技術細節

### 使用的方法
- `Text2Vedio.text_to_video()` - 生成影片幀
- `Text2Vedio.save_video()` - 儲存影片檔案
- `Text2Vedio.get_usage_statistics()` - 獲取性能統計

### 性能指標
- 生成時間（每個測試）
- GPU 記憶體峰值使用量
- 平均推論時間
- 初始化時間

## 💡 自訂實驗

在 `test_video_generation.py` 中修改：

```python
# 新增提示詞
TEST_PROMPTS.append({
    "id": "p4_whale_ocean_swimming",
    "description": "鯨魚在海洋中游泳",
    "prompt": "A majestic blue whale swimming through deep ocean waters, "
              "sunlight filtering from above, the whale gracefully diving and surfacing"
})

# 調整時長
DURATIONS = [3, 5, 10, 15]  # 測試更多時長
```

## 📈 分析報告內容

自動生成的 `REPORT.md` 包含：

1. ✅ 成功率統計
2. ⏱️ 按時長分析（平均生成時間）
3. 🎨 按提示詞分析（不同內容類型性能）
4. 📋 詳細結果表格
5. 💾 GPU 記憶體使用分析
6. 📝 完整提示詞列表

## ⚠️ 注意事項

- 需要 **8GB+ GPU VRAM**
- 完整實驗約 **15-30 分鐘**
- 使用 `quantized=False` 獲得最佳品質
- 建議關閉其他 GPU 程式

## 🎓 學習價值

此實驗可幫助你：
- 理解時長對生成時間的影響
- 評估不同內容類型（車輛 vs 生物 vs 人物）的生成難度
- 優化提示詞撰寫技巧
- 建立性能基準線

---

**版本**: 1.0
**最後更新**: 2025-11-01
