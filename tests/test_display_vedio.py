import streamlit as st
from pathlib import Path
from typing import Optional, Dict

import sys
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# 匯入你要測試的函式
from components.video_generator import display_video_result

# 建立假資料
def fake_stats():
    return {"duration": 3.2, "fps": 24, "model": "Wan2.1-T2V-14B"}

def main():
    st.title("🎬 display_video_result 測試")

    # 1️⃣ 給定影片檔案（你可先放一個 mp4 在當前資料夾）
    test_video = "tests/testcases/generated_cat.mp4"

    # 檢查影片是否存在
    if not Path(test_video).exists():
        st.warning(f"⚠️ 找不到測試影片：{test_video}")
        st.info("你可以放任意小型 MP4 檔案到同目錄，再重新執行。")
        return

    # 2️⃣ 呼叫函式
    st.header("正常情況")
    display_video_result(test_video, stats=fake_stats())


if __name__ == "__main__":
    main()
