"""
Video Generator Component for Streamlit
Handles video generation UI and controls
"""

import streamlit as st
import plotly.graph_objects as go
import subprocess
from typing import Dict, Optional, Tuple
from pathlib import Path


def render_video_generation_controls(config: Dict) -> Tuple[bool, float, int, str]:
    """
    Render video generation control button with parameters in expander

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (generate_clicked, duration, fps, model_key)
    """
    video_config = config['video']

    # Generate button
    col1, col2 = st.columns([3, 1])

    with col1:
        generate = st.button(
            "🎬 Generate Video",
            type="primary",
            use_container_width=True,
            key="generate_video_btn"
        )

    with col2:
        st.button(
            "🗑️ Clear",
            use_container_width=True,
            key="clear_video_btn"
        )

    # Video parameters in expander (dropdown)
    with st.expander("⚙️ Video Parameters", expanded=False):
        # Model selection
        st.markdown("### 🤖 Model Selection")
        model_key = st.radio(
            "Select Model",
            options=["wan_2_1", "wan_2_2"],
            format_func=lambda x: {
                "wan_2_1": "WAN 2.1 (1.3B - Stable, Supports Quantization)",
                "wan_2_2": "WAN 2.2 (5B - Larger, Experimental)"
            }[x],
            index=0 if video_config.get('model_key', 'wan_2_1') == 'wan_2_1' else 1,
            help="WAN 2.1: Smaller, faster, stable\nWAN 2.2: Larger, potentially better quality",
            key="model_selection"
        )

        st.markdown("---")
        st.markdown("### 📐 Video Settings")

        col1, col2 = st.columns(2)

        with col1:
            duration = st.slider(
                "Duration (seconds)",
                min_value=float(video_config['min_duration']),
                max_value=float(video_config['max_duration']),
                value=float(video_config['default_duration']),
                step=0.5,
                help="Length of the generated video"
            )

        with col2:
            fps = st.slider(
                "FPS (Frames Per Second)",
                min_value=video_config['min_fps'],
                max_value=video_config['max_fps'],
                value=video_config['default_fps'],
                step=2,
                help="Higher FPS = smoother video (but longer generation time)"
            )

    return generate, duration, fps, model_key


def display_generation_progress(status: str, progress: float):
    """
    Display generation progress with status message

    Args:
        status: Status message
        progress: Progress value (0.0 to 1.0)
    """
    st.progress(progress, text=status)


def convert_to_web_compatible(input_path: str) -> str:
    """
    Convert video to web-compatible format using ffmpeg

    Args:
        input_path: Path to input video file

    Returns:
        Path to converted video file

    Raises:
        RuntimeError: If ffmpeg conversion fails
    """
    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_web.mp4")

    # FFmpeg command for browser-compatible video
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-i", str(input_path),
        "-c:v", "libx264",  # H.264 video codec (universally supported)
        "-pix_fmt", "yuv420p",  # Pixel format for maximum compatibility
        "-c:a", "aac",  # AAC audio codec
        "-movflags", "faststart",  # Enable streaming (moov atom at start)
        "-loglevel", "error",  # Only show errors
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install ffmpeg: apt-get install ffmpeg")


def display_video_result(video_path: str, stats: Optional[Dict] = None):
    """
    Display generated video and statistics

    Args:
        video_path: Path to generated video file
        stats: Optional statistics dictionary
    """
    st.success("✅ Video generated successfully!")

    # Convert video to web-compatible format
    try:
        with st.spinner("Converting video for web playback..."):
            web_video_path = convert_to_web_compatible(video_path)

        # Read converted video file
        with open(web_video_path, 'rb') as f:
            video_bytes = f.read()

        # Display video player with video bytes
        st.video(video_bytes)

        # Download button (use original file)
        with open(video_path, 'rb') as f:
            original_bytes = f.read()

        filename = Path(video_path).name
        st.download_button(
            label="📥 Download Video",
            data=original_bytes,
            file_name=filename,
            mime="video/mp4"
        )

    except RuntimeError as e:
        st.error(f"❌ Video conversion failed: {str(e)}")
        st.warning("Trying to display original video (may not play in browser)")

        # Fallback: try to display original video
        try:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)

            st.download_button(
                label="📥 Download Video",
                data=video_bytes,
                file_name=Path(video_path).name,
                mime="video/mp4"
            )
        except Exception as e2:
            st.error(f"❌ Failed to load video: {str(e2)}")
            st.info(f"Video path: {video_path}")

    except Exception as e:
        st.error(f"❌ Failed to display video: {str(e)}")
        st.info(f"Video path: {video_path}")


def render_statistics_dashboard(stats: Dict):
    """
    Render performance statistics dashboard

    Args:
        stats: Statistics dictionary from Text2Vedio
    """
    with st.expander("📊 Performance Metrics", expanded=False):
        # Time metrics
        st.markdown("### ⏱️ Timing")
        col1, col2, col3 = st.columns(3)

        with col1:
            init_time = stats.get('initialization_time', 0)
            st.metric("Initialization", f"{init_time:.2f}s")

        with col2:
            inference_time = stats.get('last_inference_time', 0)
            st.metric("Generation", f"{inference_time:.2f}s")

        with col3:
            total_time = init_time + inference_time
            st.metric("Total", f"{total_time:.2f}s")

        # Memory metrics
        st.markdown("### 💾 Memory Usage")

        memory_data = []
        if 'cpu_memory_used_mb' in stats:
            memory_data.append({
                'type': 'CPU Memory',
                'value': stats['cpu_memory_used_mb'],
                'unit': 'MB'
            })

        if 'gpu_memory_used_mb' in stats:
            memory_data.append({
                'type': 'GPU Memory',
                'value': stats['gpu_memory_used_mb'],
                'unit': 'MB'
            })

        if 'peak_gpu_memory_mb' in stats:
            memory_data.append({
                'type': 'Peak GPU Memory',
                'value': stats['peak_gpu_memory_mb'],
                'unit': 'MB'
            })

        if memory_data:
            cols = st.columns(len(memory_data))
            for i, data in enumerate(memory_data):
                with cols[i]:
                    st.metric(
                        data['type'],
                        f"{data['value']:.0f} {data['unit']}"
                    )

        # GPU Utilization
        if 'gpu_utilization_percent' in stats:
            st.markdown("### 🎮 GPU Utilization")
            gpu_util = stats['gpu_utilization_percent']

            # Create gauge chart
            fig = create_gauge_chart(gpu_util, "GPU Usage")
            st.plotly_chart(fig, use_container_width=True)


def create_gauge_chart(value: float, title: str) -> go.Figure:
    """
    Create a gauge chart for metrics visualization

    Args:
        value: Percentage value (0-100)
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 70], 'color': '#FFD700'},
                {'range': [70, 100], 'color': '#FF6347'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def display_video_placeholder():
    """
    Display placeholder when no video is generated yet
    """
    st.info("🎥 Generated video will appear here")

    # Optional: Add a placeholder image or animation
    st.markdown(
        """
        <div style="text-align: center; padding: 40px; background-color: #f0f2f6; border-radius: 10px;">
            <h3 style="color: #666;">No video generated yet</h3>
            <p style="color: #888;">Generate a video from your transcription to see it here</p>
        </div>
        """,
        unsafe_allow_html=True
    )
