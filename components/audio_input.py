"""
Audio Input Component for Streamlit
Handles audio recording and file uploads
"""

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from pathlib import Path
import tempfile
from typing import Optional, Tuple


def render_audio_input() -> Tuple[Optional[str], Optional[str]]:
    """
    Render audio input UI component

    Returns:
        Tuple of (audio_file_path, input_method)
        Returns (None, None) if no audio is available
    """
    st.subheader("🎤 Audio Input")

    # Input method selection - Record Audio as default (index=0)
    input_method = st.radio(
        "Input Method",
        ["Record Audio", "Upload Audio File"],
        horizontal=True,
        index=0,  # Default to Record Audio
        label_visibility="collapsed",  # Hide label but maintain accessibility
        key="input_method"
    )

    audio_path = None

    if input_method == "Record Audio":
        audio_path = render_audio_recorder()
    else:
        audio_path = render_file_upload()

    return audio_path, input_method


def render_file_upload() -> Optional[str]:
    """
    Render file upload widget

    Returns:
        Path to uploaded audio file or None
    """
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=['mp3', 'wav', 'm4a', 'flac'],
        help="Supported formats: MP3, WAV, M4A, FLAC"
    )

    if uploaded_file is not None:
        # Save uploaded file to temporary location
        temp_dir = Path(tempfile.gettempdir()) / "audio2vedio"
        temp_dir.mkdir(exist_ok=True)

        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display audio player
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

        # Show file info
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📁 **File:** {uploaded_file.name} ({file_size_mb:.2f} MB)")

        return str(file_path)

    return None


def render_audio_recorder() -> Optional[str]:
    """
    Render audio recorder widget

    Returns:
        Path to recorded audio file or None
    """
    st.markdown("---")

    st.info("👇 Click the microphone button to start/stop recording")

    # Audio recorder
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="2x",
        key="audio_recorder"
    )

    if audio_bytes:
        # Save recorded audio
        temp_dir = Path(tempfile.gettempdir()) / "audio2vedio"
        temp_dir.mkdir(exist_ok=True)

        audio_path = temp_dir / "recorded_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Display audio player
        st.audio(audio_bytes, format="audio/wav")

        # Show recording info
        audio_size_kb = len(audio_bytes) / 1024
        st.success(f"✅ Recording saved ({audio_size_kb:.2f} KB)")

        # Download and Clear buttons
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📥 Download Recording",
                data=audio_bytes,
                file_name="recorded_audio.wav",
                mime="audio/wav",
                use_container_width=True
            )

        with col2:
            if st.button("🗑️ Clear Recording", use_container_width=True, key="clear_recording_btn"):
                # Clear the recording by returning None and rerunning
                st.session_state.pop('audio_recorder', None)
                st.rerun()

        return str(audio_path)

    return None


def display_audio_waveform(audio_path: str):
    """
    Display audio waveform visualization (optional enhancement)

    Args:
        audio_path: Path to audio file
    """
    # This is a placeholder for future waveform visualization
    # Could use librosa + matplotlib or plotly for waveform display
    pass
