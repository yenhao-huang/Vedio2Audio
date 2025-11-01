"""
Audio2Video Streamlit Application
Main application file for the web interface
"""

import streamlit as st
import yaml
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from backend.pipeline import Audio2VideoPipeline
from components.audio_input import render_audio_input
from components.transcription import (
    render_transcription_section,
    render_timestamp_chunks,
    download_transcription,
    render_transcription_controls
)
from components.video_generator import (
    render_video_generation_controls,
    display_generation_progress,
    display_video_result,
    display_video_placeholder
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Initialize pipeline with caching
@st.cache_resource
def get_pipeline(_config):
    """Get cached pipeline instance"""
    return Audio2VideoPipeline(_config)


def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None

    if 'transcription_data' not in st.session_state:
        st.session_state.transcription_data = None

    if 'video_path' not in st.session_state:
        st.session_state.video_path = None

    if 'video_stats' not in st.session_state:
        st.session_state.video_stats = None

    if 'edited_text' not in st.session_state:
        st.session_state.edited_text = None


def main():
    """Main application function"""
    # Load configuration
    config = load_config()

    # Page configuration
    st.set_page_config(
        page_title=config['ui']['page_title'],
        page_icon=config['ui']['page_icon'],
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("🎬 Audio2Video Generator")
    st.markdown("Transform your audio into stunning AI-generated videos")
    st.markdown("---")

    # Main layout - Three columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # Column 1: Audio Input
    with col1:
        audio_path, input_method = render_audio_input()

        if audio_path:
            st.session_state.audio_path = audio_path

    # Column 2: Transcription
    with col2:
        if st.session_state.audio_path:
            start_transcribe, clear_transcribe = render_transcription_controls()

            if start_transcribe:
                with st.spinner("🔄 Transcribing audio..."):
                    try:
                        # Get pipeline
                        pipeline = get_pipeline(config)

                        # Create progress placeholder
                        progress_placeholder = st.empty()

                        def update_progress(msg, progress):
                            progress_placeholder.progress(progress, text=msg)

                        # Transcribe
                        result = pipeline.transcribe_audio(
                            st.session_state.audio_path,
                            progress_callback=update_progress
                        )

                        st.session_state.transcription_data = result
                        progress_placeholder.empty()
                        st.success("✅ Transcription complete!")

                    except Exception as e:
                        st.error(f"❌ Transcription failed: {str(e)}")
                        logger.error(f"Transcription error: {str(e)}", exc_info=True)

            if clear_transcribe:
                st.session_state.transcription_data = None
                st.session_state.edited_text = None
                st.rerun()

        # Display transcription
        if st.session_state.transcription_data:
            edited_text = render_transcription_section(
                st.session_state.transcription_data,
                show_editor=True
            )
            st.session_state.edited_text = edited_text

            # Download button
            st.markdown("---")
            download_transcription(
                st.session_state.transcription_data,
                filename="transcription.json"
            )

    # Column 3: Video Generation
    with col3:
        if st.session_state.edited_text or (
            st.session_state.transcription_data and
            st.session_state.transcription_data.get('transcription', {}).get('text')
        ):
            # Get text for video generation
            text_for_video = st.session_state.edited_text or \
                st.session_state.transcription_data['transcription']['text']

            # Generation controls with parameters
            generate, duration, fps = render_video_generation_controls(config)

            if generate:
                with st.spinner("🎬 Generating video..."):
                    try:
                        # Get pipeline
                        pipeline = get_pipeline(config)

                        # Create progress placeholder
                        progress_placeholder = st.empty()

                        def update_progress(msg, progress):
                            progress_placeholder.progress(progress, text=msg)

                        # Generate video
                        video_path, stats = pipeline.generate_video(
                            text_for_video,
                            duration=duration,
                            fps=fps,
                            progress_callback=update_progress
                        )

                        st.session_state.video_path = video_path
                        st.session_state.video_stats = stats
                        progress_placeholder.empty()

                    except Exception as e:
                        st.error(f"❌ Video generation failed: {str(e)}")
                        logger.error(f"Video generation error: {str(e)}", exc_info=True)

            # Display video result
            st.markdown("---")
            if st.session_state.video_path:
                display_video_result(
                    st.session_state.video_path,
                    st.session_state.video_stats
                )
            else:
                display_video_placeholder()

        else:
            st.info("📝 Transcribe audio first to generate video")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with Streamlit | Powered by Whisper & Wan2.1-T2V"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
