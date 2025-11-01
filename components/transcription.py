"""
Transcription Component for Streamlit
Displays and manages transcription results
"""

import streamlit as st
import json
from typing import Dict, Optional


def render_transcription_section(
    transcription_data: Optional[Dict] = None,
    show_editor: bool = True
) -> Optional[str]:
    """
    Render transcription display and editor

    Args:
        transcription_data: Transcription result dictionary
        show_editor: Whether to show text editor

    Returns:
        Edited text or None
    """
    st.subheader("📝 Transcription")

    if transcription_data is None:
        st.info("🎯 Transcribe audio to see results here")
        return None

    # Extract transcription text
    transcription = transcription_data.get('transcription', {})
    full_text = transcription.get('text', '')
    chunks = transcription.get('chunks', [])

    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Characters", len(full_text))
    with col2:
        st.metric("Words", len(full_text.split()))
    with col3:
        st.metric("Segments", len(chunks))

    # Display full transcription
    st.markdown("---")
    st.markdown("**Full Transcription:**")

    if show_editor:
        # Editable text area
        edited_text = st.text_area(
            "Edit transcription if needed:",
            value=full_text,
            height=200,
            key="transcription_editor"
        )

        if edited_text != full_text:
            st.info("✏️ Transcription has been edited")

        return edited_text
    else:
        # Read-only display
        st.text_area(
            "Transcription:",
            value=full_text,
            height=200,
            disabled=True
        )
        return full_text


def render_timestamp_chunks(transcription_data: Dict):
    """
    Render timestamp chunks in an expandable section

    Args:
        transcription_data: Transcription result dictionary
    """
    transcription = transcription_data.get('transcription', {})
    chunks = transcription.get('chunks', [])

    if not chunks:
        return

    with st.expander(f"🕐 View Timestamp Segments ({len(chunks)} segments)", expanded=False):
        st.markdown("*Each segment shows the text with its corresponding timestamp.*")

        # Create a scrollable container
        for i, chunk in enumerate(chunks[:50]):  # Limit to first 50 for performance
            timestamp = chunk.get('timestamp', [0, 0])
            text = chunk.get('text', '')

            start_time = format_timestamp(timestamp[0])
            end_time = format_timestamp(timestamp[1])

            st.markdown(
                f"""
                <div style="padding: 8px; margin: 4px 0; background-color: #f0f2f6; border-radius: 4px;">
                    <small><b>[{start_time} → {end_time}]</b></small><br>
                    {text}
                </div>
                """,
                unsafe_allow_html=True
            )

        if len(chunks) > 50:
            st.info(f"Showing first 50 of {len(chunks)} segments")


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to MM:SS format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def download_transcription(transcription_data: Dict, filename: str = "transcription.json"):
    """
    Create download button for transcription JSON

    Args:
        transcription_data: Transcription result dictionary
        filename: Output filename
    """
    json_str = json.dumps(transcription_data, ensure_ascii=False, indent=2)

    st.download_button(
        label="📥 Download Transcription (JSON)",
        data=json_str,
        file_name=filename,
        mime="application/json",
        help="Download the full transcription with timestamps"
    )


def render_transcription_controls():
    """
    Render control buttons for transcription operations
    """
    col1, col2 = st.columns(2)

    with col1:
        start_transcribe = st.button(
            "🚀 Start Transcription",
            type="primary",
            use_container_width=True,
            key="start_transcribe_btn"
        )

    with col2:
        clear_results = st.button(
            "🗑️ Clear Results",
            use_container_width=True,
            key="clear_transcribe_btn"
        )

    return start_transcribe, clear_results
