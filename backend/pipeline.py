"""
Backend Pipeline for Audio2Video
Integrates audio2text and text2vedio modules with caching and state management
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.audio2text import AutomaticSpeechRecognition
from utils.text2vedio import Text2Vedio


class Audio2VideoPipeline:
    """
    Manages the complete audio-to-video pipeline with model caching
    """

    def __init__(self, config: Dict):
        """
        Initialize pipeline with configuration

        Args:
            config: Configuration dictionary from settings.yaml
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Model instances (lazy loaded)
        self._asr_model = None
        self._t2v_model = None

        # Ensure output directories exist
        self.temp_dir = Path(config['files']['temp_dir'])
        self.output_dir = Path(config['files']['output_dir'])
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def asr_model(self) -> AutomaticSpeechRecognition:
        """Lazy-load ASR model with caching"""
        if self._asr_model is None:
            self.logger.info("Loading ASR model (Whisper)...")
            self._asr_model = AutomaticSpeechRecognition()
            self.logger.info("ASR model loaded successfully")
        return self._asr_model

    @property
    def t2v_model(self) -> Text2Vedio:
        """Lazy-load Text-to-Video model with caching"""
        if self._t2v_model is None:
            self.logger.info("Loading Text2Video model...")
            quantize = self.config['video'].get('quantization', False)
            model_key = self.config['video'].get('model_key', 'wan_2_1')
            self.logger.info(f"Using model: {model_key}, quantization: {quantize}")
            self._t2v_model = Text2Vedio(model_name=model_key, quantized=quantize)
            self.logger.info("Text2Video model loaded successfully")
        return self._t2v_model

    def transcribe_audio(
        self,
        audio_path: str,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Transcribe audio file to text with timestamps

        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with transcription results
        """
        try:
            if progress_callback:
                progress_callback("Loading ASR model...", 0.1)

            # Load model
            asr = self.asr_model

            if progress_callback:
                progress_callback("Transcribing audio...", 0.3)

            # Transcribe
            result = asr.transcribe(audio_path)

            if progress_callback:
                progress_callback("Transcription complete!", 1.0)

            return {
                'audio_file': audio_path,
                'transcription': result
            }

        except Exception as e:
            self.logger.error(f"Transcription failed: {str(e)}")
            raise

    def generate_video(
        self,
        text: str,
        duration: float = 3.0,
        fps: int = 16,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[str, Dict]:
        """
        Generate video from text prompt

        Args:
            text: Text prompt for video generation
            duration: Video duration in seconds
            fps: Frames per second
            output_path: Optional custom output path
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (video_path, statistics_dict)
        """
        try:
            if progress_callback:
                progress_callback("Loading Text2Video model...", 0.1)

            # Load model
            t2v = self.t2v_model

            if progress_callback:
                progress_callback("Generating video frames...", 0.3)

            # Generate video
            video_output = t2v.text_to_video(text, duration, fps)

            if progress_callback:
                progress_callback("Saving video...", 0.8)

            # Determine output path
            if output_path is None:
                output_path = str(self.output_dir / f"generated_{len(os.listdir(self.output_dir))}.mp4")

            # Save video
            t2v.save_video(video_output, output_path, fps)

            # Get statistics
            stats = t2v.get_usage_statistics()

            if progress_callback:
                progress_callback("Video generation complete!", 1.0)

            return output_path, stats

        except Exception as e:
            self.logger.error(f"Video generation failed: {str(e)}")
            raise

    def run_full_pipeline(
        self,
        audio_path: str,
        duration: float = 3.0,
        fps: int = 16,
        save_transcription: bool = True,
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """
        Run complete audio-to-video pipeline

        Args:
            audio_path: Path to input audio file
            duration: Video duration in seconds
            fps: Frames per second
            save_transcription: Whether to save transcription JSON
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with all results
        """
        results = {}

        try:
            # Step 1: Transcribe audio
            if progress_callback:
                progress_callback("Step 1/2: Transcribing audio...", 0.0)

            transcription_result = self.transcribe_audio(
                audio_path,
                lambda msg, p: progress_callback(f"Transcription: {msg}", p * 0.5) if progress_callback else None
            )

            results['transcription'] = transcription_result

            # Save transcription if requested
            if save_transcription:
                base_name = Path(audio_path).stem
                trans_path = self.output_dir / f"{base_name}_transcription.json"
                with open(trans_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription_result, f, ensure_ascii=False, indent=2)
                results['transcription_path'] = str(trans_path)

            # Step 2: Generate video
            if progress_callback:
                progress_callback("Step 2/2: Generating video...", 0.5)

            text = transcription_result['transcription']['text']
            video_path, stats = self.generate_video(
                text,
                duration,
                fps,
                progress_callback=lambda msg, p: progress_callback(f"Video: {msg}", 0.5 + p * 0.5) if progress_callback else None
            )

            results['video_path'] = video_path
            results['statistics'] = stats

            if progress_callback:
                progress_callback("Pipeline complete!", 1.0)

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                for file in self.temp_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {str(e)}")
