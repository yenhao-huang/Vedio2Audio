import argparse
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline, DiffusionPipeline, AutoencoderKLWan
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel
import logging
import time
import psutil
import os

class Text2Vedio:
    """
    Text2Vedio class for generating videos from text prompts using the WanPipeline.
    """

    def __init__(self, quantized=False, model_name="wan_2_1"):
        """
        Initialize the Text2Vedio pipeline.
        
        Args:
            quantized: If False (default), use 4-bit quantization.
                    If True, load full precision models (more memory).
        """
        # Setup logger - save logs to logs directory
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'text2vedio.log')

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=log_file, filemode='a', level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")

        # Initialize usage tracking metrics
        self.metrics = {
            'initialization_time': 0,
            'inference_times': [],
            'peak_memory_usage': 0,
            'total_videos_generated': 0
        }

        print("🚀 Initializing Text2Vedio pipeline...")
        init_start_time = time.time()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

        # Validate model_name and set default
        if model_name not in ["wan_2_1", "wan_2_2"]:
            print(f"⚠️ Invalid model_name '{model_name}', defaulting to 'wan_2_1'")
            model_name = "wan_2_1"

        # Load the appropriate pipeline based on model selection
        if model_name == "wan_2_1":
            self.model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            self.pipe = self._load_wan2_1_pipeline(quantized)
        elif model_name == "wan_2_2":
            self.model_name = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
            self.pipe = self._load_wan_2_2_pipeline()

        # Set default video generation parameters
        self.fps = 16  # Frames per second
        self.default_duration = 1  # Default video duration in seconds

        # Record initialization time
        self.metrics['initialization_time'] = time.time() - init_start_time
        self.logger.info(f"Model initialization completed in {self.metrics['initialization_time']:.2f} seconds")
        print(f"✅ Model initialization complete! (Time: {self.metrics['initialization_time']:.2f}s)")

    def _load_wan2_1_pipeline(self, quantized):
        if not quantized:
            # Load model components
            print("📦 Loading text encoder...")
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
            ).to(self.device)
            self._log_memory_usage("After loading text encoder")

            print("📦 Loading VAE...")
            vae = AutoModel.from_pretrained(
                self.model_name, subfolder="vae", torch_dtype=torch.float32
            ).to(self.device)
            self._log_memory_usage("After loading VAE")

            print("📦 Loading transformer...")
            transformer = AutoModel.from_pretrained(
                self.model_name, subfolder="transformer", torch_dtype=torch.bfloat16
            ).to(self.device)
            self._log_memory_usage("After loading transformer")

            print("⚙️ Initializing pipeline...")

            pipe = WanPipeline.from_pretrained(
                self.model_name,
                vae=vae,
                transformer=transformer,
                text_encoder=text_encoder,
                torch_dtype=self.dtype,
            ).to(self.device)

            self._log_memory_usage("After initializing pipeline")
        else:
            # 使用 4-bit 量化（節省記憶體）
            print("🔧 Using 4-bit quantization for memory optimization...")

            pipeline_quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )

            pipe = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                pipeline_quantization_config=pipeline_quant_config,
            ).to(self.device)
        
        return pipe

    def _load_wan_2_2_pipeline(self):
        vae = AutoencoderKLWan.from_pretrained(
            self.model_name, 
            subfolder="vae", 
            torch_dtype=self.dtype
        ).to(self.device)

        pipe = WanPipeline.from_pretrained(
            self.model_name, 
            vae=vae, 
            torch_dtype=self.dtype
        ).to(self.device)
        
        return pipe


    def _log_memory_usage(self, stage):
        """
        Log current memory usage at a specific stage.
        """
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
            memory_free = (torch.cuda.get_device_properties(device_id).total_memory -
                          torch.cuda.memory_allocated(device_id)) / (1024 ** 3)

            # Update peak memory usage
            if memory_allocated > self.metrics['peak_memory_usage']:
                self.metrics['peak_memory_usage'] = memory_allocated

            self.logger.info(f"[{stage}] GPU Memory - Allocated: {memory_allocated:.2f} GB, "
                           f"Reserved: {memory_reserved:.2f} GB, Free: {memory_free:.2f} GB")
            print(f"   📊 Memory: {memory_allocated:.2f} GB allocated, {memory_free:.2f} GB free")
        else:
            # Log CPU memory usage
            process = psutil.Process()
            cpu_memory = process.memory_info().rss / (1024 ** 3)
            self.logger.info(f"[{stage}] CPU Memory Usage: {cpu_memory:.2f} GB")
            print(f"   📊 CPU Memory: {cpu_memory:.2f} GB")

    def _get_gpu_utilization(self):
        """
        Get current GPU utilization percentage.
        """
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                device_id = torch.cuda.current_device()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                pynvml.nvmlShutdown()
                return util.gpu
            except ImportError:
                self.logger.warning("pynvml not available, GPU utilization monitoring disabled")
                return None
            except Exception as e:
                self.logger.warning(f"Error getting GPU utilization: {e}")
                return None
        return None

    def get_usage_statistics(self):
        """
        Get comprehensive usage statistics for the model.
        """
        stats = {
            'initialization_time': self.metrics['initialization_time'],
            'total_videos_generated': self.metrics['total_videos_generated'],
            'peak_memory_usage_gb': self.metrics['peak_memory_usage'],
            'average_inference_time': np.mean(self.metrics['inference_times']) if self.metrics['inference_times'] else 0,
            'total_inference_time': sum(self.metrics['inference_times'])
        }

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            stats['current_memory_allocated_gb'] = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            stats['current_memory_reserved_gb'] = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
            stats['device_name'] = torch.cuda.get_device_name(device_id)

        return stats

    def log_usage_statistics(self):
        """
        Log usage statistics to logger and console.
        """
        stats = self.get_usage_statistics()

        print("\n" + "="*60)
        print("📊 MODEL USAGE STATISTICS")
        print("="*60)
        print(f"Initialization Time: {stats['initialization_time']:.2f}s")
        print(f"Total Videos Generated: {stats['total_videos_generated']}")
        print(f"Peak Memory Usage: {stats['peak_memory_usage_gb']:.2f} GB")

        if stats['total_videos_generated'] > 0:
            print(f"Average Inference Time: {stats['average_inference_time']:.2f}s")
            print(f"Total Inference Time: {stats['total_inference_time']:.2f}s")

        if torch.cuda.is_available():
            print(f"Current GPU Memory Allocated: {stats['current_memory_allocated_gb']:.2f} GB")
            print(f"Current GPU Memory Reserved: {stats['current_memory_reserved_gb']:.2f} GB")
            print(f"Device: {stats['device_name']}")

        print("="*60 + "\n")

        # Log to file
        self.logger.info("="*60)
        self.logger.info("MODEL USAGE STATISTICS")
        self.logger.info(f"Stats: {stats}")
        self.logger.info("="*60)

    def text_to_video(self, text, duration=None, fps=None, negative_prompt=None):
        """
        Generate video frames from a given text prompt.

        Args:
            text: Text prompt for video generation
            duration: Video duration in seconds (default: use self.default_duration)
            fps: Frames per second (default: use self.fps)
            negative_prompt: Negative prompt to guide what to avoid in generation
                           (default: predefined quality/artifact avoidance prompt)
        """
        # Use default values if not specified
        if duration is None:
            duration = self.default_duration
        if fps is None:
            fps = self.fps

        # Validate parameters
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")

        # Calculate number of frames
        num_frames = int(duration * fps)

        print(f"🧠 Generating video frames from text prompt...")
        print(f"   ⏱️  Duration: {duration}s, FPS: {fps}, Total frames: {num_frames}")
        self.logger.info(f"Starting video generation for prompt: {text[:100]}...")
        self.logger.info(f"Video parameters - Duration: {duration}s, FPS: {fps}, Frames: {num_frames}")

        # Log memory before inference
        self._log_memory_usage("Before inference")

        # Start timing inference
        inference_start_time = time.time()

        # Get GPU utilization before inference
        gpu_util_before = self._get_gpu_utilization()
        if gpu_util_before is not None:
            self.logger.info(f"GPU Utilization before inference: {gpu_util_before}%")
            print(f"   🔋 GPU Utilization: {gpu_util_before}%")

        # Use default negative prompt if none provided
        if negative_prompt is None:
            negative_prompt = """
            low quality, worst quality, normal quality, jpeg artifacts, blurry, out of focus,
            duplicate, watermark, text, error, cropped, extra fingers, mutated hands, poorly drawn hands,
            poorly drawn face, deformed, ugly, bad anatomy, disfigured, tiling, grainy, pixelated,
            fused fingers, extra limbs, missing limbs, malformed limbs, bad proportions,
            unrealistic, unnatural lighting, deformed body, long neck, blurry eyes
            """

        with torch.no_grad():
            output = self.pipe(
                prompt=text,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=5.0,
            ).frames
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared after inference")

        # Record inference time
        inference_time = time.time() - inference_start_time
        self.metrics['inference_times'].append(inference_time)
        self.metrics['total_videos_generated'] += 1

        # Log memory after inference
        self._log_memory_usage("After inference")

        # Get GPU utilization after inference
        gpu_util_after = self._get_gpu_utilization()
        if gpu_util_after is not None:
            self.logger.info(f"GPU Utilization after inference: {gpu_util_after}%")

        self.logger.info(f"Video generation completed in {inference_time:.2f} seconds")
        print(f"✅ Video frames generated successfully! (Time: {inference_time:.2f}s)")

        return output

    def save_video(self, output, output_path, fps=None):
        """
        Save generated video frames to a video file.

        Args:
            output: Generated video frames (numpy array)
            output_path: Path to save the video file
            fps: Frames per second (default: use self.fps)
        """
        if fps is None:
            fps = self.fps

        print(f"💾 Saving video to {output_path} ...")
        self.logger.info(f"Saving video with FPS: {fps}")

        # Debug: Check output format
        self.logger.info(f"Output type: {type(output)}")
        if isinstance(output, np.ndarray):
            self.logger.info(f"Output shape: {output.shape}, dtype: {output.dtype}")

            # Handle 5-dimensional output: (batch, frames, height, width, channels)
            if output.ndim == 5:
                # Extract first batch element to get (frames, height, width, channels)
                video_frames = output[0]
                self.logger.info(f"Extracted video_frames shape: {video_frames.shape}")
            else:
                video_frames = output
        else:
            video_frames = output

        export_to_video(video_frames, output_path, fps=fps)
        print(f"🎬 Video saved successfully: {output_path}")

    def read_text(self, text_file):
        """
        Read text prompt from a file.
        Supports both plain text and JSON format with 'transcription.text' field.
        """
        import json

        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Try to parse as JSON first
        try:
            data = json.loads(content)
            # Check if it has transcription.text field
            if 'transcription' in data and 'text' in data['transcription']:
                self.text = data['transcription']['text']
                print(f"📄 Text prompt loaded from JSON: {text_file}")
                print(f"   Preview: {self.text[:100]}...")
            else:
                # Use the whole JSON as text
                self.text = content
                print(f"📄 Raw content loaded from: {text_file}")
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            self.text = content
            print(f"📄 Plain text loaded from: {text_file}")

        return self.text

def main():
    argparser = argparse.ArgumentParser(description="Text to Video Generation")
    argparser.add_argument("--text_file", type=str, default="results/1000_transcription.json",
                          help="Input text prompt for video generation")
    argparser.add_argument("--output_dir", type=str, default="results",
                          help="Directory to save the generated video file")
    argparser.add_argument("--duration", type=float, default=1,
                          help="Video duration in seconds (default: 1.0)")
    argparser.add_argument("--fps", type=int, default=16,
                          help="Frames per second (default: 16)")
    argparser.add_argument("--model_name", type=str, default="wan_2_1",
                          choices=["wan_2_1", "wan_2_2"],
                          help="Model to use: 'wan_2_1' (Wan2.1-T2V-1.3B, stable) or 'wan_2_2' (Wan2.2-TI2V-5B, experimental)")
    argparser.add_argument("--negative_prompt", type=str, default=None,
                          help="Negative prompt to guide what to avoid in generation (optional)")
    argparser.add_argument("--quantize", action="store_true",
                          help="Use 4-bit quantization to reduce memory usage (only supported for wan_2_1)")
    args = argparser.parse_args()

    print("🔧 Starting text-to-video generation process...")
    print(f"   📋 Parameters: Model={args.model_name}, Duration={args.duration}s, FPS={args.fps}")
    if args.quantize:
        print(f"   🔧 Quantization: ENABLED (4-bit)")
        if args.model_name == "wan_2_2":
            print(f"   ⚠️ Warning: Quantization is not fully tested with wan_2_2")

    t2v = Text2Vedio(model_name=args.model_name, quantized=args.quantize)
    text = t2v.read_text(args.text_file)

    # Generate video with specified duration and fps
    video_frames = t2v.text_to_video(text, duration=args.duration, fps=args.fps,
                                     negative_prompt=args.negative_prompt)

    # Save video with specified fps
    filename = args.text_file.split('/')[-1].split('.')[0]
    output_file = f"{args.output_dir}/{filename}_video.mp4"
    t2v.save_video(video_frames, output_file, fps=args.fps)

    print(f"✅ Process complete! Video available at {output_file}")

    # Log usage statistics
    t2v.log_usage_statistics()

if __name__ == "__main__":
    main()
