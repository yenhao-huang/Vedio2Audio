"""
Quick Test Script for Video Generation

Tests a single video generation to verify the setup before running full experiments.
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.text2vedio import Text2Vedio


def quick_test(model_name="wan_2_1", quantized=False):
    """Run a single quick test"""
    print("\n" + "="*80)
    print("QUICK VIDEO GENERATION TEST")
    print("="*80)

    # Setup output directory
    output_dir = Path(__file__).parent / "results" / "quick_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test configuration - using first prompt from test suite
    test_config = {
        "id": "p1_car_city_driving",
        "description": "紅色跑車在城市街道上高速行駛",
        "prompt": "A sleek red sports car driving fast through a neon-lit city street at night, wet pavement reflecting colorful lights, the car accelerating and drifting around corners",
        "duration": 5,
        "fps": 16
    }

    print(f"\n📋 Test Configuration:")
    print(f"   Description: {test_config['description']}")
    print(f"   Duration: {test_config['duration']}s")
    print(f"   FPS: {test_config['fps']}")
    print(f"   Frames: {test_config['duration'] * test_config['fps']}")
    print(f"   Prompt: {test_config['prompt']}")

    # Initialize generator
    print(f"\n🔧 Initializing Text2Vedio generator...")
    print(f"   Model: {model_name}")
    print(f"   Quantization: {quantized}")
    try:
        generator = Text2Vedio(model_name=model_name, quantized=quantized)
        print("   ✅ Generator initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize generator: {str(e)}")
        return False

    # Generate video
    output_path = output_dir / "test_video.mp4"
    print(f"\n🎬 Generating video...")
    print(f"   Output: {output_path}")

    try:
        import time
        start_time = time.time()

        # Generate video frames using text_to_video method
        video_frames = generator.text_to_video(
            text=test_config['prompt'],
            duration=test_config['duration'],
            fps=test_config['fps']
        )

        # Save video using save_video method
        generator.save_video(video_frames, str(output_path), fps=test_config['fps'])

        end_time = time.time()
        generation_time = end_time - start_time
        video_path = output_path

        print(f"\n✅ Video generated successfully!")
        print(f"   Path: {video_path}")
        print(f"   Generation Time: {generation_time:.2f}s")
        print(f"   Time per Second: {generation_time/test_config['duration']:.2f}s")

        # Get and display stats
        stats = generator.get_usage_statistics()
        if stats:
            print(f"\n📊 Statistics:")
            print(f"   Peak GPU Memory: {stats.get('peak_memory_usage_gb', 0):.2f} GB")
            print(f"   Current GPU Memory: {stats.get('current_memory_allocated_gb', 0):.2f} GB")
            print(f"   Average Inference Time: {stats.get('average_inference_time', 0):.2f}s")
            print(f"   Initialization Time: {stats.get('initialization_time', 0):.2f}s")

        # Save results
        result = {
            "config": test_config,
            "video_path": str(video_path),
            "generation_time": generation_time,
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        result_path = output_dir / "test_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n💾 Results saved to: {result_path}")
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nYou can now run the full experiment suite:")
        print(f"  python experiments/test_video_generation.py")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Video generation failed: {str(e)}")
        import traceback
        traceback.print_exc()

        # Save error result
        result = {
            "config": test_config,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }

        result_path = output_dir / "test_result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick test for video generation")
    parser.add_argument("--model_name", type=str, default="wan_2_1",
                       choices=["wan_2_1", "wan_2_2"],
                       help="Model to use: 'wan_2_1' (default) or 'wan_2_2'")
    parser.add_argument("--quantize", action="store_true",
                       help="Use 4-bit quantization (only for wan_2_1)")
    args = parser.parse_args()

    try:
        success = quick_test(model_name=args.model_name, quantized=args.quantize)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
