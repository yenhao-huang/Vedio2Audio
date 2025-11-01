"""
Video Generation Module Testing Experiments

This script systematically tests the text-to-video generation module with:
- Different durations (5s and 10s)
- Different input types (Object, Scene, Action)
- Different prompt variations for each type
- Three different negative prompt strategies:
  1. Quality-focused: Targets low quality artifacts
  2. Anatomy-focused: Targets anatomical deformations
  3. Style-focused: Targets unwanted artistic styles

Total tests: 2 durations × 3 prompts × 3 negative prompts = 18 tests

Results are saved to experiments/results/ with detailed logs and statistics.
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.text2vedio import Text2Vedio


# Experiment Configuration
DURATIONS = [5, 10]  # seconds
FPS = 16  # frames per second

# Negative Prompt Variations
# 三種不同的負面提示詞策略
NEGATIVE_PROMPTS = [
    {
        "id": "neg1_quality",
        "description": "Quality-focused (品質導向)",
        "prompt": """
        low quality, worst quality, normal quality, jpeg artifacts, blurry, out of focus,
        duplicate, watermark, text, error, cropped, grainy, pixelated, tiling
        """
    },
    {
        "id": "neg2_anatomy",
        "description": "Anatomy-focused (解剖結構導向)",
        "prompt": """
        extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed,
        ugly, bad anatomy, disfigured, fused fingers, extra limbs, missing limbs,
        malformed limbs, bad proportions, deformed body, long neck, blurry eyes
        """
    },
    {
        "id": "neg3_style",
        "description": "Style-focused (風格導向)",
        "prompt": """
        cartoon, anime, illustration, painting, drawing, art, sketch, rendered,
        unrealistic, unnatural lighting, oversaturated, undersaturated, flat colors
        """
    }
]

# Test Prompts - Each prompt contains OBJECT + SCENE + ACTION
# 每個提示詞包含：物體、場景、動作三個元素
TEST_PROMPTS = [
    # Prompt 1: Sports Car + City Street + Driving
    {
        "id": "p1_car_city_driving",
        "description": "紅色跑車在城市街道上高速行駛",
        "prompt": "A sleek red sports car driving fast through a neon-lit city street at night, wet pavement reflecting colorful lights, the car accelerating and drifting around corners"
    },
    # Prompt 2: Bird + Forest + Flying
    {
        "id": "p2_bird_forest_flying",
        "description": "藍色小鳥在森林中飛翔",
        "prompt": "A vibrant blue bird flying gracefully through a misty forest at dawn, sunlight breaking through the trees, the bird gliding between branches and leaves"
    },
    # Prompt 3: Astronaut + Space Station + Floating
    {
        "id": "p3_astronaut_space_floating",
        "description": "太空人在太空站外漂浮",
        "prompt": "An astronaut in a white spacesuit floating outside a futuristic space station, Earth visible in the background, the astronaut slowly rotating while conducting repairs"
    }
]


def setup_experiment_dirs():
    """Create directory structure for experiment results"""
    base_dir = Path(__file__).parent / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"exp_{timestamp}"

    # Create subdirectories
    (exp_dir / "videos").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)

    return exp_dir


def run_single_test(generator, prompt_data, duration, fps, output_path, test_id, negative_prompt_data=None):
    """
    Run a single video generation test

    Args:
        generator: Text2Vedio instance
        prompt_data: Dictionary with 'prompt', 'description', 'id'
        duration: Video duration in seconds
        fps: Frames per second
        output_path: Path to save the video
        test_id: Unique test identifier
        negative_prompt_data: Dictionary with 'prompt', 'description', 'id' for negative prompt

    Returns:
        dict: Test results including stats and metadata
    """
    print(f"\n{'='*80}")
    print(f"Test ID: {test_id}")
    print(f"Description: {prompt_data['description']}")
    print(f"Duration: {duration}s | FPS: {fps} | Frames: {int(duration * fps)}")
    print(f"Prompt: {prompt_data['prompt']}")
    if negative_prompt_data:
        print(f"Negative Prompt Type: {negative_prompt_data['description']}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Get negative prompt text
        negative_prompt = negative_prompt_data['prompt'] if negative_prompt_data else None

        # Generate video frames using text_to_video method
        video_frames = generator.text_to_video(
            text=prompt_data['prompt'],
            duration=duration,
            fps=fps,
            negative_prompt=negative_prompt
        )

        # Save video using save_video method
        generator.save_video(video_frames, str(output_path), fps=fps)

        end_time = time.time()
        generation_time = end_time - start_time

        # Get statistics
        stats = generator.get_usage_statistics()

        # Compile results
        results = {
            "test_id": test_id,
            "status": "success",
            "prompt_id": prompt_data['id'],
            "description": prompt_data['description'],
            "prompt": prompt_data['prompt'],
            "negative_prompt_id": negative_prompt_data['id'] if negative_prompt_data else None,
            "negative_prompt_description": negative_prompt_data['description'] if negative_prompt_data else None,
            "negative_prompt": negative_prompt_data['prompt'] if negative_prompt_data else None,
            "duration": duration,
            "fps": fps,
            "num_frames": int(duration * fps),
            "video_path": str(output_path),
            "generation_time": generation_time,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

        print(f"✅ SUCCESS - Generated in {generation_time:.2f}s")
        print(f"   Video saved to: {output_path}")

        # Print key stats
        if stats:
            print(f"   Peak GPU Memory: {stats.get('peak_memory_usage_gb', 0):.2f} GB")
            print(f"   Average Inference Time: {stats.get('average_inference_time', 0):.2f}s")

        return results

    except Exception as e:
        end_time = time.time()
        generation_time = end_time - start_time

        print(f"❌ FAILED - Error: {str(e)}")

        return {
            "test_id": test_id,
            "status": "failed",
            "prompt_id": prompt_data['id'],
            "description": prompt_data['description'],
            "prompt": prompt_data['prompt'],
            "negative_prompt_id": negative_prompt_data['id'] if negative_prompt_data else None,
            "negative_prompt_description": negative_prompt_data['description'] if negative_prompt_data else None,
            "duration": duration,
            "fps": fps,
            "num_frames": int(duration * fps),
            "error": str(e),
            "generation_time": generation_time,
            "timestamp": datetime.now().isoformat()
        }


def run_experiments(model_name="wan_2_1", quantized=False):
    """Run all experiments"""
    print("\n" + "="*80)
    print("VIDEO GENERATION MODULE TESTING EXPERIMENTS")
    print("="*80)

    # Setup experiment directory
    exp_dir = setup_experiment_dirs()
    print(f"\n📁 Experiment directory: {exp_dir}")

    # Initialize generator
    print("\n🔧 Initializing Text2Vedio generator...")
    print(f"   Model: {model_name}")
    print(f"   Quantization: {quantized}")
    generator = Text2Vedio(model_name=model_name, quantized=quantized)

    # Store all results
    all_results = []
    test_counter = 0

    # Calculate total tests
    total_tests = len(DURATIONS) * len(TEST_PROMPTS) * len(NEGATIVE_PROMPTS)

    # Run experiments for each duration
    for duration in DURATIONS:
        print(f"\n{'#'*80}")
        print(f"# TESTING DURATION: {duration}s")
        print(f"{'#'*80}")

        # Test each prompt
        for prompt_data in TEST_PROMPTS:
            # Test each negative prompt variation
            for neg_prompt_data in NEGATIVE_PROMPTS:
                test_counter += 1
                test_id = f"d{duration}_{prompt_data['id']}_{neg_prompt_data['id']}"

                print(f"\n## Test {test_counter}/{total_tests}")

                # Output path
                output_path = exp_dir / "videos" / f"{test_id}.mp4"

                # Run test
                result = run_single_test(
                    generator=generator,
                    prompt_data=prompt_data,
                    duration=duration,
                    fps=FPS,
                    output_path=output_path,
                    test_id=test_id,
                    negative_prompt_data=neg_prompt_data
                )

                all_results.append(result)

                # Save intermediate results
                with open(exp_dir / "logs" / f"{test_id}.json", 'w') as f:
                    json.dump(result, f, indent=2)

    # Generate summary report
    generate_summary_report(all_results, exp_dir)

    print(f"\n{'='*80}")
    print(f"✅ EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total tests: {test_counter}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}\n")

    return all_results, exp_dir


def generate_summary_report(results, exp_dir):
    """Generate summary report of all experiments"""

    # Save full results
    full_results_path = exp_dir / "full_results.json"
    with open(full_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Calculate statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    # Generate markdown report
    report_lines = [
        "# Video Generation Experiment Report",
        f"\n**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Summary",
        f"- **Total Tests:** {len(results)}",
        f"- **Successful:** {len(successful)}",
        f"- **Failed:** {len(failed)}",
        f"- **Success Rate:** {len(successful)/len(results)*100:.1f}%",
        f"\n## Test Configuration",
        f"- **Durations:** {', '.join(map(str, DURATIONS))} seconds",
        f"- **FPS:** {FPS}",
        f"- **Number of Prompts:** {len(TEST_PROMPTS)}",
        f"- **Number of Negative Prompts:** {len(NEGATIVE_PROMPTS)}",
        f"- **Total Tests:** {len(DURATIONS)} durations × {len(TEST_PROMPTS)} prompts × {len(NEGATIVE_PROMPTS)} neg_prompts = {len(DURATIONS) * len(TEST_PROMPTS) * len(NEGATIVE_PROMPTS)} tests",
    ]

    # Results by duration
    report_lines.append("\n## Results by Duration\n")
    for duration in DURATIONS:
        duration_results = [r for r in successful if r['duration'] == duration]
        if duration_results:
            avg_time = sum(r['generation_time'] for r in duration_results) / len(duration_results)
            report_lines.append(f"### {duration}s Videos")
            report_lines.append(f"- Generated: {len(duration_results)}")
            report_lines.append(f"- Avg Generation Time: {avg_time:.2f}s")
            report_lines.append(f"- Avg Time per Second: {avg_time/duration:.2f}s\n")

    # Results by prompt
    report_lines.append("\n## Results by Prompt\n")
    for prompt_data in TEST_PROMPTS:
        prompt_id = prompt_data['id']
        prompt_results = [r for r in successful if r.get('prompt_id') == prompt_id]
        if prompt_results:
            avg_time = sum(r['generation_time'] for r in prompt_results) / len(prompt_results)
            report_lines.append(f"### {prompt_data['description']}")
            report_lines.append(f"- Prompt ID: `{prompt_id}`")
            report_lines.append(f"- Tests Completed: {len(prompt_results)}")
            report_lines.append(f"- Avg Generation Time: {avg_time:.2f}s\n")

    # Results by negative prompt
    report_lines.append("\n## Results by Negative Prompt\n")
    for neg_prompt_data in NEGATIVE_PROMPTS:
        neg_prompt_id = neg_prompt_data['id']
        neg_prompt_results = [r for r in successful if r.get('negative_prompt_id') == neg_prompt_id]
        if neg_prompt_results:
            avg_time = sum(r['generation_time'] for r in neg_prompt_results) / len(neg_prompt_results)
            report_lines.append(f"### {neg_prompt_data['description']}")
            report_lines.append(f"- Negative Prompt ID: `{neg_prompt_id}`")
            report_lines.append(f"- Tests Completed: {len(neg_prompt_results)}")
            report_lines.append(f"- Avg Generation Time: {avg_time:.2f}s\n")

    # Detailed results table
    report_lines.append("\n## Detailed Results\n")
    report_lines.append("| Test ID | Duration | Description | Neg Prompt | Status | Gen Time (s) | Peak GPU (GB) |")
    report_lines.append("|---------|----------|-------------|------------|--------|--------------|---------------|")

    for r in results:
        description = r.get('description', 'N/A')
        neg_prompt_desc = r.get('negative_prompt_description', 'N/A')
        gpu_mem = r.get('stats', {}).get('peak_memory_usage_gb', 'N/A') if r['status'] == 'success' else 'N/A'
        if isinstance(gpu_mem, float):
            gpu_mem = f"{gpu_mem:.2f}"

        report_lines.append(
            f"| {r['test_id']} | {r['duration']}s | {description} | {neg_prompt_desc} | "
            f"{r['status']} | {r['generation_time']:.2f} | {gpu_mem} |"
        )

    # Failed tests
    if failed:
        report_lines.append("\n## Failed Tests\n")
        for r in failed:
            report_lines.append(f"- **{r['test_id']}**: {r.get('error', 'Unknown error')}")

    # Prompts used
    report_lines.append("\n## Test Prompts\n")
    report_lines.append("Each prompt contains three elements: **Object + Scene + Action**\n")
    for idx, prompt_data in enumerate(TEST_PROMPTS, 1):
        report_lines.append(f"### Prompt {idx}: {prompt_data['description']}")
        report_lines.append(f"- **ID**: `{prompt_data['id']}`")
        report_lines.append(f"- **Full Prompt**: {prompt_data['prompt']}")
        report_lines.append("")

    # Negative prompts used
    report_lines.append("\n## Negative Prompts\n")
    report_lines.append("Three different negative prompt strategies were tested:\n")
    for idx, neg_prompt_data in enumerate(NEGATIVE_PROMPTS, 1):
        report_lines.append(f"### Negative Prompt {idx}: {neg_prompt_data['description']}")
        report_lines.append(f"- **ID**: `{neg_prompt_data['id']}`")
        report_lines.append(f"- **Full Negative Prompt**: {neg_prompt_data['prompt'].strip()}")
        report_lines.append("")

    # Save report
    report_path = exp_dir / "REPORT.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n📊 Summary report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive video generation experiments")
    parser.add_argument("--model_name", type=str, default="wan_2_1",
                       choices=["wan_2_1", "wan_2_2"],
                       help="Model to use: 'wan_2_1' (default) or 'wan_2_2'")
    parser.add_argument("--quantize", action="store_true",
                       help="Use 4-bit quantization (only for wan_2_1)")
    args = parser.parse_args()

    try:
        results, exp_dir = run_experiments(model_name=args.model_name, quantized=args.quantize)

        # Print quick summary
        successful = [r for r in results if r['status'] == 'success']
        print(f"\n📈 Quick Stats:")
        print(f"   Success: {len(successful)}/{len(results)}")
        if successful:
            avg_time = sum(r['generation_time'] for r in successful) / len(successful)
            print(f"   Avg Generation Time: {avg_time:.2f}s")

    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
