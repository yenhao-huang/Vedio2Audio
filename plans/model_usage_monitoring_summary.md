# Model Usage Monitoring Implementation Summary

**Date:** 2025-10-31
**File:** `utils/text2vedio.py`
**Purpose:** Add comprehensive monitoring of model resource usage and performance metrics

---

## Overview

Enhanced the `Text2Vedio` class with comprehensive monitoring capabilities to track model initialization, inference performance, memory usage, and GPU utilization throughout the video generation pipeline.

---

## Features Added

### 1. Memory Usage Tracking

#### GPU Memory Monitoring
- **Total GPU memory** detection and logging
- **Allocated memory** tracking in GB
- **Reserved memory** tracking in GB
- **Free memory** calculation and display
- **Peak memory usage** tracking across the entire session

#### CPU Memory Monitoring
- Process memory tracking using `psutil` for CPU-only environments
- Fallback support when GPU is not available

#### Monitoring Points
Memory is logged at the following stages:
- After loading text encoder
- After loading VAE
- After loading transformer
- After initializing pipeline
- Before inference
- After inference

**Implementation:** `_log_memory_usage(stage)` method (lines 96-119)

---

### 2. Performance Metrics

#### Initialization Time
- Tracks total time to load all model components
- Logged both to file and console
- Stored in `self.metrics['initialization_time']`

#### Inference Time Tracking
- **Per-video inference time:** Each generation is timed individually
- **Average inference time:** Calculated across all generated videos
- **Total inference time:** Cumulative time for all inferences
- All timing data stored in `self.metrics['inference_times']` list

#### Video Generation Counter
- Tracks total number of videos generated in session
- Stored in `self.metrics['total_videos_generated']`

---

### 3. GPU Utilization Monitoring

**Implementation:** `_get_gpu_utilization()` method (lines 121-140)

- Monitors GPU utilization percentage using `pynvml` library
- Captures utilization before and after inference
- Graceful fallback if `pynvml` is not installed
- Logs warnings instead of crashing when unavailable

**Requirements:**
- Optional dependency: `pynvml` (nvidia-ml-py)
- Only works with NVIDIA GPUs

---

### 4. Statistics Reporting

#### `get_usage_statistics()` Method (lines 142-160)
Returns a dictionary containing:
- `initialization_time`: Time to load model (seconds)
- `total_videos_generated`: Count of videos created
- `peak_memory_usage_gb`: Maximum memory used (GB)
- `average_inference_time`: Mean inference time (seconds)
- `total_inference_time`: Sum of all inference times (seconds)
- `current_memory_allocated_gb`: Current GPU memory allocated
- `current_memory_reserved_gb`: Current GPU memory reserved
- `device_name`: GPU device name

#### `log_usage_statistics()` Method (lines 162-190)
Displays formatted statistics to both console and log file:

```
============================================================
📊 MODEL USAGE STATISTICS
============================================================
Initialization Time: X.XX s
Total Videos Generated: X
Peak Memory Usage: X.XX GB
Average Inference Time: X.XX s
Total Inference Time: X.XX s
Current GPU Memory Allocated: X.XX GB
Current GPU Memory Reserved: X.XX GB
Device: [GPU Name]
============================================================
```

Called automatically at the end of `main()` (line 276)

---

## Code Changes Summary

### New Imports (lines 10-11)
```python
import time       # For timing measurements
import psutil     # For CPU memory monitoring
```

### New Instance Variables (lines 28-34)
```python
self.metrics = {
    'initialization_time': 0,
    'inference_times': [],
    'peak_memory_usage': 0,
    'total_videos_generated': 0
}
```

### New Methods
1. `_log_memory_usage(stage)` - Memory logging helper
2. `_get_gpu_utilization()` - GPU utilization checker
3. `get_usage_statistics()` - Statistics getter
4. `log_usage_statistics()` - Statistics formatter and logger

### Modified Methods

#### `__init__()` (lines 18-94)
- Added initialization timing
- Added total GPU memory detection
- Added memory logging after each component load
- Records initialization time at completion

#### `text_to_video()` (lines 192-247)
- Added memory logging before/after inference
- Added GPU utilization monitoring
- Added inference timing
- Updates metrics counters
- Logs all performance data

#### `main()` (lines 257-277)
- Added call to `log_usage_statistics()` at end

---

## Usage Example

### Basic Usage (no code changes needed)
```bash
python utils/text2vedio.py --text "your prompt" --output_dir results
```

Statistics are automatically displayed at the end and logged to `text2vedio.log`.

### Programmatic Access to Statistics
```python
from utils.text2vedio import Text2Vedio

t2v = Text2Vedio()
video_frames = t2v.text_to_video("your prompt")
t2v.save_video(video_frames, "output.mp4")

# Get statistics as dictionary
stats = t2v.get_usage_statistics()
print(f"Peak memory: {stats['peak_memory_usage_gb']:.2f} GB")
print(f"Avg inference time: {stats['average_inference_time']:.2f}s")

# Or display formatted statistics
t2v.log_usage_statistics()
```

---

## Logging Details

### Log File
- **Location:** `text2vedio.log` (in working directory)
- **Format:** `%(asctime)s - %(levelname)s - %(message)s`
- **Mode:** Append (`'a'`)
- **Level:** INFO

### What Gets Logged
- Device information (GPU/CPU)
- Memory usage at each pipeline stage
- GPU utilization before/after inference
- Inference timing for each video
- Complete usage statistics at end

---

## Dependencies

### Required (already in project)
- `torch` - For CUDA memory tracking
- `numpy` - For statistics calculations

### New Required
- `psutil` - For CPU memory monitoring

### Optional
- `pynvml` (nvidia-ml-py) - For GPU utilization monitoring
  - Install: `pip install nvidia-ml-py`
  - If not installed, GPU utilization monitoring is skipped with warning

---

## Benefits

1. **Performance Optimization:** Identify memory bottlenecks and optimization opportunities
2. **Resource Planning:** Understand memory requirements for different workloads
3. **Debugging:** Track memory leaks or performance degradation
4. **Capacity Planning:** Determine batch sizes and concurrent operations
5. **Accountability:** Log all operations for audit and analysis
6. **User Feedback:** Real-time progress and resource usage information

---

## Future Enhancements (Optional)

- Add memory usage graphs/plots
- Export statistics to JSON/CSV for analysis
- Add batch processing statistics
- Monitor temperature and power consumption
- Add comparison across multiple runs
- Implement memory profiling for individual components
- Add alerts for memory thresholds

---

## Testing Recommendations

1. **Test with GPU:** Verify all GPU metrics are captured correctly
2. **Test without GPU:** Ensure CPU fallback works properly
3. **Test without pynvml:** Verify graceful degradation of GPU utilization monitoring
4. **Generate multiple videos:** Verify statistics accumulate correctly
5. **Check log file:** Ensure all data is properly logged

---

## Related Files

- **Implementation:** [utils/text2vedio.py](../utils/text2vedio.py)
- **Log Output:** `text2vedio.log`
- **Git Status:** Modified (uncommitted changes)
