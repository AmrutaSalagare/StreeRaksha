# StreeRaksha Multi-Threading Implementation

This document provides a technical overview of the multi-threading architecture implemented in the StreeRaksha system for optimized performance.

## Architecture Overview

The StreeRaksha multi-threading implementation uses a pipeline architecture with the following components:

1. **Frame Capture Thread**: Captures frames from camera and places them in a priority queue
2. **Processing Thread**: Processes frames for object detection, tracking, and pose analysis
3. **Rendering Thread**: Visualizes results and handles UI updates
4. **Firebase Upload Thread**: Handles background uploads of evidence and logs
5. **Thread Health Monitor**: Monitors thread performance and resource usage

## Components

### Priority Queue System

- **PriorityQueue**: Base class implementing a thread-safe priority queue
- **PrioritizedFrameQueue**: Specialized queue that prioritizes frames with motion
- Features:
  - Priority levels (0=highest to 2=lowest)
  - Motion detection for automatic prioritization
  - Thread-safe operations with proper locking
  - Performance statistics tracking

### Thread Health Monitor

- Monitors CPU, memory, and thread performance
- Detects bottlenecks in processing pipeline
- Logs warnings for potential issues
- Collects metrics for analysis and reporting
- Can export health reports for offline analysis

### Configuration Manager

- Dynamic configuration for all thread parameters
- Runtime updates without application restart
- Configuration persistence across sessions
- Change notification through callback system

### Adaptive Processing

- Automatically adjusts processing based on system load
- Skips frames when falling behind while preserving critical frames
- Prioritizes processing for frames with potential security concerns
- Maintains consistent performance across varied hardware

## Performance Optimizations

### 1. Thread Priority Management

- Sets appropriate thread priorities based on importance:
  - Processing thread: ABOVE_NORMAL priority
  - Capture thread: NORMAL priority
  - Rendering thread: NORMAL priority
  - Upload thread: BELOW_NORMAL priority

### 2. Queue Management

- Dynamic queue sizes based on system capabilities
- Automatic adjustment of processing based on queue fullness
- Priority-based scheduling to ensure critical frames are processed first
- Queue size monitoring to detect bottlenecks

### 3. FPS Control

- Configurable maximum FPS to prevent CPU overuse
- Adaptive frame skipping when processing falls behind
- Motion detection to prioritize frames with activity
- Performance metrics collection for bottleneck identification

### 4. Resource Optimization

- Thread health monitoring for CPU and memory usage
- Adaptive processing complexity based on system load
- Periodic logging of performance metrics
- Automatic detection and reporting of resource issues

## Performance Metrics

The following metrics are collected in real-time:

- Frame capture FPS
- Processing FPS
- Rendering FPS
- Queue utilization percentages
- Processing time breakdowns:
  - Detection time
  - Tracking time
  - Pose analysis time
- System resource usage
- Thread health status

## Requirements

- Python 3.8 or higher
- Dependencies:
  - OpenCV
  - NumPy
  - psutil
  - PyYAML
  - msgpack

## Usage

The multi-threading system is automatically initialized when running the main application. For fine-tuning performance, use the `run_streeraksha.py` script with appropriate command-line arguments:

```bash
python run_streeraksha.py --performance high --camera 0
```

Performance presets:

- `low`: For resource-constrained systems (640x480, 15 FPS)
- `medium`: Balanced performance (default)
- `high`: For systems with adequate resources (1280x720, 30 FPS)

## Configuration

The system can be configured through the `config.json` file:

```json
{
  "threading": {
    "frame_queue_size": 30,
    "result_queue_size": 10,
    "upload_queue_size": 100,
    "max_fps": 30
  },
  "camera": {
    "width": 1280,
    "height": 720,
    "fps": 30
  }
}
```

The configuration manager will automatically create an optimized configuration based on system capabilities if one doesn't exist.

## Architecture Diagram

```
                                                     +-------------------+
                                                     |                   |
                                      +------------->| Firebase Manager  |
                                      |              |                   |
                                      |              +-------------------+
                                      |                  |
                                      |                  |
+------------------+     +-----------------+     +----------------+     +-----------------+
|                  |     |                 |     |                |     |                 |
| Frame Capture    |---->| Processing      |---->| Rendering      |     | Thread Health   |
| Thread           |     | Thread          |     | Thread         |     | Monitor         |
|                  |     |                 |     |                |     |                 |
+------------------+     +-----------------+     +----------------+     +-----------------+
        |                        |                      |                       |
        |                        |                      |                       |
        v                        v                      v                       v
  +-----------+          +-----------+           +-----------+          +----------------+
  |           |          |           |           |           |          |                |
  | Frame     |--------->| Result    |---------->| Display   |          | Performance    |
  | Queue     |          | Queue     |           | Output    |          | Metrics        |
  |           |          |           |           |           |          |                |
  +-----------+          +-----------+           +-----------+          +----------------+
```
