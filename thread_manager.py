"""
StreeRaksha Thread Manager Module
Handles multi-threading functionality for improved performance.
"""

import threading
import time
import queue
import cv2
import numpy as np
import psutil
import os
from datetime import datetime

# Import custom modules
from priority_queue import PriorityQueue, PrioritizedFrameQueue


class FrameCapture(threading.Thread):
    """Thread class for capturing frames from camera with enhanced FPS control"""

    def __init__(self, cap, frame_queue, stop_event, logger, max_queue_size=30, max_fps=0):
        """
        Initialize the frame capture thread

        Parameters:
            cap: OpenCV video capture object
            frame_queue: Queue for frames (should be PrioritizedFrameQueue for best performance)
            stop_event: Threading event for stopping
            logger: Logger object
            max_queue_size: Maximum queue size
            max_fps: Maximum frames per second (0 = unlimited)
        """
        super().__init__(daemon=True)
        self.name = "FrameCapture"
        self.cap = cap
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.logger = logger
        self.max_queue_size = max_queue_size
        self.max_fps = max_fps
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = 0

        # Set camera properties if available
        try:
            # Try to set camera to highest resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Try to set maximum FPS if supported by camera
            if self.max_fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        except Exception as e:
            self.logger.warning(f"Failed to set camera properties: {e}")

        # Read actual camera properties
        try:
            self.camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.logger.info(
                f"Camera configured at {self.camera_width}x{self.camera_height} @ {self.camera_fps}fps")
        except:
            self.camera_width = 0
            self.camera_height = 0
            self.camera_fps = 0

    def set_max_fps(self, max_fps):
        """Set maximum FPS (0 = unlimited)"""
        self.max_fps = max_fps

        # Try to set camera FPS if possible
        if max_fps > 0:
            try:
                self.cap.set(cv2.CAP_PROP_FPS, max_fps)
            except:
                pass  # Some cameras don't support setting FPS

    def run(self):
        """Main method for frame capture thread with optimized ultra-low latency mode"""
        self.logger.info("Frame capture thread started")
        last_log_time = time.time()
        frame_interval = 0 if self.max_fps <= 0 else 1.0 / self.max_fps
        skip_counter = 0  # Counter to track consecutive skips

        # For ultra-low latency: check for zero-copy mode setting
        config = getattr(self, 'config', {})
        processing_config = config.get('processing', {})
        zero_copy_mode = processing_config.get('zero_copy_mode', False)
        is_ultra_low = self.max_queue_size == 1

        # Ensure we get frames as fast as possible in ultra-low mode
        if is_ultra_low:
            try:
                # Request max camera FPS to minimize latency
                self.cap.set(cv2.CAP_PROP_FPS, 60)
                # Minimize buffer size to get freshest frame
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as e:
                self.logger.warning(f"Could not optimize camera settings: {e}")

        while not self.stop_event.is_set():
            current_time = time.time()

            # Optimized FPS limiting with reduced sleep time
            if frame_interval > 0:
                time_since_last_frame = current_time - self.last_frame_time
                if time_since_last_frame < frame_interval:
                    # Wait until it's time for the next frame
                    sleep_time = frame_interval - time_since_last_frame
                    # Use much shorter sleep in ultra-low mode (0.001s instead of 0.005s)
                    max_sleep = 0.001 if is_ultra_low else 0.005
                    time.sleep(min(sleep_time, max_sleep))
                    continue

            # Check queue size to decide frame handling strategy
            queue_size = self.frame_queue.qsize()
            queue_ratio = queue_size / max(1, self.max_queue_size)

            # Ultra-low latency mode - aggressive frame skipping
            if is_ultra_low:
                # First, always drain camera buffer to get freshest frame possible
                while self.cap.grab():
                    # Keep grabbing frames to empty the buffer until no more available
                    # This gets us the most recent frame with lowest latency
                    pass

                # In direct transfer mode, only capture when queue is empty
                if queue_size > 0:
                    # No sleep - just loop back and check stop event
                    continue
            # Normal strategy for larger queues
            elif queue_ratio > 0.5:  # Lower threshold for more aggressive skipping
                skip_counter += 1
                # More aggressive skipping formula - skip more frames as queue fills up
                # Even more aggressive formula (was 3)
                if skip_counter < int(queue_ratio * 4):
                    # Drain one frame from camera to keep timestamps current
                    self.cap.grab()
                    continue
                else:
                    skip_counter = 0  # Reset counter to allow a frame through
            else:
                skip_counter = 0  # Reset counter when queue is not full

            # Check if we should be reading a frame
            if queue_size < self.max_queue_size:
                # Read frame from camera with optimized handling
                ret, frame = self.cap.read()

                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    # Shorter recovery time in ultra-low mode
                    time.sleep(0.1 if is_ultra_low else 0.5)
                    continue

                timestamp = datetime.now()
                self.frame_count += 1
                self.last_frame_time = time.time()

                # Calculate FPS every second
                if current_time - last_log_time >= 1.0:
                    self.fps = self.frame_count / \
                        (current_time - self.start_time)
                    last_log_time = current_time

                # Special handling for ultra-low latency mode
                if is_ultra_low:
                    # In ultra-low mode, try to clear the queue first to ensure freshest frame
                    try:
                        # Drain the queue first if possible
                        while not self.frame_queue.empty():
                            if isinstance(self.frame_queue, PrioritizedFrameQueue):
                                _ = self.frame_queue.get_nowait()
                            else:
                                _ = self.frame_queue.get(block=False)
                            self.frame_queue.task_done()
                    except (queue.Empty, AttributeError):
                        pass

                    # Zero-copy mode for minimal latency - don't copy frame unless needed
                    if zero_copy_mode:
                        try:
                            if isinstance(self.frame_queue, PrioritizedFrameQueue):
                                self.frame_queue.put_frame(
                                    frame, timestamp, self.frame_count, block=False)
                            else:
                                self.frame_queue.put(
                                    (frame, timestamp, self.frame_count), block=False)
                        except queue.Full:
                            pass
                    else:
                        # Regular mode with frame copy
                        try:
                            if isinstance(self.frame_queue, PrioritizedFrameQueue):
                                self.frame_queue.put_frame(
                                    frame.copy(), timestamp, self.frame_count, block=False)
                            else:
                                self.frame_queue.put(
                                    (frame.copy(), timestamp, self.frame_count), block=False)
                        except queue.Full:
                            pass
                else:
                    # Regular (non-ultra-low) mode handling
                    try:
                        if isinstance(self.frame_queue, PrioritizedFrameQueue):
                            # Use priority queue's specialized frame insertion
                            self.frame_queue.put_frame(
                                frame.copy(), timestamp, self.frame_count, block=False)
                        else:
                            # Fallback for regular queue
                            self.frame_queue.put(
                                (frame.copy(), timestamp, self.frame_count), block=False)
                    except queue.Full:
                        pass
            else:
                # If queue is full, sleep to avoid CPU overuse but for much less time in ultra-low mode
                time.sleep(0.001 if is_ultra_low else 0.005)

        self.logger.info("Frame capture thread stopped")


class ProcessingThread(threading.Thread):
    """Thread class for processing frames (detection, tracking, etc.) with optimized performance"""

    def __init__(self, frame_queue, result_queue, stop_event, model, tracker, gender_detector,
                 pose_analyzer, logger, max_queue_size=10):
        """Initialize the processing thread"""
        super().__init__(daemon=True)
        self.name = "ProcessingThread"
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.model = model
        self.tracker = tracker
        self.gender_detector = gender_detector
        self.pose_analyzer = pose_analyzer
        self.logger = logger
        self.max_queue_size = max_queue_size
        self.fps = 0
        self.process_count = 0
        self.start_time = time.time()

        # Performance metrics
        self.performance_metrics = {
            'detection_time': 0.0,
            'tracking_time': 0.0,
            'pose_time': 0.0,
            'total_time': 0.0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'last_processing_times': []  # Last 10 processing times
        }

        # Adaptive processing flag - will be set if we're falling behind
        self.adaptive_mode = False
        self.adaptive_skip_count = 0

    def run(self):
        """Main method for processing thread with adaptive processing"""
        self.logger.info("Processing thread started")
        last_log_time = time.time()
        last_metrics_time = time.time()
        last_frame_count = 0

        # For adaptive processing - more aggressive settings
        skip_frame_threshold = 3  # Lower threshold to engage adaptive mode sooner
        max_skip_ratio = 0.7  # Increase maximum skip ratio to catch up faster

        # Frame age tracking for staleness detection
        frame_age_threshold = 0.5  # Seconds before a frame is considered stale
        last_processed_timestamp = time.time()

        while not self.stop_event.is_set():
            try:
                # Check if we're falling behind
                queue_size = self.frame_queue.qsize()
                if queue_size > skip_frame_threshold:
                    # Enable adaptive mode when queue is filling up
                    self.adaptive_mode = True
                elif queue_size <= 1:
                    # Disable adaptive mode when queue is mostly empty
                    self.adaptive_mode = False

                # Get frame from queue with a timeout
                # If using priority queue, extract the data from the wrapper
                frame_data = self.frame_queue.get(timeout=0.5)

                if isinstance(self.frame_queue, PrioritizedFrameQueue) and isinstance(frame_data, dict):
                    # Extract from our specialized frame queue
                    frame = frame_data['frame']
                    timestamp = frame_data['timestamp']
                    frame_count = frame_data['frame_count']
                    motion_score = frame_data.get('motion_score', 0)
                else:
                    # Standard queue format
                    frame, timestamp, frame_count = frame_data
                    motion_score = 0

                # Calculate frame age (how long it's been in the queue)
                frame_age = time.time() - timestamp.timestamp()

                # Check if frame is stale - use much stricter threshold for ultralow mode
                # Make ultra-low mode reject frames after just 0.05s (down from 0.1s)
                frame_age_threshold_actual = 0.05 if self.max_queue_size == 1 else frame_age_threshold
                is_stale = frame_age > frame_age_threshold_actual

                # Get processing flags
                config = getattr(self, 'config', {})
                processing_config = config.get('processing', {})

                # Check for ultra-low latency mode (queue size == 1)
                if self.max_queue_size == 1:
                    # In ultra-low latency mode, process minimally for speed
                    skip_gender_detection = True
                    skip_pose_estimation = True
                    minimal_tracking = True
                    early_exit_detection = processing_config.get(
                        'early_exit_detection', True)
                    max_detections = processing_config.get('max_detections', 1)
                    zero_copy_mode = processing_config.get(
                        'zero_copy_mode', False)
                else:
                    # Get processing flags from configuration
                    skip_gender_detection = processing_config.get(
                        'skip_gender_detection', False)
                    skip_pose_estimation = processing_config.get(
                        'skip_pose_estimation', False)
                    minimal_tracking = processing_config.get(
                        'minimal_tracking', False)
                    early_exit_detection = False
                    max_detections = None
                    zero_copy_mode = False

                # More aggressive frame skipping in adaptive mode
                # Skip frames that are stale or have low motion, unless it's been too many skips
                if self.adaptive_mode and self.max_queue_size > 1:  # Don't skip in ultra-low mode
                    skip_this_frame = False

                    # Skip stale frames regardless of motion
                    if is_stale:
                        skip_this_frame = True
                    # Skip low motion frames if we haven't skipped too many
                    elif motion_score < 0.15 and self.adaptive_skip_count < max_skip_ratio * queue_size:
                        skip_this_frame = True
                    # Skip frames to maintain frame age freshness - more aggressive (was 2)
                    # Process every 3rd frame
                    elif queue_size > 5 and (frame_count % 3) == 0:
                        skip_this_frame = True
                    # Additional ultra-aggressive skipping when queue is very large
                    elif queue_size > 8:
                        # Only process key frames (e.g. 1 in 5) when queue is very full
                        skip_this_frame = (frame_count % 5) != 0

                    if skip_this_frame:
                        self.adaptive_skip_count += 1
                        self.performance_metrics['skipped_frames'] += 1
                        self.frame_queue.task_done()
                        continue

                # Ultra-low mode - special handling for frame freshness
                if self.max_queue_size == 1 and queue_size > 0:
                    # In ultra-low mode, if there are more frames in queue after this one,
                    # skip this frame to process the freshest one
                    try:
                        self.adaptive_skip_count += 1
                        self.performance_metrics['skipped_frames'] += 1
                        self.frame_queue.task_done()
                        continue
                    except:
                        pass

                # Not skipped - reset skip count and process this frame
                last_processed_timestamp = time.time()
                self.adaptive_skip_count = 0

                # Time the processing steps
                start_time = time.time()

                # 1. Object Detection with early exit optimization
                detection_start = time.time()
                # Use even higher confidence threshold in ultra-low latency mode
                conf_threshold = 0.75 if self.max_queue_size == 1 else 0.5

                # In ultra-low latency mode with early exit, use a specialized approach
                if early_exit_detection and self.max_queue_size == 1:
                    # First try with very high confidence to get quick results
                    results = self.model(frame, conf=0.85, classes=0)
                    # If we got a high-confidence detection, use it
                    if any(len(r.boxes) > 0 for r in results):
                        # Great! We got a high confidence detection quickly
                        pass
                    else:
                        # No high-confidence detection, try again with normal threshold
                        results = self.model(
                            frame, conf=conf_threshold, classes=0)
                else:
                    # Standard detection for normal modes
                    results = self.model(frame, conf=conf_threshold, classes=0)

                detection_time = time.time() - detection_start

                # Extract bounding boxes with optimization for ultra-low latency
                detections = []
                for r in results:
                    boxes = r.boxes
                    # Further limit detections in ultra-low latency
                    max_detections_count = max_detections if max_detections else (
                        1 if self.max_queue_size == 1 else len(boxes))

                    # In ultra-low mode, optimize by sorting boxes by size and only processing largest
                    if self.max_queue_size == 1 and len(boxes) > 1:
                        # Sort boxes by area (largest first) to prioritize closest people
                        areas = []
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu(
                            ).numpy().astype(int)
                            area = (x2 - x1) * (y2 - y1)
                            areas.append((area, box))
                        areas.sort(reverse=True)  # Sort by area, largest first

                        # Process only the largest boxes
                        for i, (_, box) in enumerate(areas):
                            if i >= max_detections_count:
                                break
                            x1, y1, x2, y2 = box.xyxy[0].cpu(
                            ).numpy().astype(int)
                            detections.append((x1, y1, x2, y2))
                    else:
                        # Standard processing for normal mode
                        for i, box in enumerate(boxes):
                            if i >= max_detections_count:
                                break
                            x1, y1, x2, y2 = box.xyxy[0].cpu(
                            ).numpy().astype(int)
                            detections.append((x1, y1, x2, y2))

                # 2. Tracking with ultra-low latency optimizations
                tracking_start = time.time()

                # In ultra-low mode with zero-copy, avoid validation completely
                if self.max_queue_size == 1 and zero_copy_mode:
                    # Absolute minimal tracking - just keep the boxes
                    # Take only the first (largest) detection
                    valid_detections = detections[:1]
                    filtered_detections = valid_detections

                    # Create simplified persons with minimal data
                    persons = []
                    for i, bbox in enumerate(filtered_detections):
                        person = {
                            'track_id': 1,  # Assign fixed ID in ultra-low mode
                            'bbox': bbox,
                            'gender': 'Unknown',
                            'gender_confidence': 0.0,
                            'age': 0
                        }
                        persons.append(person)
                else:
                    # Use simpler validation in minimal tracking mode
                    if minimal_tracking:
                        valid_detections = detections  # Skip validation for speed
                    else:
                        valid_detections = self.tracker.filter_valid_detections(
                            detections)

                    # Apply non-maximum suppression - less aggressive in ultra-low latency
                    # Higher threshold = fewer detections
                    iou_threshold = 0.6 if self.max_queue_size == 1 else 0.4
                    filtered_detections = self.tracker.non_max_suppression(
                        valid_detections, iou_threshold=iou_threshold)

                    # Update person trackers with ultra-low latency optimization
                    if skip_gender_detection:
                        # Skip gender detection in ultra-low latency mode
                        persons = self.tracker.update_trackers(
                            filtered_detections, frame, None)
                    else:
                        persons = self.tracker.update_trackers(
                            filtered_detections, frame, self.gender_detector)

                tracking_time = time.time() - tracking_start

                # 3. Pose Analysis
                pose_start = time.time()

                # Skip pose analysis entirely in ultra-low latency mode
                if skip_pose_estimation:
                    pose_time = 0.0
                    # Set empty pose fields so downstream code doesn't break
                    for person in persons:
                        person['pose'] = {}
                        person['pose_landmarks'] = []
                else:
                    # Process pose for each person - in adaptive mode, only process the most important ones
                    important_persons = persons

                    # In ultra-low latency or adaptive mode, limit the number of people we analyze
                    if (self.max_queue_size == 1 or self.adaptive_mode) and len(persons) > 2:
                        # In ultra-low latency mode, just process the largest person (closest to camera)
                        if self.max_queue_size == 1:
                            persons.sort(key=lambda p: (
                                p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]), reverse=True)
                            important_persons = persons[:1]
                        # In adaptive mode, prioritize females and the nearest males
                        else:
                            females = [p for p in persons if p.get(
                                'gender') == "Female"]
                            males = [p for p in persons if p.get(
                                'gender') == "Male"]

                            # Sort males by size (closest to camera)
                            males.sort(key=lambda p: (
                                p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]), reverse=True)

                            # Process all females and only the nearest males
                            # Only 1 male in adaptive mode
                            important_persons = females + \
                                males[:min(1, len(males))]

                    for person in important_persons:
                        # Extract person data
                        bbox = person['bbox']

                        # Detect pose
                        pose_landmarks, pose_analysis = self.pose_analyzer.detect_pose(
                            frame, bbox)

                        if pose_analysis:
                            person['pose'] = pose_analysis

                        # Store pose landmarks for visualization
                        person['pose_landmarks'] = pose_landmarks

                    # Set empty pose for unprocessed persons
                    for person in persons:
                        if person not in important_persons:
                            person['pose'] = {}
                            person['pose_landmarks'] = []

                    pose_time = time.time() - pose_start

                # Update performance metrics
                self.performance_metrics['detection_time'] = detection_time
                self.performance_metrics['tracking_time'] = tracking_time
                self.performance_metrics['pose_time'] = pose_time

                total_time = time.time() - start_time
                self.performance_metrics['total_time'] = total_time
                self.performance_metrics['processed_frames'] += 1

                # Keep last 10 processing times for analytics
                self.performance_metrics['last_processing_times'].append(
                    total_time)
                if len(self.performance_metrics['last_processing_times']) > 10:
                    self.performance_metrics['last_processing_times'].pop(0)

                # Calculate processing FPS
                self.process_count += 1
                current_time = time.time()

                if current_time - last_log_time >= 1.0:
                    self.fps = self.process_count / \
                        (current_time - self.start_time)
                    last_log_time = current_time

                    # Log detailed metrics every 10 seconds
                    if current_time - last_metrics_time >= 10.0:
                        frames_since_last = self.process_count - last_frame_count
                        if frames_since_last > 0:
                            avg_detection = self.performance_metrics['detection_time']
                            avg_tracking = self.performance_metrics['tracking_time']
                            avg_pose = self.performance_metrics['pose_time']
                            avg_total = self.performance_metrics['total_time']

                            self.logger.debug(
                                f"Processing metrics: {self.fps:.1f} FPS, "
                                f"Detection: {avg_detection*1000:.1f}ms, "
                                f"Tracking: {avg_tracking*1000:.1f}ms, "
                                f"Pose: {avg_pose*1000:.1f}ms, "
                                f"Total: {avg_total*1000:.1f}ms"
                            )

                            last_metrics_time = current_time
                            last_frame_count = self.process_count

                # Ultra-low latency: Clear the queue to always show the latest frame
                if self.max_queue_size == 1:
                    # In ultra-low latency mode, always clear the queue before adding new frame
                    try:
                        while self.result_queue.qsize() > 0:
                            try:
                                self.result_queue.get_nowait()  # Clear any pending frames
                                self.result_queue.task_done()
                            except:
                                break
                    except:
                        pass

                # Put results in output queue with priority
                # Give higher priority to frames with people
                priority = 0 if len(persons) > 0 else 1

                # In ultra-low latency mode, always try to put frame regardless of queue size
                should_put_frame = (self.result_queue.qsize() < self.max_queue_size) or (
                    self.max_queue_size == 1)
                if should_put_frame:
                    if hasattr(self.result_queue, 'put') and callable(getattr(self.result_queue, 'put')):
                        try:
                            # If using priority queue
                            if isinstance(self.result_queue, PriorityQueue):
                                self.result_queue.put(
                                    (frame, timestamp, frame_count,
                                     persons, detections),
                                    priority=priority,
                                    block=False
                                )
                            else:
                                # Standard queue - use put_nowait for ultra-low latency mode
                                if self.max_queue_size == 1:
                                    self.result_queue.put_nowait(
                                        (frame, timestamp, frame_count, persons, detections))
                                else:
                                    self.result_queue.put(
                                        (frame, timestamp, frame_count, persons, detections), block=False)
                        except queue.Full:
                            pass  # Skip if queue is full

                # Mark task as done
                self.frame_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing thread: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                if self.frame_queue.qsize() > 0:
                    self.frame_queue.task_done()  # Make sure to mark as done even on error

        self.logger.info("Processing thread stopped")


class RenderingThread(threading.Thread):
    """Thread class for rendering and displaying frames"""

    def __init__(self, result_queue, stop_event, alert_system, visualizer, debugger, logger):
        """Initialize the rendering thread"""
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.alert_system = alert_system
        self.visualizer = visualizer
        self.debugger = debugger
        self.logger = logger
        self.last_alert_time = 0
        self.fps = 0
        self.render_count = 0
        self.start_time = time.time()

    def run(self):
        """Main method for rendering thread with ultra-low latency optimizations"""
        self.logger.info("Rendering thread started")
        last_log_time = time.time()
        frame_latency = 0  # Track frame latency
        render_frame_count = 0  # Track rendered frames for FPS calculation

        while not self.stop_event.is_set():
            try:
                # Check if this is an ultra-low latency configuration
                is_ultra_low = hasattr(
                    self.result_queue, 'maxsize') and self.result_queue.maxsize == 1

                # In ultra-low latency mode, try to get newest frame by draining old ones first
                if is_ultra_low:
                    # First, try to drain queue to get to newest frame
                    newest_frame_data = None
                    try:
                        while True:
                            # Keep getting frames until queue is empty
                            frame_data = self.result_queue.get_nowait()
                            if newest_frame_data:
                                # Mark previous frame as done since we're skipping it
                                self.result_queue.task_done()
                            newest_frame_data = frame_data
                    except (queue.Empty, AttributeError):
                        # Queue is empty or doesn't support get_nowait
                        if not newest_frame_data:
                            # If we didn't get any frame, try normal get with minimal timeout
                            newest_frame_data = self.result_queue.get(
                                timeout=0.001)
                        # Process the newest frame we got
                        frame, timestamp, frame_count, persons, detections = newest_frame_data
                else:
                    # Normal mode - just get the next frame with standard timeout
                    timeout = 0.5
                    frame, timestamp, frame_count, persons, detections = self.result_queue.get(
                        timeout=timeout)

                # Calculate and track latency (time from capture to display)
                current_time = time.time()
                frame_latency = current_time - timestamp.timestamp()
                render_frame_count += 1

                # Get processing configuration to determine visualization level
                config = getattr(self, 'config', {})
                processing_config = config.get('processing', {})
                minimal_visualization = is_ultra_low and processing_config.get(
                    'minimal_visualization', False)

                if minimal_visualization:
                    # Ultra-minimal visualization for maximum speed
                    # Just draw basic boxes and latency indicator

                    # Simplified gender classification
                    females = [p for p in persons if p.get(
                        'gender') == "Female"]
                    males = [p for p in persons if p.get('gender') == "Male"]

                    # Skip risk calculation in ultra-low mode

                    # No pose drawing in minimal mode

                    # Basic detection boxes only
                    frame = self.visualizer.draw_detection_boxes(
                        frame, persons)

                    # No risk indicators in minimal mode

                    # Just basic stats
                    hour = timestamp.hour
                    is_night = hour >= 18 or hour < 6

                    # Draw latency directly on frame
                    latency_text = f"Latency: {frame_latency*1000:.1f}ms"
                    cv2.putText(frame, latency_text, (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # Standard visualization for normal modes
                    hour = timestamp.hour
                    is_night = hour >= 18 or hour < 6
                    females = [p for p in persons if p.get(
                        'gender') == "Female"]
                    males = [p for p in persons if p.get('gender') == "Male"]

                    # Process each female to calculate risk
                    for female in females:
                        risk_score = self.alert_system.calculate_risk_score(
                            female, females, males, is_night, persons)
                        female['risk_score'] = risk_score

                    # Detect alerts
                    alert_detected, alert_message = self.alert_system.detect_alerts(
                        persons, timestamp)

                    # Draw poses for each person
                    for person in persons:
                        if 'pose_landmarks' in person and person['pose_landmarks']:
                            frame = self.visualizer.pose_analyzer.draw_pose(
                                frame, person['pose_landmarks'])

                    # Draw detection boxes
                    frame = self.visualizer.draw_detection_boxes(
                        frame, persons)

                    # Add risk indicators
                    frame = self.alert_system.add_risk_indicators(
                        frame, females, males, is_night)

                    # Draw statistics with latency info
                    frame = self.visualizer.draw_stats(
                        frame, frame_count, persons, timestamp, latency=frame_latency)

                # Handle alerts (minimal or full)
                if not minimal_visualization:
                    if alert_detected and time.time() - self.last_alert_time > self.alert_system.alert_config['ALERT_COOLDOWN']:
                        self.last_alert_time = time.time()
                        # Draw alert
                        frame = self.visualizer.draw_alert(
                            frame, alert_message)

                        # Save evidence with alert details
                        evidence_path = self.alert_system.save_evidence(
                            frame, alert_message, "ALERT")
                        self.logger.warning(
                            f"Alert detected: {alert_message}. Evidence saved to {evidence_path}")
                    elif alert_detected:
                        # Show alert but with cooldown
                        frame = self.visualizer.draw_alert(
                            frame, alert_message)

                # Add debug overlay with latency info
                debug_info = {
                    "Capture FPS": f"{getattr(self, '_capture_fps', 0):.2f}",
                    "Process FPS": f"{getattr(self, '_process_fps', 0):.2f}",
                    "Render FPS": f"{self.fps:.2f}",
                    # Convert to milliseconds
                    "Latency": f"{frame_latency*1000:.1f}ms",
                    "Detections": len(detections),
                    "Valid Tracks": len(persons),
                    "Frame Queue": getattr(self, '_frame_q_size', 0),
                    "Result Queue": getattr(self, '_result_q_size', 0),
                    "Night Mode": str(is_night)
                }

                # In minimal visualization mode, use a stripped-down debug overlay
                if minimal_visualization:
                    # Only show critical performance metrics
                    minimal_debug = {
                        "FPS": f"{self.fps:.1f}",
                        "Latency": f"{frame_latency*1000:.1f}ms"
                    }
                    frame = self.debugger.add_debug_overlay(
                        frame, minimal_debug, position="top-right")
                else:
                    # Full debug overlay
                    frame = self.debugger.add_debug_overlay(frame, debug_info)

                # Show the result - optimize for speed in ultra-low latency mode
                if is_ultra_low:
                    # Use WINDOW_AUTOSIZE for faster rendering in ultra-low mode
                    cv2.namedWindow(
                        'StreeRaksha Safety Monitoring', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('StreeRaksha Safety Monitoring', frame)
                    # Use shortest possible wait key time in ultra-low latency mode
                    cv2.waitKey(1)
                else:
                    # Normal display for other modes
                    cv2.imshow('StreeRaksha Safety Monitoring', frame)

                # Calculate rendering FPS
                self.render_count += 1
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    self.fps = self.render_count / \
                        (current_time - self.start_time)
                    last_log_time = current_time

                # Mark task as done
                self.result_queue.task_done()

                # Check for exit keys
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'):  # ESC or 'q' to exit
                    self.logger.info("User requested exit")
                    self.stop_event.set()

            except queue.Empty:
                # No new frame available, check if we should exit
                if self.stop_event.is_set():
                    break
            except Exception as e:
                self.logger.error(f"Error in rendering thread: {e}")
                if self.result_queue.qsize() > 0:
                    self.result_queue.task_done()

        self.logger.info("Rendering thread stopped")

    def update_stats(self, capture_fps, process_fps, frame_q_size, result_q_size):
        """Update statistics for debug overlay"""
        self._capture_fps = capture_fps
        self._process_fps = process_fps
        self._frame_q_size = frame_q_size
        self._result_q_size = result_q_size


class ThreadManager:
    """Advanced manager class for coordinating threads with enhanced performance and monitoring"""

    def __init__(self, model, tracker, gender_detector, pose_analyzer,
                 alert_system, visualizer, debugger, logger, firebase_manager=None,
                 config_manager=None):
        """Initialize the thread manager with enhanced monitoring and configuration"""
        self.model = model
        self.tracker = tracker
        self.gender_detector = gender_detector
        self.pose_analyzer = pose_analyzer
        self.alert_system = alert_system
        self.visualizer = visualizer
        self.debugger = debugger
        self.logger = logger
        self.firebase_manager = firebase_manager
        self.config_manager = config_manager

        # Get configuration values if config manager is available
        frame_queue_size = 30
        result_queue_size = 10

        if self.config_manager:
            frame_queue_size = self.config_manager.get(
                'threading.frame_queue_size', frame_queue_size)
            result_queue_size = self.config_manager.get(
                'threading.result_queue_size', result_queue_size)

            # Register for dynamic config updates
            self.config_manager.register_callback(self._handle_config_update, [
                'threading.frame_queue_size',
                'threading.result_queue_size',
                'threading.max_fps'
            ])

        # Create enhanced priority queues for inter-thread communication
        # Enhanced frame buffer with motion priority
        self.frame_queue = PrioritizedFrameQueue(maxsize=frame_queue_size)
        # Results buffer with priority support
        self.result_queue = PriorityQueue(maxsize=result_queue_size)

        # Stop event for signaling threads to terminate
        self.stop_event = threading.Event()

        # Thread synchronization and status
        self.threads_initialized = threading.Event()
        self.thread_status = {
            'capture': False,
            'processing': False,
            'rendering': False,
            'firebase': False,
            'health_monitor': False,
            'config_watcher': False
        }

        # Thread references
        self.threads = {
            'capture': None,
            'processing': None,
            'rendering': None,
            'health_monitor': None
        }

        # Performance metrics
        self.metrics = {
            'capture_fps': 0,
            'process_fps': 0,
            'render_fps': 0,
            'frame_latency': 0,  # Time from capture to display
            'queue_sizes': {
                'frame_queue': 0,
                'result_queue': 0,
                'firebase_queue': 0
            }
        }

        # Initialize threads but don't start them yet
        self.capture_thread = None
        self.processing_thread = None
        self.rendering_thread = None

    def start(self, cap):
        """Start all threads with enhanced monitoring and priority-based scheduling"""
        self.logger.info(
            "Starting thread manager with optimized multi-threading")

        # Reset stop event in case this is a restart
        self.stop_event.clear()

        try:
            # Create and start threads
            self.capture_thread = FrameCapture(
                cap, self.frame_queue, self.stop_event, self.logger
            )
            self.threads['capture'] = self.capture_thread

            self.processing_thread = ProcessingThread(
                self.frame_queue, self.result_queue, self.stop_event,
                self.model, self.tracker, self.gender_detector,
                self.pose_analyzer, self.logger
            )
            self.threads['processing'] = self.processing_thread

            self.rendering_thread = RenderingThread(
                self.result_queue, self.stop_event,
                self.alert_system, self.visualizer, self.debugger, self.logger
            )
            self.threads['rendering'] = self.rendering_thread

            # Start Firebase upload thread if manager is available
            if self.firebase_manager and self.firebase_manager.initialized:
                self.firebase_manager.start_upload_thread()
                self.thread_status['firebase'] = True

            # Set thread priorities if on Windows
            try:
                # Get thread priorities from config if available
                thread_priorities = {
                    'capture': psutil.NORMAL_PRIORITY_CLASS,
                    'processing': psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                    'rendering': psutil.NORMAL_PRIORITY_CLASS
                }

                if self.config_manager:
                    # Map text priorities to psutil constants
                    priority_map = {
                        'IDLE': psutil.IDLE_PRIORITY_CLASS,
                        'BELOW_NORMAL': psutil.BELOW_NORMAL_PRIORITY_CLASS,
                        'NORMAL': psutil.NORMAL_PRIORITY_CLASS,
                        'ABOVE_NORMAL': psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                        'HIGH': psutil.HIGH_PRIORITY_CLASS
                    }

                    for thread_name in ['capture', 'processing', 'rendering']:
                        config_key = f"threading.{thread_name}_thread_priority"
                        priority_text = self.config_manager.get(
                            config_key, 'NORMAL')
                        if priority_text in priority_map:
                            thread_priorities[thread_name] = priority_map[priority_text]

                # Now start threads with appropriate priorities
                self.capture_thread.start()
                if os.name == 'nt':  # Windows
                    self._set_thread_priority(
                        self.capture_thread, thread_priorities['capture'])
                self.thread_status['capture'] = True

                self.processing_thread.start()
                if os.name == 'nt':  # Windows
                    self._set_thread_priority(
                        self.processing_thread, thread_priorities['processing'])
                self.thread_status['processing'] = True

                self.rendering_thread.start()
                if os.name == 'nt':  # Windows
                    self._set_thread_priority(
                        self.rendering_thread, thread_priorities['rendering'])
                self.thread_status['rendering'] = True
            except Exception as e:
                # Fall back to normal thread starting if priority setting fails
                self.logger.warning(f"Could not set thread priorities: {e}")

                # Start threads without priority settings
                if not self.thread_status['capture']:
                    self.capture_thread.start()
                    self.thread_status['capture'] = True

                if not self.thread_status['processing']:
                    self.processing_thread.start()
                    self.thread_status['processing'] = True

                if not self.thread_status['rendering']:
                    self.rendering_thread.start()
                    self.thread_status['rendering'] = True

            # Start health monitor if available
            try:
                from thread_health_monitor import ThreadHealthMonitor
                self.health_monitor = ThreadHealthMonitor(self, self.logger)
                self.health_monitor.start()
                self.threads['health_monitor'] = self.health_monitor
                self.thread_status['health_monitor'] = True
                self.logger.info("Thread health monitor started")
            except ImportError:
                self.logger.warning(
                    "ThreadHealthMonitor not available, health monitoring disabled")
            except Exception as e:
                self.logger.error(
                    f"Failed to start thread health monitor: {e}")

            # Start stats update thread
            self._start_stats_update()

            # Signal successful initialization
            self.threads_initialized.set()

            self.logger.info("All threads started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting threads: {e}")
            self.stop()  # Clean up any threads that did start
            return False

    def _set_thread_priority(self, thread, priority):
        """Set thread priority on Windows"""
        if not thread or not thread.is_alive():
            return False

        try:
            # Get thread ID
            thread_id = thread.ident
            if not thread_id:
                return False

            # Get process handle
            p = psutil.Process(os.getpid())

            # Set priority
            p.nice(priority)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to set thread priority: {e}")
            return False

    def stop(self):
        """Stop all threads and clean up with proper resource management"""
        self.logger.info("Stopping thread manager")

        # Signal all threads to stop
        self.stop_event.set()

        # Wait for main threads to finish (with timeout)
        threads_to_stop = [
            (self.capture_thread, 'capture'),
            (self.processing_thread, 'processing'),
            (self.rendering_thread, 'rendering')
        ]

        # Add health monitor if available
        if hasattr(self, 'health_monitor') and self.health_monitor is not None:
            threads_to_stop.append((self.health_monitor, 'health_monitor'))

        # Stop threads
        for thread, name in threads_to_stop:
            if thread and thread.is_alive():
                self.logger.debug(f"Waiting for {name} thread to stop")
                thread.join(timeout=3)
                if thread.is_alive():
                    self.logger.warning(
                        f"{name} thread did not stop gracefully")
                else:
                    self.logger.debug(f"{name} thread stopped")
                self.thread_status[name] = False

        # Stop Firebase manager if it's initialized
        if self.firebase_manager and self.firebase_manager.initialized:
            self.firebase_manager.stop()
            self.thread_status['firebase'] = False

        # Clear queues with priority
        self._clear_queue(self.frame_queue)
        self._clear_queue(self.result_queue)

        # Reset thread initialization flag
        self.threads_initialized.clear()

        # Remove config callbacks if config manager is available
        if self.config_manager:
            self.config_manager.unregister_callback(self._handle_config_update)

        # Log final performance metrics
        if hasattr(self, 'metrics') and self.metrics:
            self.logger.info(
                f"Final performance: "
                f"Capture: {self.metrics.get('capture_fps', 0):.1f} FPS, "
                f"Processing: {self.metrics.get('process_fps', 0):.1f} FPS, "
                f"Rendering: {self.metrics.get('render_fps', 0):.1f} FPS"
            )

        self.logger.info("Thread manager stopped successfully")

    def _clear_queue(self, q):
        """Clear a queue safely"""
        if q is None:
            return 0

        # If the queue has a clear method, use it
        if hasattr(q, 'clear') and callable(getattr(q, 'clear')):
            try:
                return q.clear()
            except Exception as e:
                self.logger.debug(f"Error using clear() method: {e}")
                # Fall through to traditional approach

        # Traditional approach with multiple fallbacks
        try:
            count = 0
            while True:
                try:
                    # Try get_nowait first
                    if hasattr(q, 'get_nowait') and callable(getattr(q, 'get_nowait')):
                        q.get_nowait()
                    # Fall back to non-blocking get
                    else:
                        q.get(block=False)

                    # Mark task as done if the method exists
                    if hasattr(q, 'task_done') and callable(getattr(q, 'task_done')):
                        try:
                            q.task_done()
                        except ValueError:
                            # task_done called too many times, ignore
                            pass

                    count += 1

                except Exception as e:
                    if isinstance(e, queue.Empty):
                        # Queue is empty, we're done
                        break
                    else:
                        # Any other exception, log and break
                        self.logger.debug(f"Error clearing queue: {e}")
                        break

            return count
        except Exception as e:
            self.logger.debug(f"Error in queue clearing: {e}")
            return 0

    def _start_stats_update(self):
        """Start a daemon thread to update stats"""
        def update_stats():
            last_log_time = time.time()

            while not self.stop_event.is_set():
                try:
                    # Update metrics
                    self.metrics['capture_fps'] = getattr(
                        self.capture_thread, 'fps', 0)
                    self.metrics['process_fps'] = getattr(
                        self.processing_thread, 'fps', 0)
                    self.metrics['render_fps'] = getattr(
                        self.rendering_thread, 'fps', 0)

                    self.metrics['queue_sizes']['frame_queue'] = self.frame_queue.qsize(
                    )
                    self.metrics['queue_sizes']['result_queue'] = self.result_queue.qsize(
                    )

                    # Get Firebase queue size if available
                    if self.firebase_manager and self.firebase_manager.initialized:
                        firebase_stats = self.firebase_manager.get_stats()
                        self.metrics['queue_sizes']['firebase_queue'] = firebase_stats['current_queue_size']

                    # Update rendering thread with current stats
                    if self.rendering_thread:
                        self.rendering_thread.update_stats(
                            self.metrics['capture_fps'],
                            self.metrics['process_fps'],
                            self.metrics['queue_sizes']['frame_queue'],
                            self.metrics['queue_sizes']['result_queue']
                        )

                    # Log detailed stats periodically
                    current_time = time.time()
                    if current_time - last_log_time >= 30:  # Every 30 seconds
                        self._log_detailed_stats()
                        last_log_time = current_time

                    # Check for thread health
                    self._check_thread_health()

                except Exception as e:
                    self.logger.error(f"Error in stats update thread: {e}")

                time.sleep(0.5)  # Update twice per second

        stats_thread = threading.Thread(target=update_stats, daemon=True)
        stats_thread.start()

    def _check_thread_health(self):
        """Check the health of all threads"""
        # Check if any thread has died unexpectedly
        if (self.thread_status['capture'] and
                (not self.capture_thread or not self.capture_thread.is_alive())):
            self.logger.error("Capture thread has died unexpectedly")
            self.thread_status['capture'] = False

        if (self.thread_status['processing'] and
                (not self.processing_thread or not self.processing_thread.is_alive())):
            self.logger.error("Processing thread has died unexpectedly")
            self.thread_status['processing'] = False

        if (self.thread_status['rendering'] and
                (not self.rendering_thread or not self.rendering_thread.is_alive())):
            self.logger.error("Rendering thread has died unexpectedly")
            self.thread_status['rendering'] = False

        # Check for queue blockages
        if (self.frame_queue.qsize() >= self.frame_queue.maxsize * 0.9):
            self.logger.warning(
                "Frame queue is almost full - possible processing bottleneck")

        if (self.result_queue.qsize() >= self.result_queue.maxsize * 0.9):
            self.logger.warning(
                "Result queue is almost full - possible rendering bottleneck")

    def _log_detailed_stats(self):
        """Log detailed performance statistics"""
        self.logger.info(
            f"Performance: Capture: {self.metrics['capture_fps']:.1f} FPS, "
            f"Processing: {self.metrics['process_fps']:.1f} FPS, "
            f"Rendering: {self.metrics['render_fps']:.1f} FPS | "
            f"Queues: Frame: {self.metrics['queue_sizes']['frame_queue']}, "
            f"Result: {self.metrics['queue_sizes']['result_queue']}"
        )

    def get_thread_status(self):
        """Get the status of all threads and detailed performance metrics"""
        # Get enhanced metrics
        enhanced_metrics = self.metrics.copy()

        # Add queue statistics
        enhanced_metrics['queues'] = {
            'frame_queue': {
                'size': self.frame_queue.qsize(),
                'capacity': self.frame_queue.maxsize,
                'utilization': self.frame_queue.qsize() / max(1, self.frame_queue.maxsize) if self.frame_queue.maxsize > 0 else 0
            },
            'result_queue': {
                'size': self.result_queue.qsize(),
                'capacity': self.result_queue.maxsize,
                'utilization': self.result_queue.qsize() / max(1, self.result_queue.maxsize) if self.result_queue.maxsize > 0 else 0
            }
        }

        # Add Firebase queue stats if available
        if self.firebase_manager and self.firebase_manager.initialized:
            firebase_stats = self.firebase_manager.get_stats()
            enhanced_metrics['queues']['firebase_queue'] = {
                'size': firebase_stats.get('current_queue_size', 0),
                'capacity': 100,  # From FirebaseManager
                'utilization': firebase_stats.get('current_queue_size', 0) / 100
            }

        # Add detailed thread stats
        enhanced_metrics['threads'] = {}
        for thread_name, thread in self.threads.items():
            if thread is not None:
                enhanced_metrics['threads'][thread_name] = {
                    'alive': thread.is_alive(),
                    'daemon': thread.daemon,
                    'name': thread.name
                }

        # Add system stats
        try:
            import psutil
            process = psutil.Process(os.getpid())
            enhanced_metrics['system'] = {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'threads_count': process.num_threads()
            }
        except:
            enhanced_metrics['system'] = {}

        return {
            'all_running': all(self.thread_status.values()),
            'threads': self.thread_status.copy(),
            'metrics': enhanced_metrics
        }

    def _handle_config_update(self, path, value):
        """Handle dynamic configuration updates"""
        self.logger.info(f"Received configuration update: {path} = {value}")

        if path == 'threading.max_fps' and hasattr(self, 'capture_thread'):
            if hasattr(self.capture_thread, 'set_max_fps'):
                self.capture_thread.set_max_fps(value)
                self.logger.info(f"Updated max FPS to {value}")

        # For queue size changes, we'll need to use them during restart
        # since queues can't be resized dynamically
