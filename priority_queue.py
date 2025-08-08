"""
StreeRaksha Priority Queue Module
Implements priority queues for efficient inter-thread communication.
"""

import cv2
import queue
import threading
import time
import heapq
from datetime import datetime
from collections import deque


class PriorityItem:
    """Wrapper for items in PriorityQueue with comparison support"""

    def __init__(self, priority, sequence, item):
        self.priority = priority
        self.sequence = sequence  # Used to break ties in priority
        self.item = item
        self.timestamp = time.time()

    def __lt__(self, other):
        # Higher priority (lower number) comes first
        if self.priority != other.priority:
            return self.priority < other.priority
        # If same priority, use sequence to maintain FIFO within priority levels
        return self.sequence < other.sequence


class PriorityQueue:
    """
    Thread-safe priority queue for inter-thread communication
    Supports timeouts, blocking operations, and item prioritization
    """

    def __init__(self, maxsize=0):
        """Initialize the priority queue"""
        self._queue = []  # heap queue
        self._mutex = threading.RLock()
        self._not_empty = threading.Condition(self._mutex)
        self._not_full = threading.Condition(self._mutex)
        self._maxsize = maxsize
        self._counter = 0  # Sequence counter for FIFO ordering within priorities
        self._unfinished_tasks = 0
        self._finished = threading.Condition(self._mutex)

        # Statistics
        self._stats = {
            'enqueued': 0,
            'dequeued': 0,
            'high_priority': 0,
            'normal_priority': 0,
            'low_priority': 0,
            'peak_size': 0,
            'wait_times': deque(maxlen=100)  # Last 100 item wait times
        }

    def qsize(self):
        """Return the approximate size of the queue"""
        with self._mutex:
            return len(self._queue)

    def empty(self):
        """Return True if the queue is empty, False otherwise"""
        with self._mutex:
            return not self._queue

    def full(self):
        """Return True if the queue is full, False otherwise"""
        with self._mutex:
            return self._maxsize > 0 and len(self._queue) >= self._maxsize

    def put(self, item, priority=1, block=True, timeout=None):
        """
        Put an item into the queue with a specified priority
        Priority levels: 0 (highest) to 2 (lowest)
        """
        with self._not_full:
            if self._maxsize > 0:
                if not block:
                    if len(self._queue) >= self._maxsize:
                        raise queue.Full
                elif timeout is None:
                    while len(self._queue) >= self._maxsize:
                        self._not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time.time() + timeout
                    while len(self._queue) >= self._maxsize:
                        remaining = endtime - time.time()
                        if remaining <= 0.0:
                            raise queue.Full
                        self._not_full.wait(remaining)

            # Create a priority item and add to the heap
            with self._mutex:
                self._counter += 1
                priority_item = PriorityItem(priority, self._counter, item)
                heapq.heappush(self._queue, priority_item)

                # Update statistics
                self._stats['enqueued'] += 1
                if priority == 0:
                    self._stats['high_priority'] += 1
                elif priority == 1:
                    self._stats['normal_priority'] += 1
                elif priority == 2:
                    self._stats['low_priority'] += 1

                self._stats['peak_size'] = max(
                    self._stats['peak_size'], len(self._queue))

                # Signal that a new item is available
                self._unfinished_tasks += 1
                self._not_empty.notify()

    def get(self, block=True, timeout=None):
        """Remove and return the highest priority item from the queue"""
        with self._not_empty:
            if not block:
                if not self._queue:
                    raise queue.Empty
            elif timeout is None:
                while not self._queue:
                    self._not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time.time() + timeout
                while not self._queue:
                    remaining = endtime - time.time()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self._not_empty.wait(remaining)

            # Get the highest priority item
            with self._mutex:
                priority_item = heapq.heappop(self._queue)

                # Record wait time statistics
                wait_time = time.time() - priority_item.timestamp
                self._stats['wait_times'].append(wait_time)
                self._stats['dequeued'] += 1

                # Signal that a slot is now available
                self._not_full.notify()

                return priority_item.item

    def task_done(self):
        """Indicate that a formerly enqueued task is complete"""
        with self._mutex:
            unfinished = self._unfinished_tasks - 1
            if unfinished < 0:
                raise ValueError('task_done() called too many times')
            if unfinished == 0:
                self._finished.notify_all()
            self._unfinished_tasks = unfinished

    def join(self):
        """Block until all items in the queue have been gotten and processed"""
        with self._mutex:
            while self._unfinished_tasks:
                self._finished.wait()

    def clear(self):
        """Clear the queue (for emergency situations)"""
        with self._mutex:
            count = len(self._queue)
            self._queue = []
            self._unfinished_tasks = 0
            self._counter = 0
            self._not_full.notify_all()
            return count

    def get_stats(self):
        """Get statistics about queue usage"""
        with self._mutex:
            stats = self._stats.copy()

            # Calculate average wait time
            if stats['wait_times']:
                stats['avg_wait_time'] = sum(
                    stats['wait_times']) / len(stats['wait_times'])
            else:
                stats['avg_wait_time'] = 0

            # Calculate utilization
            stats['utilization'] = len(
                self._queue) / max(1, self._maxsize) if self._maxsize > 0 else 0
            stats['current_size'] = len(self._queue)

            return stats

    def get_nowait(self):
        """Remove and return an item from the queue without blocking.

        Only get an item if one is immediately available, otherwise
        raise the Empty exception.
        """
        return self.get(block=False)

    @property
    def maxsize(self):
        """Return the maximum size of the queue"""
        return self._maxsize


class PrioritizedFrameQueue(PriorityQueue):
    """
    Specialized priority queue for video frames with motion detection
    Prioritizes frames with significant motion or activity
    """

    def __init__(self, maxsize=0, motion_threshold=0.05):
        """Initialize the prioritized frame queue"""
        super().__init__(maxsize)
        self.motion_threshold = motion_threshold
        self.last_frame = None
        self.motion_history = deque(maxlen=10)  # Track recent motion

    def get_nowait(self):
        """Remove and return an item from the queue without blocking."""
        return super().get_nowait()

    @property
    def maxsize(self):
        """Return the maximum size of the queue"""
        return super().maxsize

    def put_frame(self, frame, timestamp, frame_count, block=True, timeout=None):
        """Put a frame into the queue with automatic priority based on motion"""
        # Calculate motion if we have a previous frame
        motion_score = 0
        priority = 1  # Default priority

        if self.last_frame is not None:
            try:
                # Simple motion detection by frame difference
                diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY))
                motion_score = cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[
                                                1]) / (diff.shape[0] * diff.shape[1])

                # Add to motion history
                self.motion_history.append(motion_score)

                # Determine priority based on motion
                if motion_score > self.motion_threshold:
                    # Higher motion gets higher priority (priority 0)
                    priority = 0
                elif motion_score < self.motion_threshold * 0.3:
                    # Very little motion gets lower priority (priority 2)
                    priority = 2
            except Exception:
                # In case of error, use default priority
                pass

        # Update last frame
        self.last_frame = frame.copy()

        # Create frame data with metadata
        frame_data = {
            'frame': frame,
            'timestamp': timestamp,
            'frame_count': frame_count,
            'motion_score': motion_score
        }

        # Put frame in queue with calculated priority
        self.put(frame_data, priority=priority, block=block, timeout=timeout)


# Import OpenCV for the PrioritizedFrameQueue
