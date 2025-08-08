"""
StreeRaksha Thread Health Monitor
Monitors and reports on thread health and performance.
"""

import threading
import time
import queue
from datetime import datetime, timedelta
import psutil
import os
import json


class ThreadHealthMonitor(threading.Thread):
    """
    Monitors the health of all threads in the system
    Reports on performance metrics, CPU/memory usage, and bottlenecks
    """

    def __init__(self, thread_manager, logger, report_interval=15, alert_threshold=0.8):
        """Initialize the thread health monitor"""
        super().__init__(daemon=True)
        self.thread_manager = thread_manager
        self.logger = logger
        self.report_interval = report_interval
        self.alert_threshold = alert_threshold
        self.stop_event = threading.Event()

        # Dictionary to store health metrics
        self.health_metrics = {
            'system': {
                'cpu_percent': 0,
                'memory_percent': 0,
                'start_time': datetime.now().isoformat()
            },
            'threads': {},
            'queues': {},
            'alerts': [],
            'bottlenecks': []
        }

        # History of metrics for trending analysis
        self.metrics_history = []
        self.history_max_size = 100  # Keep the last 100 measurements

        # Process for CPU/memory monitoring
        self.process = psutil.Process(os.getpid())

    def run(self):
        """Main monitoring loop"""
        self.logger.info("Thread health monitor started")
        last_report_time = time.time()

        while not self.stop_event.is_set():
            try:
                # Collect current metrics with exception handling
                try:
                    self._collect_metrics()
                except Exception as e:
                    self.logger.error(f"Error collecting metrics: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())

                # Check for potential issues with exception handling
                try:
                    self._analyze_metrics()
                except Exception as e:
                    self.logger.error(f"Error analyzing metrics: {e}")

                # Report at specified intervals
                current_time = time.time()
                if current_time - last_report_time >= self.report_interval:
                    try:
                        self._report_health()
                        self._save_metrics_snapshot()
                    except Exception as e:
                        self.logger.error(f"Error reporting health: {e}")
                    last_report_time = current_time

                # Short sleep to avoid consuming too much CPU
                time.sleep(1.0)

            except Exception as e:
                self.logger.error(f"Error in thread health monitor: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())

        self.logger.info("Thread health monitor stopped")

    def _collect_metrics(self):
        """Collect current health metrics from all threads and queues"""
        # System metrics
        try:
            self.health_metrics['system']['cpu_percent'] = self.process.cpu_percent(
            )
            self.health_metrics['system']['memory_percent'] = self.process.memory_percent(
            )
            self.health_metrics['system']['uptime_seconds'] = (
                datetime.now() -
                datetime.fromisoformat(
                    self.health_metrics['system']['start_time'])
            ).total_seconds()
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

        # Thread metrics
        thread_status = self.thread_manager.get_thread_status()
        self.health_metrics['threads'] = thread_status['threads']

        # Add thread-specific metrics (if thread manager has them)
        if hasattr(self.thread_manager, 'metrics'):
            for thread_name, metrics in self.thread_manager.metrics.items():
                if thread_name in self.health_metrics['threads']:
                    self.health_metrics['threads'][thread_name].update(metrics)

        # Queue metrics
        self.health_metrics['queues'] = {}

        # Safely add frame queue metrics
        try:
            frame_queue = self.thread_manager.frame_queue
            if frame_queue is not None:
                # Get queue size safely
                try:
                    size = frame_queue.qsize() if hasattr(frame_queue, 'qsize') else 0
                except Exception:
                    size = 0

                # Get maxsize using multiple fallback options
                try:
                    if hasattr(frame_queue, 'maxsize') and isinstance(frame_queue.maxsize, (int, float)):
                        capacity = frame_queue.maxsize
                    elif hasattr(frame_queue, '_maxsize'):
                        capacity = frame_queue._maxsize
                    else:
                        capacity = 0
                except Exception:
                    capacity = 0

                # Calculate utilization safely
                try:
                    utilization = size / \
                        max(1, capacity) if capacity > 0 else 0
                except Exception:
                    utilization = 0

                self.health_metrics['queues']['frame_queue'] = {
                    'size': size,
                    'capacity': capacity,
                    'utilization': utilization
                }
        except Exception as e:
            self.logger.debug(f"Error collecting frame queue metrics: {e}")
            self.health_metrics['queues']['frame_queue'] = {
                'size': 0, 'capacity': 0, 'utilization': 0}

        # Safely add result queue metrics
        try:
            result_queue = self.thread_manager.result_queue
            if result_queue is not None:
                # Get queue size safely
                try:
                    size = result_queue.qsize() if hasattr(result_queue, 'qsize') else 0
                except Exception:
                    size = 0

                # Get maxsize using multiple fallback options
                try:
                    if hasattr(result_queue, 'maxsize') and isinstance(result_queue.maxsize, (int, float)):
                        capacity = result_queue.maxsize
                    elif hasattr(result_queue, '_maxsize'):
                        capacity = result_queue._maxsize
                    else:
                        capacity = 0
                except Exception:
                    capacity = 0

                # Calculate utilization safely
                try:
                    utilization = size / \
                        max(1, capacity) if capacity > 0 else 0
                except Exception:
                    utilization = 0

                self.health_metrics['queues']['result_queue'] = {
                    'size': size,
                    'capacity': capacity,
                    'utilization': utilization
                }
        except Exception as e:
            self.logger.debug(f"Error collecting result queue metrics: {e}")
            self.health_metrics['queues']['result_queue'] = {
                'size': 0, 'capacity': 0, 'utilization': 0}

        # Add Firebase queue if available
        try:
            if (self.thread_manager.firebase_manager and
                hasattr(self.thread_manager.firebase_manager, 'initialized') and
                    self.thread_manager.firebase_manager.initialized):

                # Get stats safely
                try:
                    firebase_stats = self.thread_manager.firebase_manager.get_stats()
                except Exception as e:
                    self.logger.debug(f"Error getting Firebase stats: {e}")
                    firebase_stats = {}

                # Get queue size and capacity safely
                queue_size = firebase_stats.get('current_queue_size', 0)
                # Default to 100 if not specified
                capacity = firebase_stats.get('queue_capacity', 100)

                # Calculate utilization safely
                try:
                    utilization = queue_size / max(1, capacity)
                except Exception:
                    utilization = 0

                self.health_metrics['queues']['firebase_queue'] = {
                    'size': queue_size,
                    'capacity': capacity,
                    'utilization': utilization
                }
        except Exception as e:
            self.logger.debug(f"Error collecting Firebase queue metrics: {e}")
            # Add default firebase_queue metrics if there was an error
            self.health_metrics['queues']['firebase_queue'] = {
                'size': 0, 'capacity': 0, 'utilization': 0}

    def _analyze_metrics(self):
        """Analyze metrics to detect potential issues"""
        # Clear previous alerts and bottlenecks
        self.health_metrics['alerts'] = []
        self.health_metrics['bottlenecks'] = []

        # Check for thread failures
        for thread_name, status in self.health_metrics['threads'].items():
            if thread_name in ['capture', 'processing', 'rendering'] and not status:
                self.health_metrics['alerts'].append({
                    'level': 'critical',
                    'message': f"{thread_name} thread is not running",
                    'timestamp': datetime.now().isoformat()
                })

        # Check for queue bottlenecks
        for queue_name, metrics in self.health_metrics['queues'].items():
            if metrics['utilization'] > self.alert_threshold:
                self.health_metrics['bottlenecks'].append({
                    'component': queue_name,
                    'utilization': metrics['utilization'],
                    'message': f"{queue_name} is {metrics['utilization']*100:.1f}% full",
                    'timestamp': datetime.now().isoformat()
                })

        # Check for system resource issues
        if self.health_metrics['system']['cpu_percent'] > 90:
            self.health_metrics['alerts'].append({
                'level': 'warning',
                'message': f"High CPU usage: {self.health_metrics['system']['cpu_percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })

        if self.health_metrics['system']['memory_percent'] > 85:
            self.health_metrics['alerts'].append({
                'level': 'warning',
                'message': f"High memory usage: {self.health_metrics['system']['memory_percent']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })

        # Log critical alerts
        for alert in self.health_metrics['alerts']:
            if alert['level'] == 'critical':
                self.logger.error(f"Thread health alert: {alert['message']}")
            elif alert['level'] == 'warning':
                self.logger.warning(
                    f"Thread health warning: {alert['message']}")

    def _report_health(self):
        """Report on current system health"""
        # Prepare report summary
        system_metrics = self.health_metrics['system']
        queue_metrics = self.health_metrics['queues']

        # Log summary
        self.logger.info(
            f"System Health: CPU: {system_metrics['cpu_percent']:.1f}%, "
            f"Memory: {system_metrics['memory_percent']:.1f}%, "
            f"Uptime: {timedelta(seconds=int(system_metrics['uptime_seconds']))} | "
            f"Queues: Frame: {queue_metrics['frame_queue']['size']}/{queue_metrics['frame_queue']['capacity']}, "
            f"Result: {queue_metrics['result_queue']['size']}/{queue_metrics['result_queue']['capacity']}"
        )

        # Log any bottlenecks
        for bottleneck in self.health_metrics['bottlenecks']:
            self.logger.warning(
                f"Bottleneck detected: {bottleneck['message']}")

    def _save_metrics_snapshot(self):
        """Save current metrics to history for trend analysis"""
        # Create a snapshot with timestamp
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'system': self.health_metrics['system'].copy(),
            'queue_sizes': {name: metrics['size'] for name, metrics in self.health_metrics['queues'].items()},
            'fps_metrics': {
                'capture': getattr(self.thread_manager, 'metrics', {}).get('capture_fps', 0),
                'process': getattr(self.thread_manager, 'metrics', {}).get('process_fps', 0),
                'render': getattr(self.thread_manager, 'metrics', {}).get('render_fps', 0)
            }
        }

        # Add to history
        self.metrics_history.append(snapshot)

        # Trim history if too large
        if len(self.metrics_history) > self.history_max_size:
            self.metrics_history = self.metrics_history[-self.history_max_size:]

    def get_health_report(self):
        """Get the current health report for external use"""
        return {
            'current': self.health_metrics,
            'history': self.metrics_history[-10:] if self.metrics_history else []
        }

    def export_metrics(self, filepath):
        """Export metrics history to a JSON file"""
        try:
            data = {
                'export_time': datetime.now().isoformat(),
                'metrics_history': self.metrics_history,
                'current_state': self.health_metrics
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Metrics exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False

    def stop(self):
        """Stop the health monitor"""
        self.stop_event.set()
