"""
Metrics tracking and monitoring for MemoGraph.

This module provides comprehensive metrics collection for monitoring
system performance, resource usage, and operation statistics.

Example:
    >>> from memograph.core.metrics import MetricsCollector, get_metrics
    >>>
    >>> # Get global metrics collector
    >>> metrics = get_metrics()
    >>>
    >>> # Track operation
    >>> with metrics.track_operation("retrieve_nodes"):
    ...     results = kernel.retrieve_nodes("query")
    >>>
    >>> # Get statistics
    >>> stats = metrics.get_stats()
    >>> print(f"Average query time: {stats['retrieve_nodes']['avg_duration_ms']:.2f}ms")
"""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class OperationMetrics:
    """Metrics for a specific operation type."""

    count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    error_count: int = 0
    recent_durations: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_execution: Optional[datetime] = None

    def record(self, duration_ms: float, success: bool = True):
        """Record an operation execution.

        Args:
            duration_ms: Duration of the operation in milliseconds
            success: Whether the operation succeeded
        """
        self.count += 1
        self.total_duration_ms += duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.recent_durations.append(duration_ms)
        self.last_execution = datetime.now()

        if not success:
            self.error_count += 1

    @property
    def avg_duration_ms(self) -> float:
        """Average duration across all executions."""
        return self.total_duration_ms / self.count if self.count > 0 else 0.0

    @property
    def p95_duration_ms(self) -> float:
        """95th percentile duration from recent executions."""
        if not self.recent_durations:
            return 0.0
        sorted_durations = sorted(self.recent_durations)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[idx] if idx < len(sorted_durations) else sorted_durations[-1]

    @property
    def p99_duration_ms(self) -> float:
        """99th percentile duration from recent executions."""
        if not self.recent_durations:
            return 0.0
        sorted_durations = sorted(self.recent_durations)
        idx = int(len(sorted_durations) * 0.99)
        return sorted_durations[idx] if idx < len(sorted_durations) else sorted_durations[-1]

    @property
    def error_rate(self) -> float:
        """Error rate as percentage."""
        return (self.error_count / self.count * 100) if self.count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "count": self.count,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms
            if self.min_duration_ms != float("inf")
            else 0.0,
            "max_duration_ms": self.max_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "p99_duration_ms": self.p99_duration_ms,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
        }


class MetricsCollector:
    """Centralized metrics collection for MemoGraph operations."""

    def __init__(self):
        """Initialize metrics collector."""
        self._operations: dict[str, OperationMetrics] = defaultdict(OperationMetrics)
        self._lock = threading.Lock()
        self._start_time = datetime.now()

    def record_operation(
        self, operation: str, duration_ms: float, success: bool = True, **metadata
    ):
        """Record an operation execution.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            **meta Additional metadata to log
        """
        with self._lock:
            self._operations[operation].record(duration_ms, success)

    @contextmanager
    def track_operation(self, operation: str, **metadata):
        """Context manager to track operation duration.

        Args:
            operation: Operation name
            **meta Additional metadata

        Yields:
            None

        Example:
            >>> metrics = MetricsCollector()
            >>> with metrics.track_operation("retrieve_nodes", query="test"):
            ...     results = perform_retrieval()
        """
        start_time = time.time()
        success = True

        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_operation(operation, duration_ms, success, **metadata)

    def get_operation_stats(self, operation: str) -> Optional[dict[str, Any]]:
        """Get statistics for a specific operation.

        Args:
            operation: Operation name

        Returns:
            Dictionary with operation statistics or None
        """
        with self._lock:
            if operation in self._operations:
                return self._operations[operation].to_dict()
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get all collected statistics.

        Returns:
            Dictionary with all metrics
        """
        with self._lock:
            return {
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                "operations": {
                    op_name: op_metrics.to_dict()
                    for op_name, op_metrics in self._operations.items()
                },
            }

    def get_summary(self) -> dict[str, Any]:
        """Get summary of key metrics.

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            total_ops = sum(op.count for op in self._operations.values())
            total_errors = sum(op.error_count for op in self._operations.values())

            return {
                "total_operations": total_ops,
                "total_errors": total_errors,
                "error_rate": (total_errors / total_ops * 100) if total_ops > 0 else 0.0,
                "operations_count": len(self._operations),
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._operations.clear()
            self._start_time = datetime.now()

    def reset_operation(self, operation: str):
        """Reset metrics for a specific operation.

        Args:
            operation: Operation name to reset
        """
        with self._lock:
            if operation in self._operations:
                self._operations[operation] = OperationMetrics()


# Global metrics collector instance
_global_metrics: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        Global MetricsCollector instance

    Example:
        >>> metrics = get_metrics()
        >>> with metrics.track_operation("my_operation"):
        ...     do_work()
    """
    global _global_metrics

    if _global_metrics is None:
        with _metrics_lock:
            if _global_metrics is None:
                _global_metrics = MetricsCollector()

    return _global_metrics


def reset_global_metrics():
    """Reset the global metrics collector."""
    metrics = get_metrics()
    metrics.reset()


# Convenience decorators
def track_performance(operation_name: str):
    """Decorator to track function performance.

    Args:
        operation_name: Name for the operation

    Example:
        >>> @track_performance("my_function")
        ... def my_function():
        ...     time.sleep(1)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.track_operation(operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_async_performance(operation_name: str):
    """Decorator to track async function performance.

    Args:
        operation_name: Name for the operation

    Example:
        >>> @track_async_performance("my_async_function")
        ... async def my_async_function():
        ...     await asyncio.sleep(1)
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            metrics = get_metrics()
            start_time = time.time()
            success = True

            try:
                return await func(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                metrics.record_operation(operation_name, duration_ms, success)

        return wrapper

    return decorator
