"""
Cooperative cancellation support for long-running computational tasks.

This module provides infrastructure for gracefully stopping computations
mid-execution, allowing users to cancel expensive operations like elastic
tensor calculations without data corruption or resource leaks.

Key Concepts:
-------------
1. **Cooperative Cancellation**: Computation functions periodically check
   a cancellation flag and exit gracefully at safe checkpoints.

2. **Threading Events**: Uses threading.Event for thread-safe cancellation
   signaling (compatible with cl.make_async thread pool execution).

3. **Progress Tracking**: Optional callbacks allow real-time progress updates
   to the UI during computation.

Usage Example:
--------------
```python
import threading
from src.core.cancellation import check_cancellation, ComputationCancelledException

def long_computation(items, cancellation_event=None, progress_callback=None):
    for idx, item in enumerate(items):
        # Check if user requested cancellation
        check_cancellation(cancellation_event, idx, len(items))

        # Report progress to UI
        if progress_callback:
            progress_callback(idx, len(items))

        # Do actual work
        result = expensive_operation(item)

    return results

# In Chainlit tool:
event = cl.user_session.get("computation_cancellation_event")
try:
    results = await cl.make_async(long_computation)(
        items,
        cancellation_event=event,
        progress_callback=update_ui_progress
    )
except ComputationCancelledException as e:
    await cl.Message(
        content=f"⏹️ Stopped after {e.completed}/{e.total} items ({e.progress:.1%})"
    ).send()
```

For detailed documentation, see: docs/LONG_RUNNING_TASKS.md
"""

import threading
from typing import Optional, Callable


class ComputationCancelledException(Exception):
    """
    Exception raised when a computation is cancelled by user request.

    This exception carries metadata about the cancellation point, allowing
    tools to report partial progress and save intermediate results.

    Attributes:
        message: Human-readable cancellation message
        completed: Number of computation steps completed before cancellation
        total: Total number of computation steps
        progress: Fraction of work completed (0.0 to 1.0)

    Examples:
        >>> try:
        ...     compute_elastic_tensor(atoms, cancellation_event=event)
        ... except ComputationCancelledException as e:
        ...     print(f"Cancelled at {e.progress:.1%} completion")
        ...     print(f"Completed {e.completed}/{e.total} deformations")
    """

    def __init__(self, message: str, completed: int, total: int):
        """
        Initialize cancellation exception with progress metadata.

        Parameters:
            message: Description of where cancellation occurred
            completed: Number of completed steps (0 to total)
            total: Total number of steps in computation
        """
        super().__init__(message)
        self.message = message
        self.completed = completed
        self.total = total
        self.progress = completed / total if total > 0 else 0.0

    def __str__(self) -> str:
        """Return formatted cancellation message with progress."""
        return f"{self.message} (Progress: {self.completed}/{self.total}, {self.progress:.1%})"


def check_cancellation(
    event: Optional[threading.Event],
    step: int,
    total: int,
    operation_name: str = "operation"
) -> None:
    """
    Check if cancellation was requested and raise exception if so.

    This function should be called periodically in long-running computations
    at safe checkpoints (e.g., between loop iterations). It checks the
    threading.Event flag and raises ComputationCancelledException if set.

    **Best Practices for Checkpoint Placement:**

    - Call between major computation steps (e.g., after each deformation)
    - Avoid calling inside critical sections or atomic operations
    - Don't call too frequently (overhead) or too rarely (poor responsiveness)
    - Typical frequency: every 5-30 seconds of computation

    Parameters:
        event: Threading event to check (None = never cancel)
        step: Current step number (0-indexed)
        total: Total number of steps
        operation_name: Human-readable name for the operation (for error messages)

    Raises:
        ComputationCancelledException: If event is set (cancellation requested)

    Examples:
        >>> event = threading.Event()
        >>> for i in range(100):
        ...     check_cancellation(event, i, 100, "tensor deformation")
        ...     expensive_computation()

        >>> # User clicks stop button
        >>> event.set()
        >>> check_cancellation(event, 45, 100, "tensor deformation")
        ComputationCancelledException: Cancelled tensor deformation at step 45/100
    """
    if event is not None and event.is_set():
        raise ComputationCancelledException(
            message=f"Cancelled {operation_name} at step {step}/{total}",
            completed=step,
            total=total
        )


# Type alias for progress callback functions
ProgressCallback = Optional[Callable[[int, int, str], None]]
"""
Type alias for progress callback functions.

Progress callbacks receive:
- current_step (int): Current step number (0-indexed)
- total_steps (int): Total number of steps
- status_message (str): Human-readable status (e.g., "Relaxing deformation 45/180")

Example:
    >>> def update_ui(current: int, total: int, msg: str):
    ...     print(f"{msg} - {current/total:.1%} complete")
    >>>
    >>> callback: ProgressCallback = update_ui
"""


def format_progress_message(
    current: int,
    total: int,
    operation: str,
    show_percentage: bool = True
) -> str:
    """
    Format a consistent progress message string.

    Parameters:
        current: Current step (0-indexed or 1-indexed, depends on context)
        total: Total steps
        operation: Operation name (e.g., "Computing deformation")
        show_percentage: Include percentage in message

    Returns:
        Formatted message like "Computing deformation 45/180 (25%)"

    Examples:
        >>> format_progress_message(45, 180, "Computing deformation")
        'Computing deformation 45/180 (25%)'

        >>> format_progress_message(45, 180, "Relaxing structure", show_percentage=False)
        'Relaxing structure 45/180'
    """
    if show_percentage:
        percentage = (current / total * 100) if total > 0 else 0
        return f"{operation} {current}/{total} ({percentage:.0f}%)"
    else:
        return f"{operation} {current}/{total}"
