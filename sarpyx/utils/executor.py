import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import Callable, List, Any, Optional, Dict, Union
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def _validate_function(func: Callable) -> None:
    """Validate that the provided function is callable."""
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func)}")


class MultiprocessExecutor:
    """
    An enhanced multiprocess executor class for parallel task execution.
    
    Features:
    - Context manager support
    - Progress tracking
    - Timeout handling
    - Error recovery
    - Result ordering preservation
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None, 
                 mp_context: Optional[str] = None,
                 timeout: Optional[float] = None):
        """
        Initialize the multiprocess executor.
        
        Args:
            max_workers: Maximum number of worker processes. Defaults to CPU count.
            mp_context: Multiprocessing context ('spawn', 'fork', 'forkserver').
            timeout: Default timeout for task execution in seconds.
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.mp_context = mp.get_context(mp_context) if mp_context else None
        self.timeout = timeout
        self._executor = None
        self._shutdown_requested = False
    
    def __enter__(self):
        self._executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=self.mp_context
        )
        logger.info(f"MultiprocessExecutor initialized with {self.max_workers} workers")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Gracefully shutdown the executor."""
        if self._executor and not self._shutdown_requested:
            self._shutdown_requested = True
            logger.info("Shutting down MultiprocessExecutor")
            try:
                self._executor.shutdown(wait=wait, timeout=timeout)
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
    
    def execute_tasks(self, 
                     func: Callable, 
                     tasks: List[Any], 
                     timeout: Optional[float] = None,
                     progress_callback: Optional[Callable[[int, int], None]] = None,
                     preserve_order: bool = True,
                     **kwargs) -> List[Any]:
        """
        Execute multiple tasks in parallel with enhanced features.
        
        Args:
            func: Function to execute for each task
            tasks: List of task arguments
            timeout: Timeout for individual tasks (overrides default)
            progress_callback: Callback function called with (completed, total)
            preserve_order: Whether to preserve the original order of results
            **kwargs: Additional keyword arguments for the function
            
        Returns:
            List of results from task execution
            
        Raises:
            RuntimeError: If executor not initialized
            TypeError: If func is not callable
        """
        if not self._executor:
            raise RuntimeError("Executor not initialized. Use within context manager.")
        
        _validate_function(func)
        
        if not tasks:
            logger.warning("No tasks provided")
            return []
        
        task_timeout = timeout or self.timeout
        total_tasks = len(tasks)
        logger.info(f"Executing {total_tasks} tasks with {self.max_workers} workers")
        
        # Submit all tasks with indices for order preservation
        future_to_index = {}
        for i, task in enumerate(tasks):
            try:
                if isinstance(task, (list, tuple)):
                    future = self._executor.submit(func, *task, **kwargs)
                else:
                    future = self._executor.submit(func, task, **kwargs)
                future_to_index[future] = i
            except Exception as e:
                logger.error(f"Failed to submit task {i}: {e}")
                future_to_index[None] = i
        
        # Collect results
        results = [None] * total_tasks if preserve_order else []
        completed = 0
        
        for future in as_completed(future_to_index.keys()):
            if future is None:
                completed += 1
                continue
                
            try:
                result = future.result(timeout=task_timeout)
                if preserve_order:
                    results[future_to_index[future]] = result
                else:
                    results.append(result)
                    
            except TimeoutError:
                error_msg = f"Task {future_to_index[future]} timed out after {task_timeout}s"
                logger.error(error_msg)
                if preserve_order:
                    results[future_to_index[future]] = None
                else:
                    results.append(None)
                    
            except Exception as e:
                error_msg = f"Task {future_to_index[future]} failed: {e}"
                logger.error(error_msg)
                if preserve_order:
                    results[future_to_index[future]] = None
                else:
                    results.append(None)
            
            completed += 1
            if progress_callback:
                try:
                    progress_callback(completed, total_tasks)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
        
        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Completed {total_tasks} tasks: {success_count} successful, {total_tasks - success_count} failed")
        
        return results
    
    def map(self, 
            func: Callable, 
            iterable: List[Any], 
            chunksize: int = 1,
            timeout: Optional[float] = None) -> List[Any]:
        """
        Apply function to every item in iterable in parallel with timeout support.
        
        Args:
            func: Function to apply
            iterable: Items to process
            chunksize: Number of items per chunk
            timeout: Timeout for the entire operation
            
        Returns:
            List of results
            
        Raises:
            RuntimeError: If executor not initialized
            TypeError: If func is not callable
            TimeoutError: If operation times out
        """
        if not self._executor:
            raise RuntimeError("Executor not initialized. Use within context manager.")
        
        _validate_function(func)
        
        if not iterable:
            return []
        
        operation_timeout = timeout or self.timeout
        logger.info(f"Mapping function over {len(iterable)} items with chunksize={chunksize}")
        
        try:
            start_time = time.time()
            result = list(self._executor.map(func, iterable, chunksize=chunksize, timeout=operation_timeout))
            elapsed = time.time() - start_time
            logger.info(f"Map operation completed in {elapsed:.2f}s")
            return result
            
        except TimeoutError:
            logger.error(f"Map operation timed out after {operation_timeout}s")
            raise
        except Exception as e:
            logger.error(f"Map operation failed: {e}")
            raise
    
    def submit_single(self, func: Callable, *args, timeout: Optional[float] = None, **kwargs) -> Any:
        """
        Submit a single task and wait for result.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            timeout: Timeout for the task
            **kwargs: Keyword arguments
            
        Returns:
            Task result
        """
        if not self._executor:
            raise RuntimeError("Executor not initialized. Use within context manager.")
        
        _validate_function(func)
        
        task_timeout = timeout or self.timeout
        future = self._executor.submit(func, *args, **kwargs)
        
        try:
            return future.result(timeout=task_timeout)
        except TimeoutError:
            logger.error(f"Single task timed out after {task_timeout}s")
            raise
        except Exception as e:
            logger.error(f"Single task failed: {e}")
            raise


# Enhanced example usage:
"""
Example 1: Basic usage with progress tracking
--------------------------------------------

def process_data(data, multiplier=2):
    import time
    time.sleep(0.1)  # Simulate work
    return data * multiplier

def progress_tracker(completed, total):
    print(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

tasks = list(range(1, 21))  # 1 to 20

with MultiprocessExecutor(max_workers=4, timeout=5.0) as executor:
    results = executor.execute_tasks(
        process_data, 
        tasks, 
        progress_callback=progress_tracker,
        multiplier=3
    )
    print(f"Results: {results}")


Example 2: Error handling and timeout
------------------------------------

def unreliable_task(x):
    import time, random
    if random.random() < 0.3:  # 30% chance of failure
        raise ValueError(f"Task {x} failed randomly")
    if random.random() < 0.2:  # 20% chance of timeout
        time.sleep(10)  # Long task
    return x ** 2

numbers = list(range(1, 11))

with MultiprocessExecutor(max_workers=3, timeout=2.0) as executor:
    results = executor.execute_tasks(unreliable_task, numbers)
    successful = [r for r in results if r is not None]
    print(f"Successful results: {successful}")


Example 3: Single task submission
--------------------------------

def complex_calculation(base, exponent, modulo=None):
    result = pow(base, exponent)
    return result % modulo if modulo else result

with MultiprocessExecutor() as executor:
    result = executor.submit_single(complex_calculation, 2, 1000, modulo=1000000007)
    print(f"Result: {result}")


Example 4: Batch processing with chunks
--------------------------------------

def batch_processor(items):
    # Process a batch of items
    return [item * 2 for item in items]

# Split data into chunks for processing
data = list(range(1, 101))  # 1 to 100
chunk_size = 10
chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

with MultiprocessExecutor(max_workers=4) as executor:
    batch_results = executor.map(batch_processor, chunks, chunksize=2)
    # Flatten results
    final_results = [item for batch in batch_results for item in batch]
    print(f"Processed {len(final_results)} items")
"""