# -*- coding: utf-8 -*-
"""
src/utils/retry_policy.py

Generic retry decorator with exponential backoff.
"""

import time
import functools
from typing import Callable, Type, Tuple, Optional


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator: retry with exponential backoff.
    
    Args:
        max_retries: Maximum retry count (excluding initial attempt)
        initial_delay: Initial delay in seconds
        backoff_factor: Delay multiplier after each retry
        exceptions: Exception types to catch and retry
        on_retry: Callback function(attempt_num, exception) called on each retry
    
    Example:
        @retry_with_backoff(max_retries=2, initial_delay=1.0, backoff_factor=2.0)
        def unstable_api_call():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt >= max_retries:
                        raise  # Last attempt: re-raise exception
                    
                    if on_retry:
                        on_retry(attempt + 1, e)
                    
                    time.sleep(delay)
                    delay *= backoff_factor
            
            # Should not reach here (raise above)
            raise last_exc
        
        return wrapper
    return decorator

