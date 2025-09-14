"""
Logging configuration for Customer Churn Prediction pipeline
"""
import logging
import sys
import time
from pathlib import Path
from typing import Optional
import os


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up logging configuration for the entire pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name. If None, uses default naming
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger('churn_prediction')
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        if not log_file.endswith('.log'):
            log_file += '.log'
        
        file_path = Path(log_dir) / log_file
        file_handler = logging.FileHandler(file_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {file_path}")
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Logger instance
    """
    # Ensure the main logger is set up
    if not logging.getLogger('churn_prediction').handlers:
        setup_logging()
    
    # Create child logger
    logger = logging.getLogger(f'churn_prediction.{name}')
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                super().__init__()
                self.logger.info("MyClass initialized")
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)


def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time.
    
    Usage:
        @log_function_call
        def my_function(param1, param2):
            return result
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        args_str = ', '.join([str(arg) for arg in args])
        kwargs_str = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
        params_str = ', '.join(filter(None, [args_str, kwargs_str]))
        
        logger.debug(f"Calling {func.__name__}({params_str})")
        
        # Execute function and measure time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    return wrapper


# Context manager for temporary log level changes
class LogLevel:
    """
    Context manager to temporarily change log level.
    
    Usage:
        with LogLevel('DEBUG'):
            # Debug logging enabled
            logger.debug("This will be shown")
    """
    
    def __init__(self, level: str):
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.original_level = None
    
    def __enter__(self):
        logger = logging.getLogger('churn_prediction')
        self.original_level = logger.level
        logger.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger('churn_prediction')
        logger.setLevel(self.original_level)


# Performance logging utility
class PerformanceTimer:
    """
    Context manager for measuring and logging execution time.
    
    Usage:
        with PerformanceTimer("Data loading"):
            data = load_data()
    """
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"{self.operation_name} completed in {execution_time:.2f}s")
        else:
            self.logger.error(f"{self.operation_name} failed after {execution_time:.2f}s")


# Helper function to configure logging for different environments
def configure_logging_for_env(env: str = "development") -> logging.Logger:
    """
    Configure logging based on environment.
    
    Args:
        env: Environment name (development, testing, production)
        
    Returns:
        Configured logger
    """
    if env == "development":
        return setup_logging(
            log_level="DEBUG",
            log_file="churn_prediction_dev",
            log_dir="logs"
        )
    elif env == "testing":
        return setup_logging(
            log_level="WARNING",
            log_file="churn_prediction_test",
            log_dir="logs"
        )
    elif env == "production":
        return setup_logging(
            log_level="INFO",
            log_file="churn_prediction_prod",
            log_dir="logs"
        )
    else:
        # Default to development settings
        return setup_logging(log_level="INFO")