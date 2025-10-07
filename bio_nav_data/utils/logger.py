"""
Logging Configuration for Bio-Inspired Navigation Research

This module provides centralized logging configuration for the project,
ensuring consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import datetime


def setup_logger(name: str = "bio_nav_data",
                level: int = logging.INFO,
                log_file: Optional[str] = None,
                console_output: bool = True,
                file_output: bool = True) -> logging.Logger:
    """
    Set up a logger with consistent configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "bio_nav_data") -> logging.Logger:
    """
    Get a logger instance with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"bio_nav_data_{timestamp}.log"
        
        setup_logger(name, log_file=str(log_file))
    
    return logger


def log_function_call(func):
    """
    Decorator to log function calls with parameters and return values.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        
        # Log function entry
        logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            # Call the function
            result = func(*args, **kwargs)
            
            # Log successful completion
            logger.debug(f"Exiting {func.__name__} with result={result}")
            return result
            
        except Exception as e:
            # Log error
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise
    
    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.datetime.now()
        
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Completed {func.__name__} in {execution_time:.3f} seconds")
            return result
            
        except Exception as e:
            end_time = datetime.datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.error(f"Failed {func.__name__} after {execution_time:.3f} seconds: {e}")
            raise
    
    return wrapper


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_exception(self, message: str) -> None:
        """Log exception message with traceback."""
        self.logger.exception(message)


def create_log_summary(log_file: str) -> str:
    """
    Create a summary of log file contents.
    
    Args:
        log_file: Path to log file
        
    Returns:
        str: Summary of log contents
    """
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Count log levels
        level_counts = {}
        error_lines = []
        
        for line in lines:
            if ' - ' in line:
                parts = line.split(' - ')
                if len(parts) >= 3:
                    level = parts[2]
                    level_counts[level] = level_counts.get(level, 0) + 1
                    
                    if 'ERROR' in level:
                        error_lines.append(line.strip())
        
        # Create summary
        summary = f"""
Log File Summary: {log_file}
Total Lines: {len(lines)}
Log Levels: {level_counts}

Errors Found: {len(error_lines)}
"""
        
        if error_lines:
            summary += "\nRecent Errors:\n"
            for error in error_lines[-5:]:  # Last 5 errors
                summary += f"  {error}\n"
        
        return summary
        
    except Exception as e:
        return f"Error reading log file {log_file}: {e}"


# Set up default logger
default_logger = get_logger() 