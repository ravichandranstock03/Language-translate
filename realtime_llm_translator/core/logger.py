"""
Advanced Logging System with Real-time Metrics
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class MetricsLogger:
    """Specialized logger for performance metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_buffer = []
    
    def log_latency(self, operation: str, latency_ms: float, metadata: Optional[dict] = None):
        """Log latency metric"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "latency",
            "operation": operation,
            "value_ms": latency_ms,
            "metadata": metadata or {}
        }
        self.metrics_buffer.append(metric)
        self.logger.info(f"[METRIC] {operation}: {latency_ms:.2f}ms")
    
    def log_throughput(self, operation: str, items_per_second: float, metadata: Optional[dict] = None):
        """Log throughput metric"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "throughput",
            "operation": operation,
            "value_pps": items_per_second,
            "metadata": metadata or {}
        }
        self.metrics_buffer.append(metric)
        self.logger.info(f"[METRIC] {operation}: {items_per_second:.2f} items/sec")
    
    def log_error_rate(self, operation: str, error_rate: float, total_requests: int):
        """Log error rate metric"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "error_rate",
            "operation": operation,
            "value": error_rate,
            "total_requests": total_requests
        }
        self.metrics_buffer.append(metric)
        self.logger.warning(f"[METRIC] {operation} error rate: {error_rate*100:.2f}% ({total_requests} requests)")
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_buffer, f, indent=2)
    
    def clear_buffer(self):
        """Clear the metrics buffer"""
        self.metrics_buffer = []


def setup_logger(
    name: str = "RealTimeLLMTranslator",
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_metrics: bool = True
) -> logging.Logger:
    """
    Setup advanced logging system
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        enable_console: Enable console output with colors
        enable_metrics: Enable metrics tracking
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with colored formatting
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_format = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler with detailed formatting
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Add custom methods for metrics
    if enable_metrics:
        logger.metrics = MetricsLogger(logger)
    else:
        logger.metrics = None
    
    return logger
