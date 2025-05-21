"""
Logger Module - For recording key information during program execution

Author: Tian You
Date: 2025-05-21
"""

import logging
import os
from pathlib import Path
import time
import sys
from datetime import datetime

# Log storage path
LOG_DIR = Path(__file__).parent / 'logs'

def setup_logger(name=None, log_file=None, level=logging.INFO, console_output=True):
    """
    Set up a logger
    
    Parameters:
        name (str): Logger name, default is root logger
        log_file (str/Path): Log file path, default is named with current time
        level (int): Log level, default is INFO
        console_output (bool): Whether to output to console, default is True
    
    Returns:
        logger: Configured logger
    """    
    # Create log directory (if not exists)
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set default log file name
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{timestamp}.log"
    else:
        log_file = LOG_DIR / log_file
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # If console output is needed
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

# Create default logger
def get_logger(module_name=None):
    """
    Get module logger
    
    Parameters:
        module_name (str): Module name
    
    Returns:
        logger: Logger
    """
    logger_name = module_name if module_name else 'CNCityRisk'
    return logging.getLogger(logger_name)

# Initialize root logger
def init_root_logger(log_file=None, level=logging.INFO):
    """
    Initialize root logger
    
    Parameters:
        log_file (str): Log file name
        level (int): Log level
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"CNCityRisk_{timestamp}.log"
      
    setup_logger(name=None, log_file=log_file, level=level)
    
    root_logger = logging.getLogger()
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log file saved at: {LOG_DIR / log_file}")
    
    return root_logger