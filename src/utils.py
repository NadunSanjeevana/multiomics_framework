# src/utils.py
"""
Utility functions for the multi-omics pipeline
"""

import logging
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> dict:
    """Load YAML configuration file and return as dictionary"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config