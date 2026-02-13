"""Utility functions for the application."""

from .config import ArchitectureConfig, TrainingConfig, config
from .device import auto_scale_config

__all__ = ['ArchitectureConfig', 'TrainingConfig', 'auto_scale_config', 'config']
