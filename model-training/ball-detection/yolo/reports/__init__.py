"""
Reports Module Initialization

Makes the reports directory a Python package.
"""

__version__ = "1.0.0"

# Import main classes for convenience
from .report_generator import TrainingReportGenerator, create_report_generator

__all__ = [
    'TrainingReportGenerator',
    'create_report_generator'
]