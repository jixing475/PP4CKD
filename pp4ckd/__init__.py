"""
PP4CKD: Polypharmacology Predictor for Chronic Kidney Disease
A deep multi-target ligand-based bioactivity prediction framework for nephrology targets.
"""

__version__ = "0.1.0"
__author__ = "CKDdb Team"

from .model import PPB3Net
from .fingerprints import (
    compute_fingerprint,
    standardize_smiles,
    AVAILABLE_FINGERPRINTS,
    FINGERPRINT_DIMS,
)
from .predictor import PP4CKDPredictor

__all__ = [
    "PPB3Net",
    "PP4CKDPredictor",
    "compute_fingerprint",
    "standardize_smiles",
    "AVAILABLE_FINGERPRINTS",
    "FINGERPRINT_DIMS",
]
