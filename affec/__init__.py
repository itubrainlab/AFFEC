"""
AFFEC: Advancing Facial-based Emotion Classification
Multimodal emotion recognition from facial action units, GSR, EEG, and eye-tracking

Repository: https://github.com/meis7/AFFEC
Dataset: https://zenodo.org/records/14794876
"""

__version__ = "0.1.0"
__author__ = "AFFEC Contributors"

from affec.data.loader import ZenodoDataset, AFFECDataLoader
from affec.features.extractor import MultimodalFeatureExtractor, discretize_emotion
from affec.models.baseline import BaselineModel, MultimodalEvaluator, report_cv_results
from affec.utils.config import Config, TARGETS, MODALITIES
from affec.utils.logging import ExperimentLogger

__all__ = [
    'ZenodoDataset',
    'AFFECDataLoader',
    'MultimodalFeatureExtractor',
    'discretize_emotion',
    'BaselineModel',
    'MultimodalEvaluator',
    'report_cv_results',
    'Config',
    'TARGETS',
    'MODALITIES',
    'ExperimentLogger'
]
