"""
Configuration and constants for AFFEC analysis
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Config:
    """
    Central configuration for AFFEC experiment
    
    References:
    -----------
    - METHODOLOGICAL_CHOICES.md: Justification for window sizes and modality parameters
    """
    
    # Data paths
    DATA_DIR: str = "data/raw"
    ZENODO_RECORD_ID: str = "14794876"
    
    # Processing parameters per METHODOLOGICAL_CHOICES.md
    EEG_WINDOW: float = 3.0  # seconds, captures rapid event-related neural dynamics
    AU_WINDOW: float = 3.0  # seconds, matches video duration
    GSR_WINDOW: float = 10.0  # seconds, justified for tonic component
    EYE_WINDOW: float = 3.0  # seconds
    
    # Emotion discretization
    EMOTION_BINS: int = 3  # Low/Medium/High
    EMOTION_LABELS: list = ('Low', 'Medium', 'High')
    
    # Cross-validation setup per STATISTICAL_ANALYSIS.md
    N_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.1
    RANDOM_STATE: int = 42
    
    # Feature extraction
    USE_MODALITIES: Dict[str, bool] = None
    
    # Model training
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 10
    
    def __post_init__(self):
        """Initialize default modality settings"""
        if self.USE_MODALITIES is None:
            self.USE_MODALITIES = {
                'au': True,
                'gsr': False,
                'eye': False,
                'personality': True
            }
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging"""
        return {
            'data_dir': self.DATA_DIR,
            'eeg_window': self.EEG_WINDOW,
            'au_window': self.AU_WINDOW,
            'gsr_window': self.GSR_WINDOW,
            'emotion_bins': self.EMOTION_BINS,
            'n_folds': self.N_FOLDS,
            'modalities': self.USE_MODALITIES
        }


# Target variables per study design
TARGETS = {
    'perceived_arousal': 'Perceived Arousal (1-9 rating)',
    'perceived_valence': 'Perceived Valence (1-9 rating)',
    'felt_arousal': 'Felt Arousal (1-9 rating)',
    'felt_valence': 'Felt Valence (1-9 rating)'
}

# Modalities in AFFEC dataset
MODALITIES = {
    'au': 'Facial Action Units (OpenFace)',
    'gsr': 'Galvanic Skin Response',
    'eeg': 'Electroencephalography (EEG)',
    'eye': 'Eye Tracking',
    'personality': 'Big Five personality traits'
}

# Demographic information (per DATASET_BIAS.md)
DEMOGRAPHICS = {
    'gender_distribution': '52M / 20F (72.2% / 27.8%)',
    'age_range': '18-35 years',
    'nationality': 'Predominantly Danish',
    'scope': 'Research-grade (not population representative)'
}
