"""
Feature extraction from multimodal data
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ActionUnitFeatureExtractor:
    """Extract features from facial Action Units"""
    
    AU_LIST = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
               'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
               'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
    
    @staticmethod
    def extract_statistics(au_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract statistical features from AU data
        
        Parameters:
        -----------
        au_data : pd.DataFrame
            Action Unit timeseries data
            
        Returns:
        --------
        dict
            Features: mean, std, min, max for each AU
        """
        features = {}
        
        for au in ActionUnitFeatureExtractor.AU_LIST:
            if au not in au_data.columns:
                continue
            
            vals = au_data[au].dropna()
            if len(vals) == 0:
                continue
            
            features[f"{au}_mean"] = vals.mean()
            features[f"{au}_std"] = vals.std()
            features[f"{au}_min"] = vals.min()
            features[f"{au}_max"] = vals.max()
        
        return features

    @staticmethod
    def extract_temporal(au_data: pd.DataFrame, window: float = 3.0) -> Dict[str, float]:
        """
        Extract temporal features (duration, onset latency, etc.)

        Parameters:
        -----------
        au_data : pd.DataFrame
            Action Unit timeseries data
        window : float
            Time window in seconds

        Returns:
        --------
        dict
            Temporal features
        """
        if len(au_data) == 0:
            return {}

        features = {}
        features['duration'] = au_data['onset'].max() - au_data['onset'].min()
        features['n_frames'] = len(au_data)
        features['confidence_mean'] = au_data['confidence'].mean() if 'confidence' in au_data else np.nan
        features['success_rate'] = au_data['success'].mean() if 'success' in au_data else np.nan

        return features


class EyeFeatureExtractor:
    """Extract summary eye-tracking features from merged gaze+pupil stream.

    Column list matches notebook's Eye_columns:
    FPOGX, FPOGY, LPOGX, LPOGY, RPOGX, RPOGY, FPOGD,
    LPD, RPD, LPUPILD, RPUPILD (+centre coords from pupil file).
    """

    EYE_COLS = [
        # Gaze columns
        'FPOGX', 'FPOGY', 'FPOGD', 'FPOGV',
        'LPOGX', 'LPOGY', 'LPOGV',
        'RPOGX', 'RPOGY', 'RPOGV',
        'BPOGX', 'BPOGY', 'BPOGV',
        # Pupil-diameter and pupil-centre columns (from pupil physio file)
        'LPD', 'RPD', 'LPUPILD', 'RPUPILD',
        'LPCX', 'LPCY', 'RPCX', 'RPCY',
    ]

    @staticmethod
    def extract_statistics(gaze_data: pd.DataFrame) -> Dict[str, float]:
        if gaze_data is None or len(gaze_data) == 0:
            return {}

        features: Dict[str, float] = {}
        for col in EyeFeatureExtractor.EYE_COLS:
            if col not in gaze_data.columns:
                continue
            vals = pd.to_numeric(gaze_data[col], errors='coerce').dropna()
            if vals.empty:
                continue
            features[f"eye_{col}_mean"] = float(vals.mean())
            features[f"eye_{col}_std"] = float(vals.std())
            features[f"eye_{col}_min"] = float(vals.min())
            features[f"eye_{col}_max"] = float(vals.max())

        if 'FPOGID' in gaze_data.columns:
            fix_ids = pd.to_numeric(gaze_data['FPOGID'], errors='coerce').dropna()
            if not fix_ids.empty:
                features['eye_fixation_count'] = float(fix_ids.nunique())

        features['eye_n_samples'] = float(len(gaze_data))
        return features


class GSRFeatureExtractor:
    """Extract GSR/EDA features using neurokit2 SCR decomposition.

    Replicates the notebook's process_gsr_data which calls
    ``nk.eda_process(nk.standardize(signal), sampling_rate, method='neurokit')``
    and extracts 31 features:
    - SCR count
    - Per-SCR event statistics (mean / median / min / max / std) for:
      Onsets, Amplitude, Height, RiseTime, Recovery, RecoveryTime
    Falls back to raw conductance statistics when neurokit2 is unavailable
    or the signal is too short to decompose.
    """

    # Default GSR sampling rate (Hz) from AFFEC dataset JSON sidecars
    DEFAULT_SR: int = 10

    # Columns we try to read raw stats from (fallback path)
    GSR_COLS = [
        'GSR_Conductance_cal', 'GSR_cal', 'GSR_raw',
        'Temperature_cal',
        'Low_Noise_Accelerometer_X_cal',
        'Low_Noise_Accelerometer_Y_cal',
        'Low_Noise_Accelerometer_Z_cal',
    ]

    # SCR event-level columns extracted from nk.eda_process output
    SCR_STAT_COLS = [
        'SCR_Onsets', 'SCR_Amplitude', 'SCR_Height',
        'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime',
    ]

    @staticmethod
    def _scr_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
        """Return 5 statistics (mean/median/min/max/std) for an SCR attribute array."""
        v = values[~np.isnan(values)]
        if v.size == 0:
            return {f"gsr_{prefix}_{s}": np.nan for s in ('mean', 'median', 'min', 'max', 'std')}
        return {
            f"gsr_{prefix}_mean":   float(np.mean(v)),
            f"gsr_{prefix}_median": float(np.median(v)),
            f"gsr_{prefix}_min":    float(np.min(v)),
            f"gsr_{prefix}_max":    float(np.max(v)),
            f"gsr_{prefix}_std":    float(np.std(v)),
        }

    @staticmethod
    def extract_statistics(
        gsr_data: pd.DataFrame,
        sampling_rate: int = DEFAULT_SR,
    ) -> Dict[str, float]:
        """Extract GSR features.

        Tries neurokit2 SCR decomposition first (matching paper Table 3).
        Falls back to raw conductance statistics if neurokit2 is not installed
        or the window is too short.

        Parameters
        ----------
        gsr_data : pd.DataFrame
            Time-windowed GSR stream with an `onset` column and conductance columns.
        sampling_rate : int
            Signal sampling rate in Hz (default 10 Hz for AFFEC dataset).
        """
        if gsr_data is None or len(gsr_data) == 0:
            return {}

        features: Dict[str, float] = {}
        features['gsr_n_samples'] = float(len(gsr_data))

        cond_col = next(
            (c for c in ('GSR_Conductance_cal', 'GSR_cal', 'GSR_raw') if c in gsr_data.columns),
            None,
        )
        if cond_col is None:
            return features

        raw_signal = pd.to_numeric(gsr_data[cond_col], errors='coerce').dropna().to_numpy()
        if raw_signal.size < 4:
            return features

        # ── Attempt neurokit2 SCR decomposition ──────────────────────────────
        try:
            import warnings  # noqa: PLC0415
            import neurokit2 as nk  # noqa: PLC0415  (imported inside to keep optional)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                standardised = nk.standardize(raw_signal)
                eda_signals, _ = nk.eda_process(
                    standardised,
                    sampling_rate=int(sampling_rate),
                    method='neurokit',
                )

            # Locate SCR events (onset rows)
            onset_mask = eda_signals.get('SCR_Onsets', pd.Series(dtype=float)).fillna(0).astype(bool)
            n_peaks = int(onset_mask.sum())
            features['gsr_scr_n_peaks'] = float(n_peaks)

            for col in GSRFeatureExtractor.SCR_STAT_COLS:
                if col in eda_signals.columns:
                    event_vals = eda_signals[col][onset_mask].to_numpy(dtype=float)
                else:
                    event_vals = np.array([], dtype=float)
                features.update(GSRFeatureExtractor._scr_stats(event_vals, col))

            return features

        except Exception as exc:  # neurokit2 missing, signal too noisy, etc.
            logger.debug("neurokit2 GSR decomposition failed (%s) — using raw stats.", exc)

        # ── Fallback: raw conductance statistics ──────────────────────────────
        for col in GSRFeatureExtractor.GSR_COLS:
            if col not in gsr_data.columns:
                continue
            vals = pd.to_numeric(gsr_data[col], errors='coerce').dropna()
            if vals.empty:
                continue
            features[f"gsr_{col}_mean"] = float(vals.mean())
            features[f"gsr_{col}_std"]  = float(vals.std())
            features[f"gsr_{col}_min"]  = float(vals.min())
            features[f"gsr_{col}_max"]  = float(vals.max())

        # Simple peak count from derivative
        d1 = np.diff(raw_signal)
        peaks = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0]
        features['gsr_scr_peak_count_simple'] = float(len(peaks))
        return features
    
class MultimodalFeatureExtractor:
    """Extract and combine features from all modalities"""
    
    def __init__(self, use_modalities: Dict[str, bool] = None):
        """
        Initialize feature extractor
        
        Parameters:
        -----------
        use_modalities : dict
            Which modalities to use {'au': bool, 'gsr': bool, 'eye': bool, 'personality': bool}
        """
        if use_modalities is None:
            use_modalities = {
                'au': True,
                'gsr': False,  # Not always available
                'eye': False,  # Not always available
                'personality': True
            }
        self.use_modalities = use_modalities
    
    def extract_trial_features(self, trial_dict: Dict) -> Dict[str, float]:
        """
        Extract features for a single trial
        
        Parameters:
        -----------
        trial_dict : dict
            Trial data from AFFECDataLoader.merge_trial_data()
            
        Returns:
        --------
        dict
            Combined feature vector
        """
        features = {}
        
        # Add trial metadata
        features['participant'] = trial_dict['participant']
        features['run'] = trial_dict['run']
        features['stimulus_emotion'] = trial_dict['stimulus_emotion']
        features['gender'] = trial_dict.get('gender', 'unknown')
        features['age'] = trial_dict.get('age', np.nan)

        age_val = pd.to_numeric(pd.Series([features['age']]), errors='coerce').iloc[0]
        if pd.isna(age_val):
            features['age_group'] = 'unknown'
        elif age_val < 25:
            features['age_group'] = '18-24'
        elif age_val < 30:
            features['age_group'] = '25-29'
        else:
            features['age_group'] = '30+'
        
        # Target variables (labels)
        features['perceived_arousal'] = trial_dict['perceived_arousal']
        features['perceived_valence'] = trial_dict['perceived_valence']
        features['felt_arousal'] = trial_dict['felt_arousal']
        features['felt_valence'] = trial_dict['felt_valence']
        
        # Action Unit features
        if self.use_modalities.get('au', True):
            au_data = trial_dict.get('au_data', pd.DataFrame())
            au_features = ActionUnitFeatureExtractor.extract_statistics(au_data)
            au_temporal = ActionUnitFeatureExtractor.extract_temporal(au_data)
            features.update(au_features)
            features.update(au_temporal)

        # Eye-tracking features
        if self.use_modalities.get('eye', False):
            eye_data = trial_dict.get('gaze_data', pd.DataFrame())
            eye_features = EyeFeatureExtractor.extract_statistics(eye_data)
            features.update(eye_features)

        # GSR features
        if self.use_modalities.get('gsr', False):
            gsr_data = trial_dict.get('gsr_data', pd.DataFrame())
            gsr_features = GSRFeatureExtractor.extract_statistics(gsr_data)
            features.update(gsr_features)
        
        # Personality features
        if self.use_modalities.get('personality', True):
            personality = trial_dict['personality']
            for trait, value in personality.items():
                features[f'personality_{trait}'] = value
        
        return features
    
    @staticmethod
    def extract_batch_features(trials: List[Dict], use_modalities: Dict[str, bool] = None) -> pd.DataFrame:
        """
        Extract features for multiple trials
        
        Parameters:
        -----------
        trials : list of dict
            Trial data from AFFECDataLoader
        use_modalities : dict
            Which modalities to use
            
        Returns:
        --------
        pd.DataFrame
            Feature matrix (n_trials × n_features)
        """
        extractor = MultimodalFeatureExtractor(use_modalities)
        features_list = []
        
        for trial in trials:
            trial_features = extractor.extract_trial_features(trial)
            features_list.append(trial_features)
        
        return pd.DataFrame(features_list)


def discretize_emotion(values: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """
    Discretize continuous emotion ratings (1-9) into bins (Low, Medium, High)
    
    Parameters:
    -----------
    values : np.ndarray
        Continuous emotion ratings
    n_bins : int
        Number of bins (default: 3 for Low/Medium/High)
        
    Returns:
    --------
    np.ndarray
        Discretized labels
    """
    return pd.cut(values, bins=n_bins, labels=False)


def split_data_stratified(X: pd.DataFrame, y: pd.DataFrame, 
                         test_size: float = 0.2, val_size: float = 0.1,
                         random_state: int = 42) -> Tuple:
    """
    Split data into train/val/test with stratification by participant
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.DataFrame
        Labels
    test_size : float
        Fraction for test set
    val_size : float
        Fraction for validation set
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test (stratified by participant)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=X['participant'] if 'participant' in X.columns else None
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=X_temp['participant'] if 'participant' in X_temp.columns else None
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
