"""
AFFEC Dataset - Zenodo downloader and data loader
"""
import os
import subprocess
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json

ZENODO_RECORD_ID = "14794876"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

class ZenodoDataset:
    """Handle downloading and extracting AFFEC dataset from Zenodo"""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize downloader
        
        Parameters:
        -----------
        output_dir : str
            Directory to store downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download(self, filename: str = "affec_dataset.tar.gz", force: bool = False):
        """
        Download dataset from Zenodo
        
        Parameters:
        -----------
        filename : str
            Name of file to download (default: affec_dataset.tar.gz)
        force : bool
            Re-download even if file exists
        """
        filepath = self.output_dir / filename
        
        if filepath.exists() and not force:
            print(f"✓ File exists: {filepath}")
            return filepath
        
        url = f"{ZENODO_URL}/{filename}"
        print(f"📥 Downloading from: {url}")
        
        try:
            subprocess.run(
                ["curl", "-L", url, "-o", str(filepath)],
                check=True,
                capture_output=True
            )
            print(f"✓ Downloaded to: {filepath}")
            return filepath
        except subprocess.CalledProcessError as e:
            print(f"✗ Download failed: {e}")
            return None
    
    def extract(self, filepath: str = None, remove_archive: bool = False):
        """
        Extract tar.gz archive
        
        Parameters:
        -----------
        filepath : str
            Path to archive file
        remove_archive : bool
            Delete archive after extraction
        """
        if filepath is None:
            filepath = list(self.output_dir.glob("*.tar.gz"))[0]
        
        filepath = Path(filepath)
        extract_path = self.output_dir / filepath.stem.replace(".tar", "")
        
        print(f"📦 Extracting to: {extract_path}")
        
        try:
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=extract_path)
            print(f"✓ Extracted successfully")
            
            if remove_archive:
                filepath.unlink()
                print(f"✓ Removed archive")
            
            return extract_path
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return None


class AFFECDataLoader:
    """Load and manage AFFEC dataset"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader
        
        Parameters:
        -----------
        data_dir : str
            Path to extracted AFFEC data directory
        """
        self.data_dir = Path(data_dir)
        self.participants_df = None
        self.metadata = {}
        self._videostream_header_cache: Dict[int, List[str]] = {}
        # Known-good fallback file from project communication (used only if a file's own JSON is inconsistent).
        self._fallback_videostream_json = (
            self.data_dir / "sub-afri" / "beh" / "sub-afri_task-fer_run-0_recording-videostream_physio.json"
        )

    @staticmethod
    def _normalize_participants_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize participant table column names and key demographics."""
        out = df.copy()
        out.columns = [str(c).strip().replace('"', '').replace("'", "") for c in out.columns]

        # Harmonize age column naming seen in AFFEC release (e.g., `"age "`).
        if 'age' not in out.columns:
            age_candidates = [c for c in out.columns if c.lower().startswith('age')]
            if age_candidates:
                out = out.rename(columns={age_candidates[0]: 'age'})

        if 'gender' in out.columns:
            out['gender'] = out['gender'].astype(str).str.strip().str.lower().map(
                {'m': 'male', 'male': 'male', 'f': 'female', 'female': 'female'}
            ).fillna('unknown')

        if 'age' in out.columns:
            out['age'] = pd.to_numeric(out['age'], errors='coerce')

        return out

    def _get_participant_profile(self, participant: str) -> Dict[str, float]:
        """Get personality + demographics for one participant."""
        default = {
            'personality_O': np.nan,
            'personality_C': np.nan,
            'personality_E': np.nan,
            'personality_A': np.nan,
            'personality_N': np.nan,
            'gender': 'unknown',
            'age': np.nan,
        }
        if self.participants_df is None:
            return default

        pdata = self.participants_df[self.participants_df['participant_id'] == participant]
        if pdata.empty:
            return default

        row = pdata.iloc[0]
        return {
            'personality_O': row.get('O', np.nan),
            'personality_C': row.get('C', np.nan),
            'personality_E': row.get('E', np.nan),
            'personality_A': row.get('A', np.nan),
            'personality_N': row.get('N', np.nan),
            'gender': row.get('gender', 'unknown'),
            'age': row.get('age', np.nan),
        }

    @staticmethod
    def _count_first_row_fields(tsv_gz_path: Path) -> int:
        """Count tab-separated fields in the first row of a gzipped TSV."""
        try:
            with gzip.open(tsv_gz_path, 'rt', encoding='utf-8', errors='ignore') as f:
                line = f.readline().rstrip('\n')
            if not line:
                return 0
            return line.count('\t') + 1
        except Exception:
            return 0

    @staticmethod
    def _read_columns_from_json(json_path: Path) -> List[str]:
        """Read the `Columns` list from a metadata JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            cols = data.get('Columns', [])
            return cols if isinstance(cols, list) else []
        except Exception:
            return []

    def _build_videostream_header_cache(self) -> None:
        """Cache known column schemas by length from all videostream JSON sidecars."""
        if self._videostream_header_cache:
            return

        for jp in self.data_dir.glob('sub-*/beh/*_recording-videostream_physio.json'):
            cols = self._read_columns_from_json(jp)
            if cols:
                self._videostream_header_cache.setdefault(len(cols), cols)

        # Ensure fallback JSON is considered first-class if available.
        if self._fallback_videostream_json.exists():
            cols = self._read_columns_from_json(self._fallback_videostream_json)
            if cols:
                self._videostream_header_cache[len(cols)] = cols

    def _resolve_videostream_headers(self, json_path: Path, tsv_gz_path: Path) -> Optional[List[str]]:
        """
        Resolve robust headers for videostream TSV.

        Handles known AFFEC issue where some per-run JSON files have wrong `Columns` length.
        """
        n_fields = self._count_first_row_fields(tsv_gz_path)
        if n_fields <= 0:
            return None

        cols = self._read_columns_from_json(json_path)
        if len(cols) == n_fields:
            return cols

        self._build_videostream_header_cache()

        # Prefer any schema that matches the physical file width.
        cached = self._videostream_header_cache.get(n_fields)
        if cached:
            return cached

        # If a run appears truncated (e.g., only 3 TSV fields), still allow parsing
        # with a known full template so `onset` can be read and run can be retained.
        # Downstream AU features for these runs will often be NaN and should be treated as partial-quality.
        if n_fields <= 3 and self._videostream_header_cache:
            fallback_len = max(self._videostream_header_cache.keys())
            return self._videostream_header_cache[fallback_len]

        # Last-resort: adapt local JSON columns length if possible.
        if cols:
            if len(cols) > n_fields:
                return cols[:n_fields]
            return cols + [f'col_{i}' for i in range(len(cols), n_fields)]

        return None
        
    def load_participants(self, filename: str = "participants.tsv"):
        """Load participant demographics and personality data"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            return None
        
        self.participants_df = pd.read_csv(filepath, sep="\t")
        self.participants_df = self._normalize_participants_columns(self.participants_df)
        print(f"✓ Loaded {len(self.participants_df)} participants")
        return self.participants_df
    
    def load_au_data(self, participant: str, run: int = 0):
        """
        Load Action Unit data for participant
        
        Parameters:
        -----------
        participant : str
            Participant ID (e.g., 'sub-001')
        run : int
            Run number (0-3)
            
        Returns:
        --------
        pd.DataFrame
            Action Unit data with timestamps and confidence
        """
        au_path = (
            self.data_dir / participant / "beh" /
            f"{participant}_task-fer_run-{run}_recording-videostream_physio.tsv.gz"
        )
        json_path = (
            self.data_dir / participant / "beh" /
            f"{participant}_task-fer_run-{run}_recording-videostream_physio.json"
        )
        
        if not au_path.exists() or not json_path.exists():
            return None
        
        needed_cols = {
            'onset', 'timestamp', 'confidence', 'success',
            'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
            'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
            'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
        }

        au_data = self._read_tsv_with_json_headers(au_path, json_path, needed_cols=needed_cols)
        if au_data is None or au_data.empty:
            return None
        return au_data

    def _read_tsv_with_json_headers(self, tsv_path: Path, json_path: Path, needed_cols: Optional[set] = None) -> Optional[pd.DataFrame]:
        """Read a gzipped TSV file using robust header resolution from sidecar JSON."""
        if not tsv_path.exists() or not json_path.exists():
            return None

        headers = self._resolve_videostream_headers(json_path, tsv_path)
        if headers is None:
            return None

        try:
            df = pd.read_csv(
                tsv_path,
                sep='\t',
                compression='gzip',
                header=None,
                engine='python',
                names=headers,
                on_bad_lines='skip',
                usecols=(lambda c: c in needed_cols) if needed_cols else None,
            )
        except Exception:
            return None

        time_col = 'onset' if 'onset' in df.columns else 'timestamp' if 'timestamp' in df.columns else None
        if time_col is None:
            return None

        if time_col != 'onset':
            df = df.rename(columns={time_col: 'onset'})

        for col in df.columns:
            if col == 'onset':
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['onset'] = pd.to_numeric(df['onset'], errors='coerce')
        df = df.dropna(subset=['onset'])
        if df.empty:
            return None
        df['onset'] = df['onset'] - df['onset'].iloc[0]
        return df

    def load_gaze_data(self, participant: str, run: int = 0) -> Optional[pd.DataFrame]:
        """Load eye-tracking gaze+pupil merged stream for a participant/run.

        Replicates the notebook's process_eye_data which merges the Gazepoint gaze
        stream (FPOG*, LPOG*, RPOG*, BPOG*) with the Gazepoint pupil stream
        (LPCX/Y, LPD, LPUPILD, RPCX/Y, RPD, RPUPILD, etc.) on `onset`.
        Both files live in the participant's own beh/ folder.
        """
        # Primary location: same beh/ folder as AU data
        base = self.data_dir / participant / 'beh'
        # Fallback: separate gaze/ subfolder used by some dataset releases
        gaze_base_fallback = self.data_dir / 'gaze' / participant / 'beh'

        def _find(stem: str) -> tuple:
            tsv = base / f"{participant}_task-fer_run-{run}_recording-{stem}_physio.tsv.gz"
            jsn = base / f"{participant}_task-fer_run-{run}_recording-{stem}_physio.json"
            if tsv.exists() and jsn.exists():
                return tsv, jsn
            tsv = gaze_base_fallback / f"{participant}_task-fer_run-{run}_recording-{stem}_physio.tsv.gz"
            jsn = gaze_base_fallback / f"{participant}_task-fer_run-{run}_recording-{stem}_physio.json"
            return (tsv, jsn) if tsv.exists() and jsn.exists() else (None, None)

        gaze_tsv, gaze_json = _find('gaze')
        pupil_tsv, pupil_json = _find('pupil')

        gaze_needed = {
            'onset', 'TIME', 'FPOGX', 'FPOGY', 'FPOGS', 'FPOGD', 'FPOGID', 'FPOGV',
            'LPOGX', 'LPOGY', 'LPOGV', 'RPOGX', 'RPOGY', 'RPOGV', 'BPOGX', 'BPOGY', 'BPOGV'
        }
        pupil_needed = {
            'onset', 'LPCX', 'LPCY', 'LPD', 'LPS', 'LPV',
            'RPCX', 'RPCY', 'RPD', 'RPS', 'RPV',
            'LEYEX', 'LEYEY', 'LEYEZ', 'LPUPILD', 'LPUPILV',
            'REYEX', 'REYEY', 'REYEZ', 'RPUPILD', 'RPUPILV',
        }

        gaze_df = self._read_tsv_with_json_headers(gaze_tsv, gaze_json, needed_cols=gaze_needed) if gaze_tsv else None
        pupil_df = self._read_tsv_with_json_headers(pupil_tsv, pupil_json, needed_cols=pupil_needed) if pupil_tsv else None

        if gaze_df is None and pupil_df is None:
            return None
        if gaze_df is None:
            return pupil_df
        if pupil_df is None:
            return gaze_df

        # Drop duplicate TIME column from pupil before merge (notebook does this)
        pupil_df = pupil_df.drop(columns=['TIME'], errors='ignore')

        merged = pd.merge_asof(
            gaze_df.sort_values('onset'),
            pupil_df.sort_values('onset'),
            on='onset',
            direction='backward',
        )
        return merged

    def load_gsr_data(self, participant: str, run: int = 0) -> Optional[pd.DataFrame]:
        """Load GSR stream for a participant/run.

        Tries the participant's own beh/ folder first (notebook path),
        then falls back to the separate gsr/ subfolder.
        """
        def _paths(base: Path):
            tsv = base / f"{participant}_task-fer_run-{run}_recording-gsr_physio.tsv.gz"
            jsn = base / f"{participant}_task-fer_run-{run}_recording-gsr_physio.json"
            return tsv, jsn

        tsv, jsn = _paths(self.data_dir / participant / 'beh')
        if not tsv.exists():
            tsv, jsn = _paths(self.data_dir / 'gsr' / participant / 'beh')

        needed_cols = {
            'onset', 'GSR_raw', 'GSR_cal', 'GSR_Conductance_cal',
            'Temperature_cal',
            'Low_Noise_Accelerometer_X_cal', 'Low_Noise_Accelerometer_Y_cal', 'Low_Noise_Accelerometer_Z_cal',
        }
        return self._read_tsv_with_json_headers(tsv, jsn, needed_cols=needed_cols)

    @staticmethod
    def _slice_window(df: Optional[pd.DataFrame], start: float, end: float) -> pd.DataFrame:
        """Slice modality stream in [start, end]."""
        if df is None or df.empty or 'onset' not in df.columns:
            return pd.DataFrame()
        sliced = df[(df['onset'] >= start) & (df['onset'] <= end)].copy()
        return sliced.reset_index(drop=True)
    
    def load_events(self, participant: str, run: int = 0):
        """Load event markers (emotion stimuli, timing)"""
        events_path = (
            self.data_dir / participant /
            f"{participant}_task-fer_run-{run}_events.tsv"
        )
        
        if not events_path.exists():
            return None
        
        events = pd.read_csv(events_path, sep='\t')
        if 'onset' not in events.columns:
            return None
        events['onset'] = pd.to_numeric(events['onset'], errors='coerce')
        return events.dropna(subset=['onset'])
    
    def load_labels(self, participant: str, run: int = 0):
        """Load behavioral labels (emotions, arousal, valence)"""
        labels_path = (
            self.data_dir / participant / "beh" /
            f"{participant}_task-fer_run-{run}_beh.tsv"
        )
        
        if not labels_path.exists():
            return None
        
        labels = pd.read_csv(labels_path, sep='\t')
        return labels
    
    def merge_trial_data(self, participant: str, run: int = 0):
        """
        Merge AU, events, and labels for all trials
        
        Returns:
        --------
        list of dict
            Each dict contains merged data for one trial
        """
        au_data = self.load_au_data(participant, run)
        gaze_data = self.load_gaze_data(participant, run)
        gsr_data = self.load_gsr_data(participant, run)
        events = self.load_events(participant, run)
        labels = self.load_labels(participant, run)

        # Labels and event timing are mandatory; modalities may be partially missing.
        if any(x is None for x in [events, labels]):
            return []

        # Prepare modality streams
        if au_data is not None and not au_data.empty:
            au_data = au_data.dropna(subset=['onset']).sort_values('onset')
        if gaze_data is not None and not gaze_data.empty:
            gaze_data = gaze_data.dropna(subset=['onset']).sort_values('onset')
        if gsr_data is not None and not gsr_data.empty:
            gsr_data = gsr_data.dropna(subset=['onset']).sort_values('onset')

        events = events.dropna(subset=['onset']).sort_values('onset')
        if events.empty:
            return []

        # Select video window events to align trial modality segments.
        if 'flag' in events.columns:
            video_events = events[events['flag'].astype(str).str.lower() == 'video'].copy()
        else:
            video_events = events.copy()

        if 'duration' in video_events.columns:
            video_events['duration'] = pd.to_numeric(video_events['duration'], errors='coerce').fillna(3.0)
        else:
            video_events['duration'] = 3.0
        
        profile = self._get_participant_profile(participant)
        personality = {
            'O': profile['personality_O'],
            'C': profile['personality_C'],
            'E': profile['personality_E'],
            'A': profile['personality_A'],
            'N': profile['personality_N'],
        }
        
        # Merge with labels by trial
        trials = []
        if 'stim_file' not in labels.columns:
            return []

        for stim_file in labels['stim_file'].dropna().unique():
            trial_label = labels[labels['stim_file'] == stim_file]
            if trial_label.empty:
                continue

            trial_event = video_events[video_events['stim_file'] == stim_file]
            if trial_event.empty:
                # Fall back to any event row for this stimulus.
                trial_event = events[events['stim_file'] == stim_file]
                if trial_event.empty:
                    continue

            onset = float(pd.to_numeric(trial_event['onset'], errors='coerce').dropna().iloc[0])
            duration = float(pd.to_numeric(trial_event['duration'], errors='coerce').dropna().iloc[0]) if 'duration' in trial_event.columns else 3.0
            offset = onset + max(duration, 0.1)

            trial_au = self._slice_window(au_data, onset, offset)
            trial_gaze = self._slice_window(gaze_data, onset, offset)
            # GSR uses a longer integration window to capture delayed autonomic response.
            trial_gsr = self._slice_window(gsr_data, onset, offset + 10.0)
            
            trial_dict = {
                'participant': participant,
                'run': run,
                'stim_file': stim_file,
                'trial': trial_label['trial'].values[0],
                'stimulus_emotion': trial_label['trial_type'].values[0],
                'perceived_arousal': trial_label['p_emotion_a'].values[0],
                'perceived_valence': trial_label['p_emotion_v'].values[0],
                'felt_arousal': trial_label['f_emotion_a'].values[0],
                'felt_valence': trial_label['f_emotion_v'].values[0],
                'au_data': trial_au,
                'gaze_data': trial_gaze,
                'gsr_data': trial_gsr,
                'personality': personality,
                'gender': profile['gender'],
                'age': profile['age'],
            }
            trials.append(trial_dict)
        
        return trials
