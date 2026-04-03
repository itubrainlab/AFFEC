"""
Model training and evaluation for multimodal emotion classification
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None


class BaselineModel:
    """
    Baseline model with 5-fold cross-validation.

    Supports two backends:
    - **RandomForest** (default): sklearn RandomForestClassifier with class_weight='balanced'.
    - **AutoGluon** (``use_autogluon=True``): TabularPredictor with ``best_quality`` preset,
      selecting the best non-ensemble model per fold — matching the notebook's
      ``retrain_best_model_kfold`` routine.

    References:
    -----------
    - STATISTICAL_ANALYSIS.md: Section on 5-fold CV and metric reporting
    - BASELINE_CONTEXT.md: Feature importance and baseline performance benchmarks
    """

    def __init__(
        self,
        target: str = 'perceived_arousal',
        n_folds: int = 5,
        random_state: int = 42,
        use_autogluon: bool = False,
    ):
        """
        Initialize baseline model

        Parameters:
        -----------
        target : str
            Target variable ('perceived_arousal', 'perceived_valence',
            'felt_arousal', 'felt_valence')
        n_folds : int
            Number of folds for cross-validation (default: 5)
        random_state : int
            Random seed for reproducibility
        use_autogluon : bool
            If True use AutoGluon TabularPredictor (best_quality preset)
            instead of RandomForest.
        """
        self.target = target
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_autogluon = use_autogluon
        self.cv_results = []
        
    def _get_feature_columns(self, X: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude metadata)"""
        exclude = ['participant', 'run', 'stimulus_emotion', 
                   'perceived_arousal', 'perceived_valence', 'felt_arousal', 'felt_valence',
                   'gender', 'age', 'age_group', 'stim_file', 'trial']
        candidate_cols = [col for col in X.columns if col not in exclude]
        numeric_cols = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(X[col])]
        return numeric_cols
    
    @staticmethod
    def _safe_float(value: float) -> float:
        return float(np.round(value, 6))

    @staticmethod
    def _build_group_splitter(y: pd.Series, groups: Optional[pd.Series], n_folds: int, random_state: int):
        if groups is None:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            return skf.split(np.zeros(len(y)), y)

        if StratifiedGroupKFold is not None:
            sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            return sgkf.split(np.zeros(len(y)), y, groups=groups)

        gkf = GroupKFold(n_splits=n_folds)
        return gkf.split(np.zeros(len(y)), y, groups=groups)

    @staticmethod
    def _compute_stratified_oof_metrics(oof_df: pd.DataFrame, col: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        valid = oof_df.dropna(subset=[col])
        if valid.empty:
            return out

        for group, gdf in valid.groupby(col):
            if len(gdf) < 20:
                continue
            out[str(group)] = {
                'n': int(len(gdf)),
                'f1_macro': float(f1_score(gdf['y_true'], gdf['y_pred'], average='macro')),
                'accuracy': float(accuracy_score(gdf['y_true'], gdf['y_pred'])),
            }
        return out

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       stratify_by: Optional[str] = None,
                       group_column: str = 'participant') -> Dict:
        """
        Perform cross-validation
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target labels
        stratify_by : str
            Column to stratify by (e.g., 'participant' for leave-participant-out)
            
        Returns:
        --------
        dict
            CV results with metrics (f1_macro, f1_weighted, accuracy) mean ± std
        """
        feature_cols = self._get_feature_columns(X)
        if not feature_cols:
            raise ValueError("No numeric feature columns available after metadata exclusion")

        X_features = X[feature_cols].apply(pd.to_numeric, errors='coerce')
        X_features = X_features.fillna(X_features.mean(numeric_only=True)).fillna(0.0)
        groups = X[group_column] if group_column in X.columns else None

        effective_folds = self.n_folds
        if groups is not None:
            n_groups = int(pd.Series(groups).nunique())
            effective_folds = max(2, min(self.n_folds, n_groups))
        if len(pd.Series(y).dropna().unique()) < 2:
            raise ValueError("Need at least two classes for cross-validation")

        splits = self._build_group_splitter(y=y, groups=groups, n_folds=effective_folds, random_state=self.random_state)
        
        fold_metrics = []
        oof_records: List[Dict] = []
        all_classes = sorted([int(c) for c in pd.Series(y).dropna().unique().tolist()])
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(X_test) == 0:
                continue

            if self.use_autogluon:
                y_pred = self._autogluon_fold(
                    X_train, y_train, X_test, feature_cols
                )
            else:
                y_pred = self._rf_fold(X_train, y_train, X_test)

            class_f1 = {}
            for c in all_classes:
                class_f1[str(c)] = float(
                    f1_score((y_test == c).astype(int), (y_pred == c).astype(int), zero_division=0)
                )
            
            metrics = {
                'fold': fold,
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'accuracy': accuracy_score(y_test, y_pred),
                'n_train': len(y_train),
                'n_test': len(y_test),
                'class_f1': class_f1,
                'n_participants_train': int(X.iloc[train_idx]['participant'].nunique()) if 'participant' in X.columns else None,
                'n_participants_test': int(X.iloc[test_idx]['participant'].nunique()) if 'participant' in X.columns else None,
            }
            fold_metrics.append(metrics)

            fold_meta = X.iloc[test_idx][['participant', 'gender', 'age_group']].copy() if {'participant', 'gender', 'age_group'}.issubset(X.columns) else X.iloc[test_idx][['participant']].copy() if 'participant' in X.columns else pd.DataFrame(index=X.iloc[test_idx].index)
            fold_meta['y_true'] = y_test.values
            fold_meta['y_pred'] = y_pred
            fold_meta['fold'] = fold
            oof_records.extend(fold_meta.to_dict(orient='records'))
        
        # Aggregate results
        self.cv_results = fold_metrics

        classwise_summary = {}
        for c in all_classes:
            vals = [m['class_f1'].get(str(c), np.nan) for m in fold_metrics]
            vals = [v for v in vals if not pd.isna(v)]
            classwise_summary[str(c)] = {
                'mean': self._safe_float(float(np.mean(vals))) if vals else np.nan,
                'std': self._safe_float(float(np.std(vals))) if vals else np.nan,
            }

        oof_df = pd.DataFrame(oof_records)
        stratified_metrics = {}
        for col in ['gender', 'age_group']:
            if col in oof_df.columns:
                stratified_metrics[col] = self._compute_stratified_oof_metrics(oof_df, col)
        
        results = {
            'target': self.target,
            'n_folds': self.n_folds,
            'split_strategy': 'subject_disjoint_grouped_cv' if groups is not None else 'stratified_cv',
            'f1_macro_mean': self._safe_float(np.mean([m['f1_macro'] for m in fold_metrics])),
            'f1_macro_std': self._safe_float(np.std([m['f1_macro'] for m in fold_metrics])),
            'f1_weighted_mean': self._safe_float(np.mean([m['f1_weighted'] for m in fold_metrics])),
            'f1_weighted_std': self._safe_float(np.std([m['f1_weighted'] for m in fold_metrics])),
            'accuracy_mean': self._safe_float(np.mean([m['accuracy'] for m in fold_metrics])),
            'accuracy_std': self._safe_float(np.std([m['accuracy'] for m in fold_metrics])),
            'chance_level': self._safe_float(1.0 / max(len(all_classes), 1)),
            'classwise_f1': classwise_summary,
            'stratified_metrics': stratified_metrics,
            'fold_metrics': fold_metrics,
            'oof_predictions': oof_df.to_dict(orient='records'),
        }
        
        return results
    
    # ------------------------------------------------------------------
    # Private training helpers
    # ------------------------------------------------------------------

    def _rf_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """Train RandomForest on one fold and return predictions."""
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=self.random_state,
        )
        model.fit(X_tr, y_train)
        return model.predict(X_te)

    def _autogluon_fold(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Train AutoGluon TabularPredictor on one fold and return predictions.

        Replicates the notebook's ``retrain_best_model_kfold``:
        1. Fit with ``best_quality`` preset, excluding KNN.
        2. Pick the best non-ensemble model from the leaderboard.
        3. Re-train that model on the full training fold.
        4. Return predictions on the test fold.
        """
        try:
            from autogluon.tabular import TabularPredictor  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "AutoGluon is not installed. "
                "Run: pip install autogluon.tabular"
            ) from exc

        import tempfile, os  # noqa: PLC0415

        _LABEL = '_target_'
        train_df = X_train.copy()
        train_df[_LABEL] = y_train.values

        with tempfile.TemporaryDirectory() as tmp_dir:
            predictor = TabularPredictor(
                label=_LABEL,
                eval_metric='f1_macro',
                path=os.path.join(tmp_dir, 'ag'),
                verbosity=0,
            ).fit(
                train_df,
                presets='best_quality',
                excluded_model_types=['KNN'],
            )

            # Select best non-ensemble model (matching notebook's logic)
            board = predictor.leaderboard(silent=True)
            non_ens = board[~board['model'].str.contains('WeightedEnsemble', na=False)]
            if non_ens.empty:
                non_ens = board
            best_model_name = non_ens.sort_values('score_val', ascending=False).iloc[0]['model']

            y_pred = predictor.predict(X_test, model=best_model_name).to_numpy()

        return y_pred

    def get_fold_results_df(self) -> pd.DataFrame:
        """Return cross-validation results as DataFrame"""
        return pd.DataFrame(self.cv_results)


class MultimodalEvaluator:
    """
    Evaluate model performance with fairness metrics
    
    References:
    -----------
    - DATASET_BIAS.md: Section on demographic stratification (gender, age)
    - STATISTICAL_ANALYSIS.md: Effect sizes and reporting guidelines
    """
    
    @staticmethod
    def stratified_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          strata: pd.Series) -> Dict:
        """
        Compute performance metrics stratified by demographic variable
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        strata : pd.Series
            Stratification variable (e.g., gender, age group)
            
        Returns:
        --------
        dict
            Per-stratum metrics
        """
        results = {}
        overall = {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'accuracy': accuracy_score(y_true, y_pred)
        }
        results['overall'] = overall
        
        # Per-stratum metrics
        for group in strata.unique():
            mask = strata == group
            if mask.sum() < 5:  # Skip small groups
                continue
            
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            results[f'{group}'] = {
                'f1_macro': f1_score(y_true_group, y_pred_group, average='macro'),
                'f1_weighted': f1_score(y_true_group, y_pred_group, average='weighted'),
                'accuracy': accuracy_score(y_true_group, y_pred_group),
                'n': mask.sum()
            }
        
        return results
    
    @staticmethod
    def effect_size(model_metric: float, baseline_metric: float) -> Dict:
        """
        Compute effect size (Cohen's d approximation) vs baseline
        
        References:
        -----------
        STATISTICAL_ANALYSIS.md: Section 3 on effect size computation
        """
        # Approximation: treat accuracy/F1 as standardized difference
        effect_size = model_metric - baseline_metric
        
        interpretation = 'small' if abs(effect_size) < 0.2 else \
                        'medium' if abs(effect_size) < 0.5 else 'large'
        
        return {
            'effect_size': effect_size,
            'interpretation': interpretation,
            'percent_improvement': 100 * (model_metric - baseline_metric) / baseline_metric if baseline_metric > 0 else 0
        }


def report_cv_results(results: Dict, target: str) -> str:
    """
    Format cross-validation results for reporting
    
    References:
    -----------
    - BASELINE_CONTEXT.md: Reporting format and metric interpretation
    - STATISTICAL_ANALYSIS.md: Guidelines for metric reporting
    
    Parameters:
    -----------
    results : dict
        Output from BaselineModel.cross_validate()
    target : str
        Target variable name
        
    Returns:
    --------
    str
        Formatted report
    """
    report = f"""
=== {target} ===
Split:        {results.get('split_strategy', 'n/a')}
Chance:       {results.get('chance_level', float('nan')):.3f}
Macro F1:     {results['f1_macro_mean']:.3f} ± {results['f1_macro_std']:.3f}
Weighted F1:  {results['f1_weighted_mean']:.3f} ± {results['f1_weighted_std']:.3f}
Accuracy:     {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}
n_folds:      {results['n_folds']}
"""
    classwise = results.get('classwise_f1', {})
    if classwise:
        report += "Class-wise F1:\n"
        for cls, stat in classwise.items():
            report += f"  class {cls}: {stat['mean']:.3f} ± {stat['std']:.3f}\n"
    return report.strip()
