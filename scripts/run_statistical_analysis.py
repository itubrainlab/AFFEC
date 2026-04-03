"""Run AFFEC statistical analysis with mixed-effects models for personality-emotion relations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affec.data.loader import AFFECDataLoader

try:
    import statsmodels.formula.api as smf
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "statsmodels is required for mixed-effects analysis. "
        "Install it with: pip install statsmodels"
    ) from exc


TARGETS = [
    "perceived_arousal",
    "perceived_valence",
    "felt_arousal",
    "felt_valence",
]
TRAITS = ["O", "C", "E", "A", "N"]


def _load_trials(loader: AFFECDataLoader, participant_ids: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for participant in participant_ids:
        for run in range(4):
            for trial in loader.merge_trial_data(participant, run):
                row = {
                    "participant": trial["participant"],
                    "run": trial["run"],
                    "trial": trial["trial"],
                    "gender": trial.get("gender", "unknown"),
                    "age": trial.get("age", np.nan),
                }
                for target in TARGETS:
                    row[target] = pd.to_numeric(trial.get(target, np.nan), errors="coerce")
                personality = trial.get("personality", {})
                for trait in TRAITS:
                    row[trait] = pd.to_numeric(personality.get(trait, np.nan), errors="coerce")
                rows.append(row)
    return pd.DataFrame(rows)


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def _fit_mixed_model(df: pd.DataFrame, outcome: str) -> Dict:
    model_df = df[["participant", outcome] + TRAITS].dropna().copy()
    if model_df.empty:
        return {"outcome": outcome, "status": "failed", "reason": "no valid rows"}

    model_df[f"{outcome}_z"] = _zscore(model_df[outcome])
    for trait in TRAITS:
        model_df[f"{trait}_z"] = _zscore(model_df[trait])

    model_df = model_df.dropna(subset=[f"{outcome}_z"] + [f"{t}_z" for t in TRAITS])
    if model_df.empty:
        return {"outcome": outcome, "status": "failed", "reason": "zscore resulted in no rows"}

    summary_rows = []
    for trait in TRAITS:
        key = f"{trait}_z"
        formula = f"{outcome}_z ~ {key}"
        try:
            fit = smf.mixedlm(formula, data=model_df, groups=model_df["participant"]).fit(
                reml=False, method="lbfgs", maxiter=200, disp=False
            )
            conf = fit.conf_int()
            ci_low = float(conf.loc[key, 0]) if key in conf.index else np.nan
            ci_high = float(conf.loc[key, 1]) if key in conf.index else np.nan

            exog = fit.model.exog
            fixed_pred = np.dot(exog, fit.fe_params.values)
            var_fixed = float(np.var(fixed_pred))
            var_random = float(fit.cov_re.iloc[0, 0]) if getattr(fit, "cov_re", None) is not None else 0.0
            var_resid = float(fit.scale)
            denom = var_fixed + var_random + var_resid
            r2_marginal = float(var_fixed / denom) if denom > 0 else np.nan

            summary_rows.append(
                {
                    "predictor": trait,
                    "beta_std": float(fit.fe_params.get(key, np.nan)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "r2_marginal": r2_marginal,
                    "status": "ok",
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "predictor": trait,
                    "beta_std": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "r2_marginal": np.nan,
                    "status": f"failed: {exc}",
                }
            )

    successful = [r for r in summary_rows if r.get("status") == "ok"]
    mean_r2_marginal = float(np.nanmean([r["r2_marginal"] for r in successful])) if successful else np.nan

    return {
        "outcome": outcome,
        "status": "ok",
        "n_rows": int(len(model_df)),
        "n_participants": int(model_df["participant"].nunique()),
        "effects": summary_rows,
        "r2_marginal_mean": mean_r2_marginal,
    }


def _write_markdown(path: Path, results: List[Dict], n_participants: int, n_trials: int) -> None:
    lines = [
        "# AFFEC Statistical Analysis Report",
        "",
        "Mixed-effects models for personality-emotion associations (random intercept by participant).",
        "",
        "## Scope",
        f"- Participants analyzed: {n_participants}",
        f"- Trials analyzed: {n_trials}",
        "- Reporting emphasizes standardized effect sizes and 95% confidence intervals.",
        "- Trial-level naive p-value interpretation is avoided.",
        "",
    ]

    for res in results:
        lines.append(f"## {res['outcome']}")
        if res.get("status") != "ok":
            lines.append(f"- Status: failed ({res.get('reason', 'unknown')})")
            lines.append("")
            continue

        lines.extend(
            [
                f"- Rows: {res['n_rows']}",
                f"- Participants: {res['n_participants']}",
                f"- Mean marginal $R^2$ across one-trait models: {res['r2_marginal_mean']:.4f}",
                "",
                "| Predictor | Std. beta | 95% CI low | 95% CI high | Marginal R² | Status |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in res["effects"]:
            lines.append(
                f"| {row['predictor']} | {row['beta_std']:.4f} | {row['ci95_low']:.4f} | {row['ci95_high']:.4f} | {row['r2_marginal']:.4f} | {row['status']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes for Revision Letter",
            "- This analysis uses mixed-effects models to account for repeated measurements within participants.",
            "- Results should be interpreted by effect size magnitude and confidence intervals.",
            "- This script does not perform domain adaptation or external validation.",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mixed-effects statistical analysis for AFFEC revision")
    parser.add_argument("--max-participants", type=int, default=None, help="Limit participants for a smoke run")
    args = parser.parse_args()

    data_dir = ROOT / "data" / "raw"
    loader = AFFECDataLoader(data_dir=str(data_dir))
    participants = loader.load_participants()
    if participants is None:
        raise FileNotFoundError(f"participants.tsv not found under {data_dir}")

    participant_ids = participants["participant_id"].dropna().astype(str).tolist()
    if args.max_participants is not None:
        participant_ids = participant_ids[: args.max_participants]

    trials_df = _load_trials(loader, participant_ids)
    if trials_df.empty:
        raise RuntimeError("No trials loaded for statistical analysis")

    results = [_fit_mixed_model(trials_df, target) for target in TARGETS]

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    md_path = logs_dir / "statistical_analysis_report.md"
    json_path = logs_dir / "statistical_analysis_report.json"

    _write_markdown(md_path, results, n_participants=len(participant_ids), n_trials=len(trials_df))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_participants": len(participant_ids),
                "n_trials": int(len(trials_df)),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Saved: {md_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
