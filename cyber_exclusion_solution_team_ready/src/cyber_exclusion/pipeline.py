from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import yaml
from .io import read_csv
from .dataset import build_feature_frame
from .model import train_grouped_ensemble
from .trust import attach_trust_fields


def _save_features(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        df.to_csv(path.with_suffix(".csv"), index=False)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(train_csv: str, cfg_path: str, out_dir: str, raw_root: str | None = None) -> dict:
    cfg = load_cfg(cfg_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(train_csv)
    features = build_feature_frame(rows, raw_root or cfg.get("raw_root"), str(out / "raw_cache"), cfg.get("aws_region"))
    _save_features(features, out / "train_features.parquet")
    bundle, oof, metrics = train_grouped_ensemble(features, cfg)
    bundle.save(str(out / "model.joblib"))
    pred_name = "oof_predictions.csv" if metrics.get("grouped_oof_available") else "training_predictions_unvalidated.csv"
    oof.to_csv(out / pred_name, index=False)
    with open(out / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    return metrics



def _validate_validation_rows(validation_rows: pd.DataFrame) -> None:
    key_cols = ["scan_id", "asset"]
    if validation_rows.duplicated(key_cols).any():
        dup = validation_rows.loc[validation_rows.duplicated(key_cols, keep=False), key_cols].head(10)
        raise ValueError(
            "validation_data.csv contains duplicate (scan_id, asset) keys, which conflicts with "
            f"the competition submission contract; examples: {dup.to_dict('records')}"
        )


def _build_submission(validation_rows: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["scan_id", "asset"]
    _validate_validation_rows(validation_rows)
    if pred.duplicated(key_cols).any():
        raise ValueError("Predictions contain duplicate (scan_id, asset) keys.")

    # Merge onto validation order to guarantee exact row identity and ordering.
    expected = validation_rows[key_cols].copy()
    submission = expected.merge(pred[key_cols + ["y_pred"]], on=key_cols, how="left", validate="one_to_one", sort=False)
    if len(submission) != len(validation_rows) or submission["y_pred"].isna().any():
        raise ValueError("Submission keys do not exactly cover validation_data.csv.")
    if set(map(tuple, submission[key_cols].to_numpy())) != set(map(tuple, expected[key_cols].to_numpy())):
        raise ValueError("Submission (scan_id, asset) set differs from validation_data.csv.")
    return submission[["scan_id", "asset", "y_pred"]]

def predict(validation_csv: str, model_path: str, cfg_path: str, out_dir: str, raw_root: str | None = None) -> pd.DataFrame:
    cfg = load_cfg(cfg_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(validation_csv)
    _validate_validation_rows(rows)
    features = build_feature_frame(rows, raw_root or cfg.get("raw_root"), str(out / "raw_cache"), cfg.get("aws_region"))
    _save_features(features, out / "validation_features.parquet")
    bundle = joblib.load(model_path)
    pred = bundle.predict(features)
    pred = attach_trust_fields(features, pred)
    pred.to_csv(out / "validation_scored.csv", index=False)
    submission = _build_submission(rows, pred)
    submission.to_csv(out / "submission.csv", index=False)
    return pred
