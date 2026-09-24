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
    oof.to_csv(out / "oof_predictions.csv", index=False)
    with open(out / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    return metrics


def predict(validation_csv: str, model_path: str, cfg_path: str, out_dir: str, raw_root: str | None = None) -> pd.DataFrame:
    cfg = load_cfg(cfg_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(validation_csv)
    features = build_feature_frame(rows, raw_root or cfg.get("raw_root"), str(out / "raw_cache"), cfg.get("aws_region"))
    _save_features(features, out / "validation_features.parquet")
    bundle = joblib.load(model_path)
    pred = bundle.predict(features)
    pred = attach_trust_fields(features, pred)
    pred.to_csv(out / "validation_scored.csv", index=False)
    # Default competition-shaped output; rename label column if competition spec differs.
    pred[["client_id", "scan_id", "asset", "competition_prediction"]].rename(columns={"competition_prediction": "prediction"}).to_csv(out / "submission.csv", index=False)
    return pred
