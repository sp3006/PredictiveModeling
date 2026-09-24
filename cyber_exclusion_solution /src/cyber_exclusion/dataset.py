from __future__ import annotations
from pathlib import Path
import pandas as pd
from .features import extract_asset_features, add_scan_relative_features, add_seed_features
from .io import load_raw_for_row


def build_feature_frame(rows: pd.DataFrame, raw_root: str | None = None, cache_dir: str = ".cache/raw", region: str | None = None) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        raw, raw_path = load_raw_for_row(row, raw_root, cache_dir, region)
        f = extract_asset_features(str(row["asset"]), raw, str(row.get("vendor", "censys")))
        rec = row.to_dict()
        rec.update(f)
        rec["raw_path"] = raw_path or ""
        records.append(rec)
    out = pd.DataFrame(records)
    out = add_scan_relative_features(out)
    out = add_seed_features(out)
    return out
