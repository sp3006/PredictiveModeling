from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import urlparse
import boto3
import pandas as pd
from .utils import safe_json_load


def read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"client_id", "scan_id", "asset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for c in ["client_id", "scan_id", "asset"]:
        df[c] = df[c].astype(str)
    return df


def fetch_s3_json(uri: str, cache_dir: str | Path, region: str | None = None) -> Path:
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"Not an S3 URI: {uri}")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / p.netloc / p.path.lstrip("/")
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        boto3.client("s3", region_name=region).download_file(p.netloc, p.path.lstrip("/"), str(local))
    return local


def locate_raw(row: pd.Series, raw_root: str | Path | None, cache_dir: str | Path, region: str | None = None) -> Path | None:
    if "s3_uri" in row and isinstance(row.get("s3_uri"), str) and row["s3_uri"].startswith("s3://"):
        return fetch_s3_json(row["s3_uri"], cache_dir, region)
    if not raw_root:
        return None
    root = Path(raw_root)
    asset = str(row["asset"])
    client = str(row["client_id"])
    scan = str(row["scan_id"])
    candidates = [
        root / client / scan / f"{asset}.json",
        root / scan / f"{asset}.json",
        root / f"{asset}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(root.rglob(f"{asset}.json"))
    if len(matches) == 1:
        return matches[0]
    return None


def load_raw_for_row(row: pd.Series, raw_root: str | Path | None, cache_dir: str | Path, region: str | None = None) -> tuple[dict, str | None]:
    p = locate_raw(row, raw_root, cache_dir, region)
    if p is None:
        return {}, None
    try:
        return safe_json_load(p), str(p)
    except Exception:
        return {}, str(p)
