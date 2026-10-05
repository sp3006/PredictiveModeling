from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse
import boto3
import pandas as pd
from .utils import safe_json_load

# Process-local cache so every asset in the same scan does not re-list S3.
# Key: (bucket, scan_prefix) -> tuple(object keys)
_S3_SCAN_INDEX: dict[tuple[str, str], tuple[str, ...]] = {}


def read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Competition files occasionally contain incidental whitespace in headers
    # (the provided training file has `asset `). Normalize defensively.
    normalized = [str(c).strip() for c in df.columns]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Column names collide after whitespace normalization: {normalized}")
    df.columns = normalized

    # Accept the known validation typo while normalizing internally.
    if "client_id" not in df.columns and "cleint_id" in df.columns:
        df = df.rename(columns={"cleint_id": "client_id"})

    required = {"client_id", "scan_id", "asset"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for c in ["client_id", "scan_id", "asset"]:
        df[c] = df[c].astype(str).str.strip()
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


def _parse_s3_root(raw_root: str) -> tuple[str, str]:
    p = urlparse(raw_root)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"Expected S3 root like s3://bucket/prefix, got: {raw_root}")
    return p.netloc, p.path.lstrip("/").rstrip("/")


def _list_scan_keys(bucket: str, scan_prefix: str, region: str | None = None, s3_client=None) -> tuple[str, ...]:
    cache_key = (bucket, scan_prefix)
    if s3_client is None and cache_key in _S3_SCAN_INDEX:
        return _S3_SCAN_INDEX[cache_key]

    client = s3_client or boto3.client("s3", region_name=region)
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=scan_prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(str(key))
    result = tuple(keys)
    if s3_client is None:
        _S3_SCAN_INDEX[cache_key] = result
    return result


def resolve_s3_asset_uri(
    row: pd.Series,
    raw_root: str,
    region: str | None = None,
    s3_client=None,
) -> str | None:
    """Resolve one competition asset to raw S3 JSON under an arbitrarily deep scan subtree.

    Expected layout:
      s3://<bucket>/<root>/<client_id>/<scan_id>/<zero-or-more-prefix-elements>/<filename>

    Resolution is intentionally exact on the asset basename. We do not silently repair
    malformed asset identifiers because (scan_id, asset) is part of the competition key.
    If multiple exact filenames exist, an optional `vendor` column is used only as a
    disambiguator; otherwise the ambiguity is surfaced instead of guessed.
    """
    bucket, root_prefix = _parse_s3_root(raw_root)
    client_id = str(row["client_id"]).strip()
    scan_id = str(row["scan_id"]).strip()
    asset = str(row["asset"]).strip()
    scan_prefix = "/".join(x for x in [root_prefix, client_id, scan_id] if x) + "/"
    keys = _list_scan_keys(bucket, scan_prefix, region=region, s3_client=s3_client)

    target_names = {asset, f"{asset}.json"}
    matches = [k for k in keys if Path(k).name in target_names]
    if not matches:
        return None
    if len(matches) == 1:
        return f"s3://{bucket}/{matches[0]}"

    vendor = str(row.get("vendor", "") or "").strip().lower()
    if vendor:
        vendor_matches = [
            k for k in matches
            if vendor in [part.lower() for part in Path(k).parts]
        ]
        if len(vendor_matches) == 1:
            return f"s3://{bucket}/{vendor_matches[0]}"

    raise ValueError(
        "Multiple raw S3 objects matched the same competition asset. "
        f"client_id={client_id} scan_id={scan_id} asset={asset} matches={matches[:10]}. "
        "Add a vendor/source column or an explicit s3_uri rather than choosing arbitrarily."
    )


def locate_raw(row: pd.Series, raw_root: str | Path | None, cache_dir: str | Path, region: str | None = None) -> Path | None:
    # Highest-precedence option: exact object URI supplied in the manifest.
    if "s3_uri" in row and isinstance(row.get("s3_uri"), str) and row["s3_uri"].startswith("s3://"):
        return fetch_s3_json(row["s3_uri"], cache_dir, region)

    if not raw_root:
        return None

    raw_root_str = str(raw_root)
    if raw_root_str.startswith("s3://"):
        uri = resolve_s3_asset_uri(row, raw_root_str, region=region)
        return fetch_s3_json(uri, cache_dir, region) if uri else None

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

    # Local development fallback supports arbitrary nesting below the scan directory.
    scan_root = root / client / scan
    if scan_root.exists():
        matches = list(scan_root.rglob(f"{asset}.json")) + list(scan_root.rglob(asset))
        matches = list(dict.fromkeys(matches))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple local raw files matched client_id={client} scan_id={scan} asset={asset}: {matches[:10]}"
            )
    return None


def load_raw_for_row(row: pd.Series, raw_root: str | Path | None, cache_dir: str | Path, region: str | None = None) -> tuple[dict, str | None]:
    p = locate_raw(row, raw_root, cache_dir, region)
    if p is None:
        return {}, None
    try:
        return safe_json_load(p), str(p)
    except Exception:
        return {}, str(p)
