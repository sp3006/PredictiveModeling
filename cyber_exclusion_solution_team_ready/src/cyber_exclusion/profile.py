from __future__ import annotations
import ipaddress
import re
from typing import Any
import pandas as pd


def _asset_kind(asset: str) -> str:
    s = str(asset).strip()
    try:
        ipaddress.ip_address(s)
        return "valid_ip"
    except ValueError:
        pass
    domain_re = re.compile(r"(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}")
    if domain_re.fullmatch(s):
        return "domain"
    if re.fullmatch(r"[0-9.]+", s):
        return "malformed_ip_like"
    return "other"


def profile_training_data(df: pd.DataFrame, label_col: str = "excluded") -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(df)),
        "clients": int(df["client_id"].nunique()),
        "scans": int(df["scan_id"].nunique()),
        "assets": int(df["asset"].nunique()),
        "duplicate_scan_asset": int(df.duplicated(["scan_id", "asset"]).sum()),
    }
    kinds = df["asset"].map(_asset_kind).value_counts().to_dict()
    out["asset_kinds"] = {str(k): int(v) for k, v in kinds.items()}
    if label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        out["positive_rows"] = int((y > 0).sum())
        out["negative_rows"] = int((y <= 0).sum())
        out["positive_rate"] = float((y > 0).mean())
        out["positive_clients"] = int(df.loc[y > 0, "client_id"].nunique())
        out["positive_scans"] = int(df.loc[y > 0, "scan_id"].nunique())
    return out
