from __future__ import annotations
import hashlib
import ipaddress
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

COMMON_MULTI_SUFFIXES = {
    "co.uk", "com.au", "co.jp", "com.br", "co.in", "com.sg", "com.mx", "co.nz", "com.cn"
}


def stable_hash(value: Any, n: int = 16) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="ignore")).hexdigest()[:n]


def is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(str(value).strip("[]"))
        return True
    except Exception:
        return False


def registrable_domain(name: str | None) -> str:
    if not name:
        return ""
    s = str(name).strip().lower().rstrip(".")
    if is_ip(s):
        return s
    try:
        import tldextract  # type: ignore
        ext = tldextract.TLDExtract(suffix_list_urls=None)(s)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return ext.domain or s
    except Exception:
        parts = [p for p in s.split(".") if p]
        if len(parts) <= 2:
            return s
        suffix2 = ".".join(parts[-2:])
        if suffix2 in COMMON_MULTI_SUFFIXES and len(parts) >= 3:
            return ".".join(parts[-3:])
        return suffix2


def flatten_scalars(obj: Any, prefix: str = "", max_items: int = 2500) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    stack = [(prefix, obj)]
    while stack and len(out) < max_items:
        key, val = stack.pop()
        if isinstance(val, dict):
            for k, v in val.items():
                stack.append((f"{key}.{k}" if key else str(k), v))
        elif isinstance(val, list):
            for i, v in enumerate(val[:100]):
                stack.append((f"{key}[{i}]", v))
        elif val is None or isinstance(val, (str, int, float, bool)):
            out.append((key, val))
    return out


def strings_at_keys(obj: Any, key_terms: Iterable[str], max_values: int = 500) -> list[str]:
    terms = tuple(t.lower() for t in key_terms)
    values: list[str] = []
    for k, v in flatten_scalars(obj):
        lk = k.lower()
        if any(t in lk for t in terms) and isinstance(v, str) and v.strip():
            values.append(v.strip())
            if len(values) >= max_values:
                break
    return values


def numeric_count_at_keys(obj: Any, key_terms: Iterable[str]) -> int:
    terms = tuple(t.lower() for t in key_terms)
    n = 0
    for k, v in flatten_scalars(obj):
        lk = k.lower()
        if any(t in lk for t in terms):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                n += int(v > 0)
            elif isinstance(v, str) and v.strip():
                n += 1
    return n


def safe_json_load(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"payload": data}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def tokenize_asset(s: str) -> list[str]:
    return [x for x in re.split(r"[^a-z0-9]+", str(s).lower()) if x]
