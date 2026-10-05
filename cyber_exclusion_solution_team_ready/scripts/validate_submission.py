#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cyber_exclusion.io import read_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Validate the competition submission contract.")
    p.add_argument("--validation", required=True)
    p.add_argument("--submission", required=True)
    a = p.parse_args()

    validation = read_csv(a.validation)
    submission = pd.read_csv(a.submission)
    submission.columns = [str(c).strip() for c in submission.columns]

    required = ["scan_id", "asset", "y_pred"]
    if list(submission.columns) != required:
        raise SystemExit(f"FAIL: columns must be exactly {required}; got {list(submission.columns)}")

    if validation.duplicated(["scan_id", "asset"]).any():
        raise SystemExit("FAIL: validation contains duplicate (scan_id, asset) keys.")
    if submission.duplicated(["scan_id", "asset"]).any():
        raise SystemExit("FAIL: submission contains duplicate (scan_id, asset) keys.")
    if len(submission) != len(validation):
        raise SystemExit(f"FAIL: row count {len(submission)} != validation row count {len(validation)}")

    expected = validation[["scan_id", "asset"]].reset_index(drop=True).astype(str)
    actual = submission[["scan_id", "asset"]].reset_index(drop=True).astype(str)
    if not expected.equals(actual):
        raise SystemExit("FAIL: submission scan_id/asset keys or row order differ from validation.")

    scores = pd.to_numeric(submission["y_pred"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise SystemExit("FAIL: y_pred contains missing/non-numeric/infinite values.")

    print(f"PASS: {len(submission)} rows; unique keys; exact validation order; finite y_pred values.")


if __name__ == "__main__":
    main()
