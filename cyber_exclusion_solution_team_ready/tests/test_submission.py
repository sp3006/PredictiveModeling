import pandas as pd
import pytest
from cyber_exclusion.pipeline import _build_submission


def test_submission_exact_shape_and_order():
    val = pd.DataFrame({
        "client_id": ["c1", "c2"],
        "scan_id": ["s2", "s1"],
        "asset": ["5.6.7.8", "1.2.3.4"],
    })
    pred = pd.DataFrame({
        "client_id": ["c2", "c1"],
        "scan_id": ["s1", "s2"],
        "asset": ["1.2.3.4", "5.6.7.8"],
        "y_pred": [100.0, 0.5],
    })
    out = _build_submission(val, pred)
    assert list(out.columns) == ["scan_id", "asset", "y_pred"]
    assert out[["scan_id", "asset"]].to_records(index=False).tolist() == val[["scan_id", "asset"]].to_records(index=False).tolist()
    assert out["y_pred"].tolist() == [0.5, 100.0]


def test_submission_rejects_duplicate_validation_keys():
    val = pd.DataFrame({"client_id": ["c1", "c1"], "scan_id": ["s", "s"], "asset": ["a.com", "a.com"]})
    pred = pd.DataFrame({"scan_id": ["s"], "asset": ["a.com"], "y_pred": [0.2]})
    with pytest.raises(ValueError, match="duplicate"):
        _build_submission(val, pred)
