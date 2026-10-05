import pandas as pd
import pytest
from cyber_exclusion.io import read_csv
from cyber_exclusion.profile import profile_training_data


def test_read_csv_strips_header_and_values(tmp_path):
    f = tmp_path / "train.csv"
    f.write_text("client_id,scan_id,asset ,excluded\n c1 , s1 , 1.2.3.4 ,0\n")
    df = read_csv(f)
    assert list(df.columns) == ["client_id", "scan_id", "asset", "excluded"]
    assert df.loc[0, "asset"] == "1.2.3.4"
    assert df.loc[0, "client_id"] == "c1"


def test_profile_flags_malformed_ip_like():
    df = pd.DataFrame({
        "client_id": ["c1", "c1"],
        "scan_id": ["s1", "s1"],
        "asset": ["1.2.3.4", "173.20174.27"],
        "excluded": [0, 1],
    })
    p = profile_training_data(df)
    assert p["asset_kinds"]["valid_ip"] == 1
    assert p["asset_kinds"]["malformed_ip_like"] == 1


def test_read_csv_accepts_known_validation_client_typo(tmp_path):
    p = tmp_path / "validation.csv"
    p.write_text("cleint_id,scan_id,asset\nc1,s1,example.com\n")
    out = read_csv(p)
    assert list(out.columns) == ["client_id", "scan_id", "asset"]
    assert out.loc[0, "client_id"] == "c1"


def test_s3_resolver_supports_arbitrary_nested_prefixes():
    from cyber_exclusion.io import resolve_s3_asset_uri

    class FakePaginator:
        def paginate(self, **kwargs):
            assert kwargs["Bucket"] == "my-competition-bucket"
            assert kwargs["Prefix"] == "2026/data/raw_scan_data/c1/s1/"
            return [{"Contents": [
                {"Key": "2026/data/raw_scan_data/c1/s1/censys/hosts/raw/1.2.3.4.json"},
                {"Key": "2026/data/raw_scan_data/c1/s1/dns/results/other.json"},
            ]}]

    class FakeS3:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

    row = pd.Series({"client_id": "c1", "scan_id": "s1", "asset": "1.2.3.4"})
    uri = resolve_s3_asset_uri(
        row,
        "s3://my-competition-bucket/2026/data/raw_scan_data",
        s3_client=FakeS3(),
    )
    assert uri == "s3://my-competition-bucket/2026/data/raw_scan_data/c1/s1/censys/hosts/raw/1.2.3.4.json"


def test_s3_resolver_uses_vendor_to_disambiguate_duplicate_filenames():
    from cyber_exclusion.io import resolve_s3_asset_uri

    class FakePaginator:
        def paginate(self, **kwargs):
            return [{"Contents": [
                {"Key": "2026/data/raw_scan_data/c1/s1/censys/1.2.3.4.json"},
                {"Key": "2026/data/raw_scan_data/c1/s1/other_vendor/1.2.3.4.json"},
            ]}]

    class FakeS3:
        def get_paginator(self, name):
            return FakePaginator()

    row = pd.Series({"client_id": "c1", "scan_id": "s1", "asset": "1.2.3.4", "vendor": "censys"})
    uri = resolve_s3_asset_uri(
        row,
        "s3://my-competition-bucket/2026/data/raw_scan_data",
        s3_client=FakeS3(),
    )
    assert uri.endswith("/censys/1.2.3.4.json")
