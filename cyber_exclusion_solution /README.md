# Cyber Scan Asset Exclusion Predictor

Production-oriented baseline for predicting which discovered domains/IPs CRS will exclude from a client scan.

## Design

The system separates **prediction** from **trust**:

1. Structured cyber evidence is extracted from Censys-style JSON (DNS, WHOIS/ASN, services/ports, TLS/certificates, vulnerability hints, provider/shared-host signals).
2. Scan-relative features estimate whether an asset agrees with the scan's dominant infrastructure identity or behaves like an outlier/shared service.
3. A CatBoost tabular model is blended with a character-TFIDF logistic model over asset/evidence text.
4. Out-of-fold probabilities are calibrated with grouped CV by `client_id` to reduce leakage.
5. Production decisions are selective: `EXCLUDE`, `KEEP`, or `REVIEW`, with conformal prediction sets and evidence coverage.
6. Optional Bedrock usage is explanation-only by default. The LLM receives structured evidence and cannot overwrite the predictive score.

## Expected inputs

`training_data.csv` must contain at least:

- `client_id`
- `scan_id`
- `asset`
- one label column among `excluded`, `is_excluded`, `label`, `target`, `y`

Useful optional columns:

- `vendor`
- `s3_uri` (direct S3 URI to the raw JSON for that asset)
- `seed_domain` (strongly recommended if available)

`validation_data.csv` must contain `client_id`, `scan_id`, `asset` and may also contain the optional columns above.

If `s3_uri` is absent, pass `--raw-root` and organize JSON as one of:

- `<raw_root>/<client_id>/<scan_id>/<asset>.json`
- `<raw_root>/<scan_id>/<asset>.json`
- `<raw_root>/<asset>.json`

## EC2/Jupyter setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
export PYTHONPATH=$PWD/src
```

## Train

```bash
python train.py \
  --train /data/training_data.csv \
  --raw-root /data/raw \
  --config config/default.yaml \
  --out artifacts/run_001
```

Artifacts include `model.joblib`, `oof_predictions.csv`, `metrics.json`, and materialized features.

## Predict / submission

```bash
python predict.py \
  --validation /data/validation_data.csv \
  --model artifacts/run_001/model.joblib \
  --raw-root /data/raw \
  --out artifacts/submission_001
```

Outputs include `validation_scored.csv` (probability, conformal set, 3-way decision, evidence coverage) and `submission.csv`.

## Competition strategy

Before final submission, replace the default F1 threshold if the official competition metric differs. Do not tune on the holdout. Prefer nested/grouped CV by client and, where timestamps exist, a temporal backtest. If a client has repeated scans, historical features must use **only prior scans** to avoid target leakage.

## High-value feature upgrades once schema is inspected

- Exact discovery-path / parent-asset features.
- Certificate SAN overlap with seed and confirmed client domains.
- Certificate reuse cardinality across clients (shared infrastructure signal).
- ASN / WHOIS / rDNS agreement and disagreement features.
- DNS CNAME/MX/NS/SPF provider ownership and third-party fingerprints.
- Passive-DNS history and asset age if available.
- Shared-host/CDN/WAF labels and co-tenancy degree.
- Graph features: independent evidence paths from seed, node degree/hubness, community membership.
- Client-history state based only on chronologically prior analyst-reviewed scans.

## Trust policy

For production, do not equate a high model probability with high operational trust. Auto-action should require:

- calibrated probability beyond a business-selected threshold,
- singleton conformal set,
- minimum evidence coverage,
- no hard conflict rule (for example strong seed-domain control evidence conflicting with a shared-host heuristic),
- stable calibration and drift diagnostics on recent analyst-reviewed outcomes.

Everything else goes to CRS review.
