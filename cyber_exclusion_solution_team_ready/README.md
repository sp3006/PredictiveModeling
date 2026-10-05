# Cyber Scan Asset Exclusion Predictor

Predict which automatically discovered domains/IPs are likely to be excluded by CRS **before manual review**, while preserving an auditable path for human review and future decision memory.

This repository has two goals:

1. **Competition:** rank every validation asset using `y_pred` for the organizer's **Average Precision** metric.
2. **Production / CRS trust:** keep prediction separate from operational trust by retaining evidence, confidence, review status, and optional Claude/Bedrock explanations.

> Start here for a fresh AWS machine: **[`docs/EC2_JUPYTER_RUNBOOK.md`](docs/EC2_JUPYTER_RUNBOOK.md)**
>
> Jupyter walkthrough: **`notebooks/02_ec2_s3_bedrock_submission.ipynb`**

---

## 1. Architecture at a glance

```text
training-data.csv / validation-data.csv
                 |
                 v
        client_id + scan_id + asset
                 |
                 v
        boto3 S3 raw-data resolver
                 |
                 v
s3://my-competition-bucket/2026/data/raw_scan_data/
  <client_id>/<scan_id>/<arbitrary nested vendor/type path>/<asset>.json
                 |
                 v
       Structured feature extraction
  DNS | ASN/WHOIS | TLS | services | provider hints
       + scan-relative consistency/outlier features
                 |
          +------+------+
          |             |
          v             v
       CatBoost    char-TFIDF Logistic
          |             |
          +------v------+
                 |
       ranking-oriented ensemble
                 |
       +---------+---------+
       |                   |
       v                   v
competition y_pred    trust diagnostics
(Average Precision)   p_exclude / evidence /
                      KEEP-EXCLUDE-REVIEW
                              |
                              v
                    optional Claude via
                    Amazon Bedrock Converse
                    (explanation only)
```

**Important:** Claude does not generate or overwrite `y_pred`. The competition score comes from the ML pipeline. Bedrock is an optional, grounded explanation/audit layer.

---

## 2. Repository layout

```text
.
├── config/default.yaml                 # competition/S3/model configuration
├── docs/EC2_JUPYTER_RUNBOOK.md         # detailed AWS + Jupyter instructions
├── examples/iam-policy-example.json    # least-privilege starting policy
├── notebooks/
│   ├── 01_eda_and_baseline.ipynb
│   └── 02_ec2_s3_bedrock_submission.ipynb
├── scripts/
│   ├── bootstrap_ec2.sh                # fresh-EC2 bootstrap helper
│   └── validate_submission.py          # final upload guardrail
├── src/cyber_exclusion/
│   ├── io.py                           # CSV + nested S3 resolver
│   ├── features.py                     # cyber feature extraction
│   ├── model.py                        # ranking ensemble / trust model
│   ├── pipeline.py                     # train + predict + submission
│   ├── bedrock_explainer.py            # Claude/Bedrock audit narrator
│   ├── profile.py                      # dataset profiling
│   └── trust.py                        # evidence/trust diagnostics
├── train.py
├── predict.py
└── tests/
```

---

## 3. Competition inputs

### Training CSV

Required logical columns:

```text
client_id,scan_id,asset,excluded
```

The loader also normalizes incidental header whitespace such as `asset `.

### Validation CSV

Required columns:

```text
client_id,scan_id,asset
```

The submission contract is exactly:

```text
scan_id,asset,y_pred
```

`y_pred` is a continuous ranking score. It does **not** need to be a probability between 0 and 1 because Average Precision depends on row ordering.

The pipeline rejects duplicate `(scan_id, asset)` keys and guarantees the submission keys/row order match the validation set.

---

## 4. Competition S3 layout

Configured raw-data root:

```text
s3://my-competition-bucket/2026/data/raw_scan_data/
```

Object keys may contain any number of path components after `scan_id`:

```text
2026/data/raw_scan_data/
  <client_id>/
    <scan_id>/
      <vendor-or-data-type>/
        <optional-more-folders>/
          <asset>.json
```

The resolver:

- lists only the current `<client_id>/<scan_id>/` subtree;
- matches the exact asset filename (`asset` or `asset.json`);
- caches each scan listing during the process;
- optionally uses a `vendor` manifest column to disambiguate duplicate filenames;
- refuses to guess if multiple objects still match.

Default configuration:

```yaml
raw_root: s3://my-competition-bucket/2026/data/raw_scan_data
aws_region: us-east-1
competition_metric: average_precision
```

---

## 5. Fastest EC2 setup

Use an **EC2 instance role / instance profile** rather than static access keys.

On a fresh Amazon Linux 2023 or Ubuntu instance:

```bash
# Clone the project
git clone https://github.com/sp3006/PredictiveModeling.git
cd PredictiveModeling

# If implementation is currently on the feature branch
git checkout feature

# Bootstrap Python/Jupyter dependencies
bash scripts/bootstrap_ec2.sh
source .venv/bin/activate
```

Sanity-check AWS identity and S3 access:

```bash
aws sts get-caller-identity
aws s3 ls s3://my-competition-bucket/2026/data/raw_scan_data/ --recursive | head
```

If the AWS CLI is not installed, the Jupyter runbook contains a Boto3-only connectivity check.

---

## 6. Put the CSV manifests on EC2

Recommended local working layout:

```text
PredictiveModeling/
  data/
    training-data.csv
    validation-data.csv
```

For example:

```bash
mkdir -p data
# copy/download the two competition CSV files into data/
```

The raw JSON does **not** need to be copied to EC2. Boto3 resolves and caches required objects from S3 as the pipeline processes each row.

---

## 7. Train from the EC2 terminal

```bash
source .venv/bin/activate

python train.py \
  --train data/training-data.csv \
  --config config/default.yaml \
  --out artifacts/run_001
```

Expected outputs:

```text
artifacts/run_001/
├── model.joblib
├── metrics.json
├── train_features.parquet        # or CSV fallback
├── raw_cache/                    # downloaded S3 JSON used by this run
└── oof_predictions.csv           # when honest grouped OOF is possible
    # OR training_predictions_unvalidated.csv for insufficient positive groups
```

### Current small-data warning

The currently supplied training sample has positives from only one independent client. The code therefore allows a final ranking model to be fit, but does **not** present grouped OOF Average Precision or calibration as validated when the data cannot support that claim.

---

## 8. Score validation and build the competition upload

```bash
python predict.py \
  --validation data/validation-data.csv \
  --model artifacts/run_001/model.joblib \
  --config config/default.yaml \
  --out artifacts/submission_001
```

Outputs:

```text
artifacts/submission_001/
├── validation_features.parquet   # or CSV fallback
├── validation_scored.csv         # diagnostic/trust output
├── raw_cache/
└── submission.csv                # THIS is the competition upload
```

Validate it one final time:

```bash
python scripts/validate_submission.py \
  --validation data/validation-data.csv \
  --submission artifacts/submission_001/submission.csv
```

Expected success message:

```text
PASS: submission matches validation keys/order and contains finite y_pred values.
```

Then upload:

```text
artifacts/submission_001/submission.csv
```

Its columns must be exactly:

```text
scan_id,asset,y_pred
```

---

## 9. JupyterLab workflow

Start Jupyter on the EC2 instance:

```bash
source .venv/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

From your laptop, use an SSH tunnel rather than exposing Jupyter directly to the internet:

```bash
ssh -i /path/to/key.pem -L 8888:127.0.0.1:8888 <ec2-user-or-ubuntu>@<EC2_PUBLIC_DNS>
```

Then open `http://127.0.0.1:8888` locally and run:

```text
notebooks/02_ec2_s3_bedrock_submission.ipynb
```

That notebook walks through:

1. AWS identity verification;
2. S3 raw-prefix connectivity;
3. training/validation profiling;
4. model training;
5. validation scoring;
6. exact submission validation;
7. optional Claude/Bedrock explanation for a selected scored asset.

See the detailed runbook for troubleshooting: [`docs/EC2_JUPYTER_RUNBOOK.md`](docs/EC2_JUPYTER_RUNBOOK.md).

---

## 10. Bedrock / Claude integration

Bedrock is optional for the competition score. It is useful for demonstrating how CRS could inspect a model recommendation with grounded evidence.

The code uses:

```python
boto3.client("bedrock-runtime", region_name=region).converse(...)
```

Enable it only after your EC2 role can invoke the selected model:

```yaml
bedrock:
  enabled: true
  model_id: <approved Bedrock model or inference-profile ID>
  region: us-east-1
```

The EC2 role needs `bedrock:InvokeModel` for Converse. The notebook also supports setting the model ID via `BEDROCK_MODEL_ID` so a model identifier is not committed to source control.

**Recommended demo:** pick one row from `validation_scored.csv`, pass only its structured model/evidence fields to Claude, and show that Claude explains the evidence and missing evidence without changing the underlying `y_pred`.

---

## 11. IAM permissions

Use [`examples/iam-policy-example.json`](examples/iam-policy-example.json) as a starting point. At minimum the pipeline needs:

- `s3:ListBucket` on `my-competition-bucket`, preferably limited to `2026/data/raw_scan_data/*`;
- `s3:GetObject` on `arn:aws:s3:::my-competition-bucket/2026/data/raw_scan_data/*`;
- `bedrock:InvokeModel` only if the optional explanation workflow is enabled.

For EC2, attach these permissions through an IAM role/instance profile. Do not put long-lived AWS access keys in notebooks, `.env` files committed to Git, or source code.

---

## 12. What to show in a competition demo

A clean 5-minute walkthrough is:

1. **Data contract:** show `training-data.csv` and `validation-data.csv`.
2. **S3:** show the raw scan hierarchy and Boto3 resolving a row to its JSON evidence.
3. **Model:** show extracted features and `metrics.json` / small-data warnings.
4. **Trust:** show `validation_scored.csv` with `y_pred`, `p_exclude`, evidence coverage, and `REVIEW` where confidence is not production-validated.
5. **Bedrock:** explain one selected asset using the structured evidence package.
6. **Submission:** run `scripts/validate_submission.py`, then show the three-column `submission.csv` ready for upload.

This demonstrates both the competition solution and the production direction without claiming more statistical certainty than the available training labels support.

---

## 13. Persistent Cyber Decision Memory (next production layer)

The production architecture should append every reviewed outcome to a durable store (S3/Parquet, DynamoDB, or a governed feature store) with fields such as:

```text
client_id, scan_id, asset, observed_at,
model_version, y_pred, p_exclude,
CRS_final_decision, analyst_override,
reason_codes, evidence_summary,
unresolved_risks, evidence_coverage
```

Future scans can then derive **time-safe** historical features only from decisions that occurred before the new scan. This implements the persistent-evidence idea from the project's CC-SLT/Sentinel research while preventing future-label leakage.

---

## 14. Troubleshooting quick reference

**`AccessDenied` listing S3**  
Check EC2 instance profile and `s3:ListBucket` permission on the bucket/prefix.

**`AccessDenied` downloading JSON**  
Check `s3:GetObject` on the raw object prefix and any KMS decrypt requirement if objects use a customer-managed KMS key.

**Raw data missing for many rows**  
Inspect the scan subtree. If an asset filename appears more than once, add `vendor` or an exact `s3_uri` to the manifest rather than guessing.

**Bedrock `AccessDeniedException`**  
Check the EC2 role's `bedrock:InvokeModel` permission, selected Region, and that the chosen model/inference profile is available to the account.

**Training says grouped OOF unavailable**  
This is expected when positive labels occur in too few independent client groups. Do not treat in-sample metrics as expected leaderboard or production performance.

**Submission rejected locally**  
Do not manually drop/reorder validation rows. Fix the validation data contract, regenerate predictions, and re-run `scripts/validate_submission.py`.

---

## 15. Run tests before handoff

```bash
source .venv/bin/activate
pytest -q
```

The test suite covers feature extraction, input normalization, S3 resolution, small-data behavior, and submission contract enforcement.
