# EC2 + JupyterLab Implementation Runbook

This runbook is designed for a teammate starting from a fresh EC2 instance and ending with a competition-ready `submission.csv`.

## A. AWS prerequisites

Before launching code, prepare:

- EC2 instance with network access to S3 and Amazon Bedrock (if Bedrock demo is required).
- IAM role / instance profile attached to EC2.
- Read access to:
  `s3://my-competition-bucket/2026/data/raw_scan_data/`
- Optional Bedrock model invocation permission.
- Git access to `sp3006/PredictiveModeling`.

The application uses the normal AWS SDK credential chain. On EC2, Boto3 automatically uses temporary credentials from the instance role, so no static keys are required.

## B. Suggested EC2 sizing

For the current small competition sample, a general-purpose CPU instance is sufficient. If the full raw dataset/model becomes much larger, increase RAM/CPU before changing the architecture. CatBoost and TF-IDF are CPU-friendly and do not require a GPU for this baseline.

Keep the root/EBS volume large enough for:

- Python environment;
- S3 raw-object cache for the scans touched by the manifests;
- feature matrices;
- model artifacts.

## C. Bootstrap the instance

```bash
git clone https://github.com/sp3006/PredictiveModeling.git
cd PredictiveModeling
git checkout feature   # only while feature is the implementation branch
bash scripts/bootstrap_ec2.sh
source .venv/bin/activate
```

Verify Python:

```bash
python --version
python -c "import boto3, pandas, sklearn, catboost; print('python dependencies OK')"
```

## D. Verify AWS identity

With AWS CLI:

```bash
aws sts get-caller-identity
```

Or Python/Boto3:

```python
import boto3
print(boto3.client("sts").get_caller_identity()["Arn"])
```

If this fails, fix the EC2 instance profile before configuring credentials manually.

## E. Verify S3 access

```python
import boto3

s3 = boto3.client("s3", region_name="us-east-1")
resp = s3.list_objects_v2(
    Bucket="my-competition-bucket",
    Prefix="2026/data/raw_scan_data/",
    MaxKeys=5,
)
for obj in resp.get("Contents", []):
    print(obj["Key"])
```

The actual pipeline narrows each lookup to:

```text
2026/data/raw_scan_data/<client_id>/<scan_id>/
```

and recursively matches the asset basename under any intermediate vendor/type folders.

## F. Configure the project

Review `config/default.yaml`:

```yaml
raw_root: s3://my-competition-bucket/2026/data/raw_scan_data
aws_region: us-east-1
competition_metric: average_precision
```

For an optional Bedrock demo, either configure a model ID in YAML or export it only for the current shell/notebook:

```bash
export BEDROCK_MODEL_ID='<approved-model-or-inference-profile-id>'
```

Do not commit secrets or access keys.

## G. Copy the competition CSV files onto EC2

Recommended:

```text
data/training-data.csv
data/validation-data.csv
```

Quick checks:

```bash
python - <<'PY'
from cyber_exclusion.io import read_csv
from cyber_exclusion.profile import profile_training_data

tr = read_csv('data/training-data.csv')
va = read_csv('data/validation-data.csv')
print('training', tr.shape)
print(profile_training_data(tr))
print('validation', va.shape)
print('validation duplicate keys', va.duplicated(['scan_id','asset']).sum())
PY
```

## H. Train

```bash
python train.py \
  --train data/training-data.csv \
  --config config/default.yaml \
  --out artifacts/run_001
```

Inspect:

```bash
cat artifacts/run_001/metrics.json
ls -lh artifacts/run_001/
```

If `grouped_oof_available` is false, treat training metrics as diagnostic only. This is expected for the currently supplied sample because positives do not span enough independent clients.

## I. Score validation + create submission

```bash
python predict.py \
  --validation data/validation-data.csv \
  --model artifacts/run_001/model.joblib \
  --config config/default.yaml \
  --out artifacts/submission_001
```

Inspect the competition file:

```bash
head -20 artifacts/submission_001/submission.csv
```

It must contain only:

```text
scan_id,asset,y_pred
```

## J. Validate before upload

```bash
python scripts/validate_submission.py \
  --validation data/validation-data.csv \
  --submission artifacts/submission_001/submission.csv
```

This verifies:

- columns are exactly `scan_id,asset,y_pred`;
- `(scan_id, asset)` is unique;
- all validation rows are represented exactly once;
- row order matches validation;
- `y_pred` is numeric and finite.

## K. JupyterLab

Start Jupyter on loopback only:

```bash
source .venv/bin/activate
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Tunnel from your workstation:

```bash
ssh -i /path/to/key.pem \
  -L 8888:127.0.0.1:8888 \
  <ec2-user-or-ubuntu>@<EC2_PUBLIC_DNS>
```

Open `http://127.0.0.1:8888` and use:

`notebooks/02_ec2_s3_bedrock_submission.ipynb`

The notebook calls the same package code as the CLI, so notebook and terminal runs remain consistent.

## L. Optional Bedrock / Claude demonstration

The code intentionally uses Claude as a grounded audit narrator, not as the ranking model.

Connectivity test:

```python
import os
from cyber_exclusion.bedrock_explainer import explain_with_bedrock

model_id = os.environ['BEDROCK_MODEL_ID']
text = explain_with_bedrock(
    model_id=model_id,
    region='us-east-1',
    evidence={
        'asset': 'example.com',
        'y_pred': 0.82,
        'evidence_coverage': 0.7,
        'reason_codes': ['EXAMPLE_ONLY'],
        'note': 'Use actual structured row evidence in the real demo.'
    },
)
print(text)
```

The IAM role requires `bedrock:InvokeModel` for Converse. If your organization restricts Bedrock models, use an approved model/inference profile and corresponding resource policy.

## M. Optionally archive outputs to S3

If your IAM role also has write permission to an approved output prefix:

```bash
aws s3 cp artifacts/submission_001/submission.csv \
  s3://my-competition-bucket/2026/data/submissions/submission.csv
```

Do not assume the competition raw-data prefix is writable. Use a team-approved output location.

## N. Recommended team demo script

1. Open training and validation CSVs.
2. Show one asset mapping to its nested S3 raw JSON.
3. Run/inspect feature extraction.
4. Train or load the model.
5. Show `validation_scored.csv`.
6. Invoke Claude for one selected asset explanation.
7. Run the standalone submission validator.
8. Open `submission.csv` and upload it to the competition portal.

## O. Productionization after the competition

Add an append-only reviewed-decision store containing the model version, evidence, CRS outcome, override status, and timestamps. Build historical features only from records whose decision timestamp is earlier than the scan being scored. That preserves auditability and prevents future-label leakage.
