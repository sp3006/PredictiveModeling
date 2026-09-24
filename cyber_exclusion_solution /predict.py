import argparse
from cyber_exclusion.pipeline import predict
p=argparse.ArgumentParser()
p.add_argument('--validation', required=True)
p.add_argument('--model', required=True)
p.add_argument('--config', default='config/default.yaml')
p.add_argument('--out', default='artifacts/predict')
p.add_argument('--raw-root', default=None)
a=p.parse_args()
print(predict(a.validation,a.model,a.config,a.out,a.raw_root).head(20).to_string(index=False))
