import argparse, json
from cyber_exclusion.pipeline import train
p=argparse.ArgumentParser()
p.add_argument('--train', required=True)
p.add_argument('--config', default='config/default.yaml')
p.add_argument('--out', default='artifacts/run')
p.add_argument('--raw-root', default=None)
a=p.parse_args()
print(json.dumps(train(a.train,a.config,a.out,a.raw_root), indent=2))
