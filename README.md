# M3plus (Skeleton)

## Install

pip install -r requirements.txt  
Run one (mock, end-to-end)  
python scripts/run_one.py --prompt "A watercolor painting of five pandas sitting on a bench." --backend mock


Outputs:

runs/demo/traces.json

runs/demo/summary.json

Run compare (mock)
python scripts/run_compare.py --input data/demo.json --backend mock


Outputs:

runs/compare/strategy_comparison.json