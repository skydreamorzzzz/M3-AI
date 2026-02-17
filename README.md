# M3plus (Structured Constraint + Scheduling Framework)

M3plus is a structured visual refinement framework:

1. Generate initial image
2. Extract structured constraints from prompt
3. Build constraint graph (DAG / conflict-aware)
4. Judge constraints via VLM
5. Refine in multi-round loop with scheduling strategies

Supports:
- Mock mode (no API calls)
- Real API mode (DeepSeek + DashScope)
- Linear / Topo / Conflict-aware schedulers

---

# 1. Environment Setup

## Python
Recommended: Python 3.10+

Create virtual environment:
python -m venv .venv
.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

---

# 2. API Configuration (Real Mode Only)
This project uses:
- DeepSeek → text LLM (constraint extraction)
- DashScope (Qwen / Wanx) → vision judge + image generation

## Set Environment Variables (Windows PowerShell)
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-xxxx", "User")
[Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-xxxx", "User")

Restart terminal after setting.

Verify:
echo $env:DEEPSEEK_API_KEY
echo $env:DASHSCOPE_API_KEY

---

# 3. Running Modes
| Mode    | Backend | Image Gen | Judge | Cost  |
|---------|---------|-----------|-------|-------|
| mock    | mock    | ❌ fake   | ❌ fake| Free  |
| dry_run | openai  | ✅ real   | ✅ real| Medium|
| full    | openai  | ✅ real   | ✅ real + edit | High |

---

# 4. Quick Test (Mock End-to-End)
No API calls.

python scripts/run_experiment.py --input data/prompts.json --backend mock --strategies linear --max_rounds 2

Outputs:
runs/<exp_id>/
  ├── summary.csv
  ├── trace_long.csv
  └── _precomputed/

---

# 5. Real API (Safe Version – No Editing)
This runs:
- DeepSeek → extract constraints
- Wanx → generate initial image
- Qwen-VL → judge
- Editor = dry_run (no image edit cost)

python scripts/run_experiment.py --input data/prompts.json --backend openai --dry_run 1 --strategies linear --max_rounds 2

Recommended before running full edit loop.

---

# 6. Full Real Pipeline (Includes Editing)
⚠️ This consumes image edit credits.

python scripts/run_experiment.py --input data/prompts.json --backend openai --dry_run 0 --strategies topo_conflict --max_rounds 4

---

# 7. Strategy Comparison (Multiple Schedulers)
python scripts/run_experiment.py --input data/prompts.json --backend openai --dry_run 1 --strategies linear,topo_conflict --max_rounds 3

Compare:
- total rounds
- final pass
- conflict reduction
- scheduling efficiency

---

# 8. Project Structure
src/
  llm/
  refine/
  scheduler/
  graph/
  config/

scripts/
  run_experiment.py
  run_one.py
  run_compare.py

data/
  prompts.json

runs/
  exp_xxx/

---

# 9. Debugging Tips
If all prompts finish with:
- rounds = 0
- final_pass = True

Possible causes:
- Vision input not processed
- Judge prompt too loose
- Constraint extraction missing structure
- Image not passed correctly to VLM

Enable judge printing in:
src/llm/prompts/judge_constraint.py

Check:
- raw_response
- parsed_json
- constraint_id alignment

---

# 10. Research Notes
This framework implements:
- Structured constraint extraction
- Graph-based dependency scheduling
- Conflict-aware refinement
- LLM-based visual constraint verification

Goal:
Improve refinement stability and reduce oscillation in complex prompts.