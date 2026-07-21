#!/bin/bash
# Horizon experiment data pipeline (tiered): problems -> plans -> paraphrase
# -> rule verdicts (+oracle cross-check) -> LLM-extraction verdicts (full at
# h10/h20, 12-record stratified subset at h40/h80) -> baselines (all).
# Idempotent: every stage skips if its output file already exists, so the
# pipeline can be resumed after an interruption without repeating API calls.
set -e
PY=.venv/bin/python
DATA=verifier/data/horizon
CONDS=baseline,goal_omission,resource_blind,distractor
mkdir -p $DATA

for d in blocksworld logistics tools; do
  for h in 10 20 40 80; do
    base=$DATA/${d}_h${h}
    if [ ! -f ${base}_problems.jsonl ]; then
      $PY -m scripts.generate_problems --domain $d --n 10 --seed 0 --horizon $h \
        --out ${base}_problems.jsonl
    fi
    if [ ! -f ${base}_plans.jsonl ]; then
      $PY -m scripts.generate_plans --problems ${base}_problems.jsonl \
        --conditions $CONDS --seed 0 --out ${base}_plans.jsonl
    fi
    if [ ! -f ${base}_plans_nl.jsonl ]; then
      $PY -m scripts.paraphrase_plans --plans ${base}_plans.jsonl \
        --out ${base}_plans_nl.jsonl
    fi
    if [ ! -f ${base}_verdicts.jsonl ]; then
      $PY -m scripts.verify_plans --plans ${base}_plans.jsonl \
        --out ${base}_verdicts.jsonl   # rule parser: cross-checks vs oracle, exits 1 on mismatch
    fi
    # tiered extraction: full plan set at h10/h20, stratified 12-record subset at h40/h80
    if [ $h -le 20 ]; then
      ext_in=${base}_plans_nl.jsonl
    else
      ext_in=${base}_plans_nl_sub.jsonl
      if [ ! -f $ext_in ]; then
        $PY - "$base" << 'EOF'
import json, sys
base = sys.argv[1]
recs = [json.loads(l) for l in open(f"{base}_plans_nl.jsonl")]
by_cond = {}
for r in recs:
    by_cond.setdefault(r["condition"], []).append(r)
sub = [r for cond in sorted(by_cond) for r in by_cond[cond][:3]]
with open(f"{base}_plans_nl_sub.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"subset: {len(sub)} records -> {base}_plans_nl_sub.jsonl")
EOF
      fi
    fi
    if [ ! -f ${base}_verdicts_llm.jsonl ]; then
      $PY -m scripts.verify_plans --plans $ext_in \
        --out ${base}_verdicts_llm.jsonl --parser llm --k 3
    fi
    if [ ! -f ${base}_baselines.jsonl ]; then
      $PY -m scripts.run_baselines --plans ${base}_plans_nl.jsonl \
        --out ${base}_baselines.jsonl
    fi
    echo "=== DONE ${d} h${h} ==="
  done
done
echo "=== HORIZON PIPELINE COMPLETE ==="
