# Reproduce the full pipeline from a clean checkout.
#
#   make reproduce         — smoke config: small n, fast, cheap (~15 min, ~$1)
#   make reproduce FULL=1  — full-scale config used for the reported numbers
#   make test              — run the offline test suite
#
# Requires: uv (or an activated venv with the project installed) and
# ANTHROPIC_API_KEY in the environment or in .env at the repo root.

PY := .venv/bin/python
DATA := verifier/data/synthetic
CONDS := baseline,goal_omission,resource_blind,distractor

ifdef FULL
N_PROBLEMS := 100
N_PLANS := 30
else
N_PROBLEMS := 20
N_PLANS := 8
endif

DOMAINS := blocksworld logistics tools

.PHONY: reproduce test problems plans paraphrase verdicts baselines eval

test:
	$(PY) -m pytest -q

problems:
	@for d in $(DOMAINS); do \
		$(PY) -m scripts.generate_problems --domain $$d --n $(N_PROBLEMS) --seed 0 \
			--out $(DATA)/$${d}_problems.jsonl || exit 1; \
	done

plans:
	@for d in $(DOMAINS); do \
		$(PY) -m scripts.generate_plans --problems $(DATA)/$${d}_problems.jsonl \
			--conditions $(CONDS) --n $(N_PLANS) --seed 0 \
			--out $(DATA)/$${d}_plans.jsonl || exit 1; \
	done

paraphrase:
	@for d in $(DOMAINS); do \
		$(PY) -m scripts.paraphrase_plans --plans $(DATA)/$${d}_plans.jsonl \
			--out $(DATA)/$${d}_plans_nl.jsonl || exit 1; \
	done

verdicts:
	@for d in $(DOMAINS); do \
		$(PY) -m scripts.verify_plans --plans $(DATA)/$${d}_plans.jsonl \
			--out $(DATA)/$${d}_verdicts.jsonl || exit 1; \
		$(PY) -m scripts.verify_plans --plans $(DATA)/$${d}_plans_nl.jsonl \
			--out $(DATA)/$${d}_verdicts_llm.jsonl --parser llm --k 3 || exit 1; \
	done

baselines:
	@for d in $(DOMAINS); do \
		$(PY) -m scripts.run_baselines --plans $(DATA)/$${d}_plans_nl.jsonl \
			--out $(DATA)/$${d}_baselines.jsonl || exit 1; \
	done

eval:
	$(PY) -m scripts.run_full_eval --out results/

reproduce: test problems plans paraphrase verdicts baselines eval
	@echo "Done. See results/summary.json and results/figures/."
