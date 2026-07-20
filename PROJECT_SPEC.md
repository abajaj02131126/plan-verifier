# Pre-Execution Plan Verifier: Research Design Spec

**Target:** AAAI main-track paper (8 pages + references)
**Subject areas:** PRS (Planning, Routing & Scheduling), KRR (Knowledge Representation & Reasoning), MAS (Multi-Agent Systems, framed as future work), ML (learned uncertainty/calibration)
**Constraint:** everything runs on a laptop — no fine-tuning, no GPU cluster, no heavyweight simulators.

---

## 1. One-sentence pitch

A lightweight, model-agnostic verifier that sits between "LLM proposes a plan" and "system executes it," combining a sound symbolic checker (consistency, goal-completeness, resource-feasibility) with a small learned model that estimates how much to trust the symbolic checker's own parse of the LLM's plan — catching both hard constraint violations and the softer semantic gaps that symbolic checking alone is blind to.

## 2. Why this is a main-track paper, not a demo

A pure symbolic checker is old news (PDDL validators like VAL have existed for 20 years) and a pure "ask an LLM if this plan is good" judge is what most current agent papers already do (and it's known to be unreliable — LLMs are bad at catching their own resource/consistency errors, and are overconfident). The contribution is the **fusion**: showing that (a) hard, provably-checkable constraints should never be delegated to a learned model, but (b) the translation from an LLM's free-text/tool-call plan into a checkable symbolic form is itself a lossy, uncertain step, and a small learned scorer over cheap features can flag exactly the plans where that translation is untrustworthy — recovering errors invisible to either component alone. This is a **neuro-symbolic reliability** story, which is a well-recognized AAAI framing, plus a real empirical payoff (does gating execution on this actually raise task success rate).

## 3. Problem formalization

**Input:**
- A goal specification `G` (natural language + optionally a structured goal condition)
- An initial state / environment description `S0`
- A plan `P = (a1, ..., an)` produced by an LLM, where each `ai` is either a natural-language step, a structured tool call, or a PDDL-style action — depending on domain

**Verifier output**, per plan (and optionally per step):
1. **Consistency** — no action's stated preconditions are contradicted by the effects of prior actions; no two actions in the plan assert incompatible facts; argument/type-checking passes.
2. **Goal-completeness** — simulating `P` from `S0` (via progression) entails `G`; if `G` is a conjunction, report which conjuncts are unmet.
3. **Resource-feasibility** — for each declared resource dimension (budget, time, API quota, item count, capacity), the cumulative consumption trace never goes negative / never exceeds a cap, checked at every prefix of the plan.
4. **Trust score** — a calibrated probability, from the learned layer, that the symbolic verdict above is actually faithful to what the LLM intended (i.e., that the NL→schema parse didn't drop an implicit precondition, hallucinate an effect, or mis-extract a resource cost).

**Final decision rule:** reject if any hard symbolic check fails; else reject/flag if trust score < threshold; else accept. This gives a precision-recall curve you can sweep for the paper, and a clean explanation for every reject ("violates precondition X" vs. "low-confidence parse, step 3").

## 4. Hybrid architecture

### 4.1 Symbolic layer (sound, deterministic)
- A small typed predicate/resource schema per domain (hand-authored for the synthetic domain, semi-automatically derived from the tool spec for the tool-use domain).
- An LLM-based **structured extractor** (function-calling / constrained JSON output) maps each plan step into `{action_type, args, preconditions_referenced, resource_deltas}`. This extractor is *not* the thing being trusted blindly — its own confidence/consistency is a key input feature to the learned layer.
- Forward simulation (progression) over the extracted action sequence checks consistency and goal-completeness; a simple running-sum / linear feasibility check per resource dimension checks resource-feasibility. All of this is classical, explainable, and cheap (milliseconds).

### 4.2 Learned layer (calibrated uncertainty)
A lightweight model (logistic regression or small MLP over hand/embedding features — no fine-tuning) trained to predict "is the symbolic parse of this step/plan trustworthy" using features such as:
- Self-consistency across k=3-5 resampled extractions of the same step (do they agree?)
- Token-level log-probability / entropy of the LLM's plan text at that step, if available from the API
- Embedding similarity between the stated preconditions and the schema's expected preconditions for that action type
- Parser confidence / whether required slots were filled vs. defaulted
- Plan-level features: plan length, number of distinct resource types touched, domain familiarity (seen action types vs. novel ones)

Output: a calibrated probability (train with a proper scoring rule; report ECE). This is the "learned uncertainty scoring" — small enough to train on a few thousand labeled examples on a laptop in minutes.

### 4.3 Fusion
Hard symbolic failures always reject (soundness preserved — never execute a provably broken plan). Among symbolic passes, the trust score ranks residual risk. Report both the hard-rejection catch rate and the ROC/PR curve of the trust score on the remaining "symbolically clean but actually flawed" plans (this second number is the one that differentiates the paper from "just build a PDDL validator").

## 5. Evaluation domains

**Primary — synthetic PDDL-style planning (rigorous, cheap ground truth).**
Extend classic short-horizon domains (Blocksworld/Logistics/Gripper-style) with resource dimensions (fuel, budget, capacity). Generate problems programmatically, solve with a simple custom/existing planner for gold plans, then prompt an LLM for candidate plans under prompt variations designed to induce realistic failure modes (goal omission, precondition slip, resource overrun, hallucinated effect). Ground truth flaw labels come for free by diffing against the domain model — fully automatic, scalable, no human labeling needed.

**Secondary — tool-use / API-calling benchmark (generality, topical relevance).**
A small self-built mock tool environment (5-10 tools with auth scopes, rate limits, typed args, quotas). LLM plans sequences of tool calls to satisfy a task; the mock environment auto-labels flaws when the plan is dry-run against it (invalid call ordering, missing prerequisite call, quota exceeded, wrong arg type). This is intentionally scoped smaller than ToolBench/API-Bank so it stays laptop-fast, but mirrors their failure taxonomy.

*(A third, web-environment domain was considered and deliberately dropped — MiniWoB/WebArena-style harnesses add engineering cost and noisy ground truth without proportionate rigor gain; two domains with clean, automatic labeling is a stronger reviewer story than three domains with a shakier one. This can be revisited as a rebuttal-stage addition if reviewers ask for more breadth.)*

## 6. Baselines (needed for main-track credibility)

1. **Pure LLM-judge** — zero-shot / CoT "is this plan valid?" prompt, same base LLM.
2. **Pure symbolic checker** — the verifier's own symbolic layer alone (ablation; shows what learned layer adds).
3. **Pure learned layer** — a classifier (few-shot prompted or the same small model) predicting plan validity with no symbolic grounding (ablation; shows what symbolic layer adds).
4. **Self-repair / Reflexion-style** — re-prompt the LLM to critique and fix its own plan; compare catch rate.
5. **Oracle sanity check** (synthetic domain only) — compare against an off-the-shelf PDDL validator (e.g., VAL) to confirm the custom symbolic layer's soundness.

## 7. Metrics

- Precision / recall / F1 for flaw detection, overall and broken down by flaw type (inconsistency, goal-incompleteness, resource-infeasibility, "symbolically clean but semantically flawed").
- Calibration of the trust score (ECE, reliability diagrams).
- Downstream task success rate with vs. without the verifier gating execution (allow one re-plan attempt on rejection).
- Latency / cost overhead vs. pure LLM-judge baseline.
- Ablation table: symbolic-only vs. learned-only vs. hybrid, on both domains.

## 8. Related work positioning (for the paper's intro/related work section)

Position against: classical plan validation (VAL, POPF), LLM self-critique / Reflexion, tool-use safety/verification work (ToolEmu-style), neuro-symbolic reasoning generally, and calibration/uncertainty-quantification for LLMs. The novelty claim is specifically the **two-tier trust model** (hard symbolic gate + calibrated soft trust on the parse step), not either component alone.

## 9. Build phases (laptop-scale)

- Repo scaffolding, schema DSL, synthetic domain generator + gold planner.
- Symbolic verifier (consistency/goal/resource checks) + LLM plan-generation harness with induced failure modes; automatic labeling.
- Structured extractor + feature pipeline for the learned layer; label a training set.
- Train/calibrate the learned trust model; build fusion + decision rule; first end-to-end pass on synthetic domain.
- Build tool-use mock environment + harness; port pipeline to second domain.
- Implement all baselines; run full evaluation grid on both domains.
- Ablations, calibration analysis, downstream-success experiment (gated execution loop), error analysis / qualitative examples.
- Writing, figures, tables, reproducibility pass (README, seeds, config files).
- Buffer for reviewer-style self-critique and submission.

## 10. Risks / things to watch

- LLM-induced failure modes must look *realistic*, not just adversarially broken — otherwise reviewers will say the benchmark is too easy. Mitigation: calibrate failure-injection prompts against a pilot batch of "naturally occurring" LLM plan failures (unprompted) and match the distribution.
- The learned layer must be shown to add value beyond what a slightly-better symbolic schema would catch — this is the most likely reviewer pushback. Mitigation: the ablation table and specifically the "symbolically clean but flagged by trust score, later confirmed flawed" case studies are the paper's load-bearing evidence.
- Keep the tool-use mock environment small and clean — the goal is a controlled second domain, not a from-scratch ToolBench clone.
