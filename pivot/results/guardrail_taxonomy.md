# Guardrail-collapse taxonomy (Task 5)

Observed judge-budget sweep (Section 5.5, 360 short-horizon records):

| budget (tok) | unparseable | P | R | F1 |
|---|---|---|---|---|
| 64 | 360/360 (100%) | 0.0 | 0.0 | 0.0 |
| 512 | 146/360 (41%) | - | - | - |
| 2048 | 0/360 (0%) | 1.0 | 1.0 | 1.0 |

## Silent-failure modes

| category | trigger | observed effect on P/R | deployment default | checker immune? |
|---|---|---|---|---|
| truncation-before-verdict | output budget too small for the model's actual verbosity (haiku simulates the plan step-by-step regardless of the 'answer in one line' instruction); observed at 64 and 512 tokens | unparseable-verdict rate 100% at 64 tokens, ~41% at 512; no exception is raised — the call 'succeeds' with a truncated body | n/a (mechanism); becomes catastrophic or benign depending on the unparseable-handling policy below | yes — the symbolic checker has no generative budget to exhaust |
| fail-open-on-unparseable-output | any unparseable verdict combined with the naive deployment default of treating 'no parseable verdict' as ACCEPT | at 64 tokens this turns 100% truncation into P=R=F1=0.0: every flawed plan is accepted and executed, while the guardrail appears to run normally (no errors, fast 'approvals') | FAIL-OPEN (observed default). A FAIL-CLOSED default would instead reject all 360 (false-reject storm) — same bug, opposite failure; either way the judge's verdict is meaningless, just silently | yes — the checker is a total deterministic function; it always emits a parseable verdict and fails CLOSED (an unrunnable step is a violation) |

## Worked example: apparent vs true expected cost

| domain | stakes S | apparent (truncation undetected) | true (recall 0) | hidden gap |
|---|---|---|---|---|
| blocksworld | 1 | 0 | 1 | 1 |
| logistics | 10 | 0 | 10 | 10 |
| tools | 50 | 0 | 50 | 50 |

Because Sonnet-5 matches the checker on accuracy (Task 1/1b), the checker's durable, tier-INDEPENDENT value is that it structurally cannot enter these failure modes: no token budget (immune to truncation), total deterministic verdict (cannot produce unparseable output), fails CLOSED by construction, and zero marginal decision-time cost. These hold regardless of which judge tier a deployer would otherwise trust.
