"""Task 12 (pivot A3): real-API-trace extension of the Tools domain.

Supplements the self-authored synthetic Tools domain with a handful of REAL
public-API operations, encoded in the SAME Tools DSL representation
(identity-predicate-pinned auth scopes, api-quota + budget resource
dimensions, boolean prerequisite facts) WITHOUT extending the DSL. These are
modeled faithfully from the public API specs of Stripe, SendGrid, GitHub,
Slack, and Google Calendar — real OAuth scopes and real prerequisite chains:

  authenticate(scope)                 -> authed(scope)
  stripe_create_intent(order, scope)  needs stripe-write            -> intent-created(order); -amount budget
  stripe_confirm(order, scope)        needs stripe-write + intent   -> payment-confirmed(order)
  sendgrid_send(order, scope)         needs sendgrid-mail + payment -> email-sent(order)
  github_create_issue(order, scope)   needs github-repo             -> issue-created(order)
  github_comment(order, scope)        needs github-repo + issue     -> comment-added(order)
  slack_post(order, scope)            needs slack-chatwrite         -> slack-posted(order)
  gcal_insert(order, scope)           needs gcal-events             -> event-created(order)

NAMED DSL LIMITATION (reported, not silently approximated, per Task 12):
real APIs return typed values (a Stripe PaymentIntent id, a GitHub issue
number) that later calls consume by value. The DSL has no typed return
values, so — exactly as the synthetic Tools domain already does — a return
value is modeled as a BOOLEAN prerequisite fact (intent-created(order))
rather than a passed id. This is adequate for the ordering/prerequisite/
quota/spend flaw taxonomy but does not capture value-level errors (e.g.
passing the wrong intent id); disclosed as a concrete limitation.

Runs the same oracle cross-check (independent labeler == symbolic checker)
used for every other domain, on gold plans plus deterministically injected
flaws, and reports the result labeled as the REAL-TRACE subset of Tools.
"""
from __future__ import annotations
import collections
import json
import random
from pathlib import Path

from verifier.schema import (
    ActionSchema, Domain, Literal, Parameter, PredicateAtom, PredicateDefinition,
    Problem, ResourceDimension, TypedObject,
)
from verifier.schema.state import GroundAction
from verifier.domains.planner import bfs_plan, PlannerTimeout
from verifier.generation.labeler import label_plan
from verifier.generation.parser import ParsedStep, ParseResult
from verifier.symbolic.checker import verify

SCOPES = ["stripe-write", "sendgrid-mail", "github-repo", "slack-chatwrite", "gcal-events"]
MILESTONES = ["payment-confirmed", "email-sent", "issue-created", "comment-added",
              "slack-posted", "event-created"]


def build_domain(api_quota: float = 40.0, budget: float = 5000.0) -> Domain:
    P = PredicateDefinition
    preds = [
        P(name="authed", arg_types=["scope"]),
        P(name="intent-created", arg_types=["order"]),
        P(name="payment-confirmed", arg_types=["order"]),
        P(name="email-sent", arg_types=["order"]),
        P(name="issue-created", arg_types=["order"]),
        P(name="comment-added", arg_types=["order"]),
        P(name="slack-posted", arg_types=["order"]),
        P(name="event-created", arg_types=["order"]),
    ] + [P(name=f"is-{s}", arg_types=["scope"]) for s in SCOPES]

    o = lambda: Parameter(name="order", type="order")
    s = lambda: Parameter(name="scope", type="scope")
    atom = lambda p, *a: PredicateAtom(predicate=p, args=list(a))

    def tool(name, scope_pred, extra_pre, add, deltas):
        pre = [Literal.pos(f"is-{scope_pred}", "scope"), Literal.pos("authed", "scope")] + extra_pre
        return ActionSchema(name=name, parameters=[o(), s()], preconditions=pre,
                            add_effects=[atom(add, "order")], del_effects=[], resource_deltas=deltas)

    actions = [
        ActionSchema(name="authenticate", parameters=[s()], preconditions=[],
                     add_effects=[atom("authed", "scope")], del_effects=[], resource_deltas={"api-quota": -1.0}),
        tool("stripe_create_intent", "stripe-write", [], "intent-created", {"api-quota": -1.0, "budget": -100.0}),
        tool("stripe_confirm", "stripe-write", [Literal.pos("intent-created", "order")], "payment-confirmed", {"api-quota": -1.0}),
        tool("sendgrid_send", "sendgrid-mail", [Literal.pos("payment-confirmed", "order")], "email-sent", {"api-quota": -1.0}),
        tool("github_create_issue", "github-repo", [], "issue-created", {"api-quota": -1.0}),
        tool("github_comment", "github-repo", [Literal.pos("issue-created", "order")], "comment-added", {"api-quota": -1.0}),
        tool("slack_post", "slack-chatwrite", [], "slack-posted", {"api-quota": -1.0}),
        tool("gcal_insert", "gcal-events", [], "event-created", {"api-quota": -1.0}),
    ]
    return Domain(name="tools", types=["order", "scope"], predicates=preds, action_schemas=actions,
                  resource_dimensions=[ResourceDimension(name="api-quota", initial=api_quota, cap=api_quota),
                                       ResourceDimension(name="budget", initial=budget, cap=budget)])


def generate_problem(seed: int, index: int):
    domain = build_domain()
    for attempt in range(50):
        rng = random.Random(f"tools-real:{seed}:{index}:{attempt}")
        n_orders = rng.choice([1, 1, 2])
        orders = [f"order{i}" for i in range(1, n_orders + 1)]
        objects = [TypedObject(name=o, type="order") for o in orders] + \
                  [TypedObject(name=s, type="scope") for s in SCOPES]
        init = [PredicateAtom(predicate=f"is-{s}", args=[s]) for s in SCOPES]
        for s in SCOPES:
            if rng.random() < 0.3:
                init.append(PredicateAtom(predicate="authed", args=[s]))
        goal = []
        for order in orders:
            for m in rng.sample(MILESTONES, rng.choice([1, 2, 2])):
                goal.append(Literal.pos(m, order))
        if not goal:
            continue
        problem = Problem(name=f"tools-real-{seed}-{index}", domain=domain, objects=objects,
                          init=init, goal=sorted(goal, key=str))
        try:
            gold = bfs_plan(problem, max_depth=12)
        except PlannerTimeout:
            continue
        if gold and 3 <= len(gold) <= 12:
            return problem, gold
    raise RuntimeError("no real-trace tools problem")


def as_parse_result(plan):
    return ParseResult(steps=[ParsedStep(raw_line=repr(g), action=g) for g in plan])


def _inject(gold):
    out = []
    # goal_incompleteness: drop last action
    if len(gold) > 1:
        out.append((gold[:-1], "goal_incompleteness"))
    # inconsistency: drop an authenticate so a later call lacks its scope
    for i, g in enumerate(gold):
        if g.schema_name == "authenticate":
            out.append((gold[:i] + gold[i+1:], "inconsistency"))
            break
    # hallucinated
    out.append(([GroundAction(schema_name="refund_everything", args=("order1",))] + gold, "hallucinated_action"))
    return out


def main():
    problems = [generate_problem(seed=0, index=i) for i in range(8)]
    agree = 0; total = 0
    per_type = collections.Counter(); per_type_caught = collections.Counter()
    for problem, gold in problems:
        for plan, expect_valid, ftype in [(gold, True, None)] + [(p, False, ft) for p, ft in _inject(gold)]:
            labels = label_plan(problem, as_parse_result(plan))
            verdict = verify(plan, problem)
            total += 1
            if labels.overall_valid == verdict.overall_valid:
                agree += 1
            if not expect_valid:
                per_type[ftype] += 1
                if not verdict.overall_valid:
                    per_type_caught[ftype] += 1
    out = {
        "subset": "tools/real-trace",
        "apis_modeled": ["Stripe", "SendGrid", "GitHub", "Slack", "Google Calendar"],
        "n_real_trace_problems": len(problems),
        "n_records": total,
        "oracle_cross_check_agreement": f"{agree}/{total}",
        "per_flaw_type_recall": {ft: {"recall": round(per_type_caught[ft]/per_type[ft], 3), "support": per_type[ft]}
                                 for ft in sorted(per_type)},
        "named_limitation": "no typed return values: API return ids modeled as boolean prerequisite facts "
                            "(same simplification as the synthetic Tools domain); value-level errors not captured.",
    }
    Path("pivot/results").mkdir(parents=True, exist_ok=True)
    Path("pivot/results/tools_realtrace.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
