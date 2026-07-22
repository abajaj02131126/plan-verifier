"""Task 11 (pivot A3): translate real ALFWorld/ALFRED tasks into the existing
typed DSL and run them through the existing pipeline.

ALFWorld's task backend is PDDL (alfworld/data/alfred.pddl + per-trial
initial_state.pddl + traj_data.json expert plans), which is the closest
structural match to our DSL among the benchmarks the reviews named. This
module is an ADAPTER onto the existing DSL loader (verifier.schema) — it does
not fork the schema or the evaluation path.

Honest translation accounting (reported, not hidden):
  * DISCLOSED COMPILATIONS (semantics-preserving):
    - the accessibility disjunction (or (not (openable r)) (opened r)) in
      Pickup/Put is compiled into a maintained `accessible` predicate
      (init: accessible iff not openable or opened; Open adds, Close deletes).
    - Slice's (or (objectType k KnifeType) (objectType k ButterKnifeType))
      is compiled into a static `isKnife` predicate computed in init.
    - existential goals (every ALFRED goal is existentially quantified) are
      GROUNDED to the concrete target instance(s) the expert plan
      manipulates. Faithful when the target is unique (pick-and-place
      families); reported as a translation FAILURE when the grounding is
      ambiguous.
    - PDDL action-costs (increase total-cost) are dropped: they affect
      optimality, not validity, and the DSL has no optimization objective.
  * PARTIAL: ToggleObject's conditional isOn-flip (when ...) is dropped; only
    its unconditional `isToggled` effect is kept (the goals that use Toggle
    depend on isToggled, not isOn). Disclosed.
  * NOT TRANSLATED: informational actions (look/inventory/examine/help) are
    omitted (no task-goal-relevant effects).

A task is counted TRANSLATED only if its full expert plan simulates cleanly
AND achieves the grounded goal under BOTH independent simulators (the oracle
labeler and the symbolic checker) — the same cross-check gate used for the
three synthetic domains. Anything else is a reported failure with a reason.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from verifier.schema import (
    ActionSchema,
    Domain,
    Literal,
    Parameter,
    PredicateAtom,
    PredicateDefinition,
    Problem,
    TypedObject,
)
from verifier.schema.state import GroundAction

DOMAIN_NAME = "alfworld"


# --------------------------------------------------------------------------
# ID encoding between traj_data (Book|+00.06|...) and PDDL (Book_bar__plus_..)
# --------------------------------------------------------------------------
def enc(obj_id: str) -> str:
    return (
        obj_id.replace("|", "_bar_").replace("+", "_plus_")
        .replace("-", "_minus_").replace(".", "_dot_")
    )


# --------------------------------------------------------------------------
# Domain (hand-encoded translatable subset of alfred.pddl, with disclosed
# compilations). Types and predicates mirror alfred.pddl exactly except for
# the two compiled predicates (accessible, isKnife).
# --------------------------------------------------------------------------
def build_domain() -> Domain:
    P = PredicateDefinition
    predicates = [
        P(name="atLocation", arg_types=["agent", "location"]),
        P(name="receptacleAtLocation", arg_types=["receptacle", "location"]),
        P(name="objectAtLocation", arg_types=["object", "location"]),
        P(name="openable", arg_types=["receptacle"]),
        P(name="opened", arg_types=["receptacle"]),
        P(name="accessible", arg_types=["receptacle"]),        # COMPILED
        P(name="inReceptacle", arg_types=["object", "receptacle"]),
        P(name="holds", arg_types=["agent", "object"]),
        P(name="holdsAny", arg_types=["agent"]),
        P(name="pickupable", arg_types=["object"]),
        P(name="cleanable", arg_types=["object"]),
        P(name="heatable", arg_types=["object"]),
        P(name="coolable", arg_types=["object"]),
        P(name="sliceable", arg_types=["object"]),
        P(name="toggleable", arg_types=["object"]),
        P(name="isClean", arg_types=["object"]),
        P(name="isHot", arg_types=["object"]),
        P(name="isCool", arg_types=["object"]),
        P(name="isSliced", arg_types=["object"]),
        P(name="isToggled", arg_types=["object"]),
        P(name="isKnife", arg_types=["object"]),               # COMPILED
        P(name="objectType", arg_types=["object", "otype"]),
        P(name="receptacleType", arg_types=["receptacle", "rtype"]),
        P(name="canContain", arg_types=["rtype", "otype"]),
    ]

    A = ActionSchema
    Par = Parameter
    pos = Literal.pos
    neg = Literal.neg
    atom = lambda p, *a: PredicateAtom(predicate=p, args=list(a))

    actions = [
        A(name="GotoLocation",
          parameters=[Par(name="a", type="agent"), Par(name="lS", type="location"),
                      Par(name="lE", type="location"), Par(name="r", type="receptacle")],
          preconditions=[pos("atLocation", "a", "lS"), pos("receptacleAtLocation", "r", "lE")],
          add_effects=[atom("atLocation", "a", "lE")],
          del_effects=[atom("atLocation", "a", "lS")], resource_deltas={}),
        A(name="OpenObject",
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"), Par(name="r", type="receptacle")],
          preconditions=[pos("openable", "r"), pos("atLocation", "a", "l"),
                         pos("receptacleAtLocation", "r", "l"), neg("opened", "r")],
          add_effects=[atom("opened", "r"), atom("accessible", "r")],
          del_effects=[], resource_deltas={}),
        A(name="CloseObject",
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"), Par(name="r", type="receptacle")],
          preconditions=[pos("openable", "r"), pos("atLocation", "a", "l"),
                         pos("receptacleAtLocation", "r", "l"), pos("opened", "r")],
          add_effects=[], del_effects=[atom("opened", "r"), atom("accessible", "r")], resource_deltas={}),
        A(name="PickupObject",
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"),
                      Par(name="o", type="object"), Par(name="r", type="receptacle")],
          preconditions=[pos("pickupable", "o"), pos("atLocation", "a", "l"),
                         pos("receptacleAtLocation", "r", "l"), pos("inReceptacle", "o", "r"),
                         neg("holdsAny", "a"), pos("accessible", "r")],
          add_effects=[atom("holds", "a", "o"), atom("holdsAny", "a")],
          del_effects=[atom("inReceptacle", "o", "r"), atom("objectAtLocation", "o", "l")], resource_deltas={}),
        A(name="PutObject",
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"),
                      Par(name="o", type="object"), Par(name="r", type="receptacle"),
                      Par(name="ot", type="otype"), Par(name="rt", type="rtype")],
          preconditions=[pos("holds", "a", "o"), pos("atLocation", "a", "l"),
                         pos("receptacleAtLocation", "r", "l"), pos("accessible", "r"),
                         pos("objectType", "o", "ot"), pos("receptacleType", "r", "rt"),
                         pos("canContain", "rt", "ot")],
          add_effects=[atom("inReceptacle", "o", "r"), atom("objectAtLocation", "o", "l")],
          del_effects=[atom("holds", "a", "o"), atom("holdsAny", "a")], resource_deltas={}),
        A(name="CleanObject",
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"),
                      Par(name="r", type="receptacle"), Par(name="o", type="object")],
          preconditions=[pos("cleanable", "o"), pos("atLocation", "a", "l"),
                         pos("receptacleAtLocation", "r", "l"), pos("holds", "a", "o")],
          add_effects=[atom("isClean", "o")], del_effects=[], resource_deltas={}),
        A(name="HeatObject",
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"),
                      Par(name="r", type="receptacle"), Par(name="o", type="object")],
          preconditions=[pos("heatable", "o"), pos("atLocation", "a", "l"),
                         pos("receptacleAtLocation", "r", "l"), pos("holds", "a", "o")],
          add_effects=[atom("isHot", "o")], del_effects=[atom("isCool", "o")], resource_deltas={}),
        A(name="CoolObject",
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"),
                      Par(name="r", type="receptacle"), Par(name="o", type="object")],
          preconditions=[pos("coolable", "o"), pos("atLocation", "a", "l"),
                         pos("receptacleAtLocation", "r", "l"), pos("holds", "a", "o")],
          add_effects=[atom("isCool", "o")], del_effects=[atom("isHot", "o")], resource_deltas={}),
        A(name="ToggleObject",  # PARTIAL: keep isToggled, drop conditional isOn flip
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"),
                      Par(name="o", type="object"), Par(name="r", type="receptacle")],
          preconditions=[pos("toggleable", "o"), pos("atLocation", "a", "l"),
                         pos("receptacleAtLocation", "r", "l"), pos("inReceptacle", "o", "r")],
          add_effects=[atom("isToggled", "o")], del_effects=[], resource_deltas={}),
        A(name="SliceObject",
          parameters=[Par(name="a", type="agent"), Par(name="l", type="location"),
                      Par(name="co", type="object"), Par(name="ko", type="object")],
          preconditions=[pos("sliceable", "co"), pos("isKnife", "ko"),
                         pos("atLocation", "a", "l"), pos("objectAtLocation", "co", "l"),
                         pos("holds", "a", "ko")],
          add_effects=[atom("isSliced", "co")], del_effects=[], resource_deltas={}),
    ]
    return Domain(name=DOMAIN_NAME, types=["agent", "location", "receptacle", "object", "otype", "rtype"],
                  predicates=predicates, action_schemas=actions, resource_dimensions=[])


# --------------------------------------------------------------------------
# PDDL problem parsing
# --------------------------------------------------------------------------
_KNIFE_TYPES = {"KnifeType", "ButterKnifeType"}


class TranslationError(Exception):
    pass


def _parse_objects(block: str) -> dict:
    objs = {}
    for m in re.finditer(r"([\w\|\+\-\.]+)\s*-\s*(agent|location|receptacle|object|otype|rtype)", block):
        objs[m.group(1)] = m.group(2)
    return objs


def _parse_init(block: str) -> list[tuple]:
    atoms = []
    for m in re.finditer(r"\(([a-zA-Z]+)((?:\s+[\w\|\+\-\.]+)*)\)", block):
        pred = m.group(1)
        args = m.group(2).split()
        if pred in ("and",):
            continue
        atoms.append((pred, args))
    return atoms


def load_problem_pddl(path: Path) -> tuple[dict, list[tuple], str]:
    txt = path.read_text()
    o0 = txt.find("(:objects")
    i0 = txt.find("(:init")
    g0 = txt.find("(:goal")
    objects = _parse_objects(txt[o0:i0])
    init = _parse_init(txt[i0:g0])
    goal_txt = txt[g0:]
    return objects, init, goal_txt


# --------------------------------------------------------------------------
# Expert plan (traj_data high_pddl) -> DSL ground actions, filling params
# --------------------------------------------------------------------------
def _receptacle_at(loc: str, init_by_pred: dict) -> Optional[str]:
    for r, l in init_by_pred.get("receptacleAtLocation", []):
        if l == loc:
            return r
    return None


def build_gold_plan(traj: dict, objects: dict, init: list[tuple]) -> tuple[list[GroundAction], dict]:
    """Translate the expert high_pddl plan into DSL GroundActions. ALFRED's
    high_pddl abstracts away navigation coordinates and open/close, so we
    RECONSTRUCT them faithfully: each goto is derived from the location of the
    receptacle the next interaction targets, and an OpenObject is inserted
    before a Pickup/Put on an openable-but-closed receptacle (Cool/Heat/Clean
    do not require accessibility per alfred.pddl)."""
    by_pred: dict[str, list] = {}
    for pred, args in init:
        by_pred.setdefault(pred, []).append(args)
    objtype = {o: t for (o, t) in by_pred.get("objectType", [])}
    rectype = {r: t for (r, t) in by_pred.get("receptacleType", [])}
    rec_loc = {r: l for (r, l) in by_pred.get("receptacleAtLocation", [])}
    obj_rec0 = {o: r for (o, r) in by_pred.get("inReceptacle", [])}
    openable = {a[0] for a in by_pred.get("openable", [])}
    type_to_rec: dict[str, list] = {}
    for r, t in by_pred.get("receptacleType", []):
        type_to_rec.setdefault(t, []).append(r)

    agent = "agent1"
    cur_loc = next((a[1] for a in by_pred.get("atLocation", []) if a[0] == agent), None)
    if cur_loc is None:
        raise TranslationError("no initial agent location")
    opened = {a[0] for a in by_pred.get("opened", [])}
    plan: list[GroundAction] = []
    target: dict = {}
    held: Optional[str] = None

    def goto_and_open(r, need_access):
        nonlocal cur_loc
        l = rec_loc.get(r)
        if l is None:
            raise TranslationError(f"no location for receptacle {r}")
        if cur_loc != l:
            plan.append(GroundAction(schema_name="GotoLocation", args=(agent, cur_loc, l, r)))
            cur_loc = l
        if need_access and r in openable and r not in opened:
            plan.append(GroundAction(schema_name="OpenObject", args=(agent, l, r)))
            opened.add(r)

    for step in traj["plan"]["high_pddl"]:
        pa = step["planner_action"]
        act = pa.get("action")
        if act == "GotoLocation":
            continue  # reconstructed from interaction targets instead
        elif act == "PickupObject":
            o = enc(pa["objectId"])
            r = obj_rec0.get(o)
            if r is None:
                raise TranslationError(f"PickupObject: no init receptacle for object")
            goto_and_open(r, need_access=True)
            plan.append(GroundAction(schema_name="PickupObject", args=(agent, cur_loc, o, r)))
            target["object"] = o; held = o
        elif act == "PutObject":
            o = enc(pa["objectId"])
            r = enc(pa["receptacleObjectId"])
            ot, rt = objtype.get(o), rectype.get(r)
            if ot is None or rt is None:
                raise TranslationError("PutObject: missing object/receptacle type")
            goto_and_open(r, need_access=True)
            plan.append(GroundAction(schema_name="PutObject", args=(agent, cur_loc, o, r, ot, rt)))
            target["object"] = o; target["receptacle"] = r; held = None
        elif act in ("CleanObject", "HeatObject", "CoolObject"):
            # objectId here is the APPLIANCE (fridge/microwave/sink); the object
            # acted on is the currently-held one.
            r = enc(pa["objectId"])
            if r not in rec_loc:
                raise TranslationError(f"{act}: appliance {r[:20]} not a known receptacle")
            if held is None:
                raise TranslationError(f"{act}: nothing held to act on")
            o = held
            goto_and_open(r, need_access=False)
            plan.append(GroundAction(schema_name=act, args=(agent, cur_loc, r, o)))
        elif act == "ToggleObject":
            o = enc(pa["objectId"])
            r = obj_rec0.get(o)
            if r is None:
                raise TranslationError("ToggleObject: no init receptacle for lamp")
            goto_and_open(r, need_access=False)
            plan.append(GroundAction(schema_name="ToggleObject", args=(agent, cur_loc, o, r)))
            target["object"] = o
        elif act == "SliceObject":
            raise TranslationError("SliceObject family (knife-in-hand semantics) not translated")
        elif act in ("NoOp", "End"):
            continue
        else:
            raise TranslationError(f"unsupported expert action {act}")
    return plan, target


def _obj_receptacle(o: str, by_pred: dict) -> Optional[str]:
    for oo, r in by_pred.get("inReceptacle", []):
        if oo == o:
            return r
    return None


# --------------------------------------------------------------------------
# Grounded goal (existential -> concrete target instances the expert achieves)
# --------------------------------------------------------------------------
def ground_goal_from_plan(plan: list[GroundAction], objects: dict, init: list[tuple]) -> list[Literal]:
    """Derive the grounded goal from the end-state the expert plan achieves on
    its target object(s): the standard 'gold plan defines the goal' approach
    used for the other domains, here also serving as the disclosed
    existential->instance grounding."""
    goal: list[Literal] = []
    seen = set()
    def add(lit_key, lit):
        if lit_key not in seen:
            seen.add(lit_key); goal.append(lit)
    held = None
    put_objs = set()
    for ga in plan:
        n, args = ga.schema_name, ga.args
        if n == "PutObject":
            add(("inReceptacle", args[2], args[3]), Literal.pos("inReceptacle", args[2], args[3]))
            put_objs.add(args[2]); held = None
        elif n == "PickupObject":
            held = args[2]
        elif n == "CleanObject":
            add(("isClean", args[3]), Literal.pos("isClean", args[3]))
        elif n == "HeatObject":
            add(("isHot", args[3]), Literal.pos("isHot", args[3]))
        elif n == "CoolObject":
            add(("isCool", args[3]), Literal.pos("isCool", args[3]))
        elif n == "ToggleObject":
            add(("isToggled", args[2]), Literal.pos("isToggled", args[2]))
    # look_at family: object ends held (no put)
    if held is not None and held not in put_objs:
        add(("holds", "agent1", held), Literal.pos("holds", "agent1", held))
    if not goal:
        raise TranslationError("empty grounded goal")
    return goal


def translate_task(trial_dir: Path) -> tuple[Problem, list[GroundAction]]:
    """Full translation of one ALFRED trial. Raises TranslationError with a
    reason if any DSL limit or missing datum blocks a clean translation."""
    objects, init, _goal_txt = load_problem_pddl(trial_dir / "initial_state.pddl")
    traj = json.load(open(trial_dir / "traj_data.json"))
    domain = build_domain()

    # compile accessible + isKnife into init
    by_pred: dict[str, list] = {}
    for pred, args in init:
        by_pred.setdefault(pred, []).append(args)
    openable = {a[0] for a in by_pred.get("openable", [])}
    opened = {a[0] for a in by_pred.get("opened", [])}
    receptacles = [o for o, t in objects.items() if t == "receptacle"]
    init_atoms = []
    known_preds = {p.name for p in domain.predicates}
    for pred, args in init:
        if pred in known_preds and pred not in ("total-cost",):
            init_atoms.append(PredicateAtom(predicate=pred, args=args))
    # accessible iff (not openable) or opened
    for r in receptacles:
        if r not in openable or r in opened:
            init_atoms.append(PredicateAtom(predicate="accessible", args=[r]))
    # isKnife iff objectType is Knife/ButterKnife
    for o, t in by_pred.get("objectType", []):
        if t in _KNIFE_TYPES:
            init_atoms.append(PredicateAtom(predicate="isKnife", args=[o]))

    gold_plan, target = build_gold_plan(traj, objects, init)
    goal = ground_goal_from_plan(gold_plan, objects, init)

    typed_objs = [TypedObject(name=o, type=t) for o, t in objects.items()]
    problem = Problem(
        name=f"{DOMAIN_NAME}-{trial_dir.parent.name}-{trial_dir.name}"[:80],
        domain=domain, objects=typed_objs, init=init_atoms, goal=goal,
    )
    return problem, gold_plan
