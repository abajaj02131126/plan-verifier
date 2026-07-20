"""Synthetic PDDL-style domains with resource dimensions, plus a registry
mapping domain name -> (build_domain, generate_problem)."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from verifier.schema import Domain, Problem
from verifier.schema.state import GroundAction
from verifier.domains import blocksworld, logistics
from verifier.domains import tools as tools_domain


@dataclass(frozen=True)
class DomainSpec:
    name: str
    build_domain: Callable[..., Domain]
    generate_problem: Callable[..., Tuple[Problem, List[GroundAction]]]


DOMAIN_REGISTRY: Dict[str, DomainSpec] = {
    "blocksworld": DomainSpec("blocksworld", blocksworld.build_domain, blocksworld.generate_problem),
    "logistics": DomainSpec("logistics", logistics.build_domain, logistics.generate_problem),
    "tools": DomainSpec("tools", tools_domain.build_domain, tools_domain.generate_problem),
}

__all__ = ["DomainSpec", "DOMAIN_REGISTRY"]