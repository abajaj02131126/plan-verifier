"""Mock tool-use domain (spec section 5, secondary domain).

Eight mock tools for a travel-booking assistant, expressed in the SAME schema
DSL as the planning domains — action = tool call, prerequisite-call
constraints and auth scopes = predicates (a completed call's "return value"
is threaded to later calls as a trip-scoped boolean predicate, e.g.
search_flights adds flight-options(trip) which book_flight requires), rate
limits/quotas and spend = resource dimensions. No DSL changes were needed;
the one representational simplification is that return values are boolean
availability facts rather than typed values (adequate for the flaw taxonomy:
invalid ordering, missing prerequisite, quota exceeded, wrong arg type via
the type system, budget exceeded).

The "mock environment dry-run" that auto-labels flaws is therefore exactly
the Phase 2 oracle labeler / Phase 4 symbolic verifier running on this
domain — mirroring the synthetic domain's auto-labeling with zero new
labeling code.
"""

from verifier.domains.tools.domain import DOMAIN_NAME, build_domain, generate_problem

__all__ = ["DOMAIN_NAME", "build_domain", "generate_problem"]
