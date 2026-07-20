"""Named numeric resource dimensions (fuel, budget, crane energy, ...)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, model_validator


class ResourceDimension(BaseModel):
    """A resource tracked as a running sum across a plan.

    ``initial`` is the starting amount (set on the Domain, since the spec
    treats caps/initial values as domain-level). The value must never drop
    below ``floor`` or rise above ``cap`` at any prefix of a plan.
    """

    name: str
    initial: float
    cap: Optional[float] = None
    floor: float = 0.0

    @model_validator(mode="after")
    def _check_bounds(self) -> "ResourceDimension":
        if self.cap is not None and self.initial > self.cap:
            raise ValueError(f"resource '{self.name}' initial {self.initial} exceeds cap {self.cap}")
        if self.initial < self.floor:
            raise ValueError(f"resource '{self.name}' initial {self.initial} below floor {self.floor}")
        return self