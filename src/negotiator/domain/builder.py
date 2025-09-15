"""Domain and utility construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun

from ..config.schema import DomainSpec, IssueSpec, RoleSpec, ValueFunctionSpec

__all__ = ["DomainFactory", "build_issue", "build_utility"]


def build_issue(spec: IssueSpec):
    """Create a NegMAS issue from a configuration specification."""

    if spec.values is not None:
        values: Iterable[int] = spec.values
    elif spec.minimum is not None and spec.maximum is not None:
        if spec.step != 1:
            values = list(range(spec.minimum, spec.maximum + spec.step, spec.step))
        else:
            values = (spec.minimum, spec.maximum)
    else:
        raise ValueError(f"Issue {spec.name} must define either values or min/max range")
    return make_issue(name=spec.name, values=values)


def _build_value_function(spec: ValueFunctionSpec):
    kind = spec.kind.lower()
    if kind in {"identity", "id", "linear_identity"}:
        if spec.weight == 1.0 and spec.bias == 0.0:
            return IdentityFun()
        return AffineFun(spec.weight, bias=spec.bias)
    if kind in {"linear", "slope"}:
        return LinearFun(spec.weight)
    if kind in {"affine", "bias"}:
        return AffineFun(spec.weight, bias=spec.bias)
    raise ValueError(f"Unsupported value function kind: {spec.kind}")


def build_utility(role: RoleSpec, outcome_space) -> LinearAdditiveUtilityFunction:
    """Create an additive utility function for the given role."""

    values: Dict[str, object] = {}
    for issue in outcome_space.issues:
        spec = role.utility.get(issue.name)
        if spec is None:
            values[issue.name] = IdentityFun()
        else:
            values[issue.name] = _build_value_function(spec)
    return LinearAdditiveUtilityFunction(values=values, outcome_space=outcome_space)


@dataclass
class DomainFactory:
    """Utility class for instantiating negotiation sessions."""

    spec: DomainSpec

    def make_mechanism(self) -> SAOMechanism:
        issues = [build_issue(issue) for issue in self.spec.issues]
        return SAOMechanism(issues=issues, n_steps=self.spec.n_steps, time_limit=self.spec.time_limit)

    def build_utilities(self, mechanism: SAOMechanism, roles: Dict[str, RoleSpec]) -> Dict[str, LinearAdditiveUtilityFunction]:
        return {role: build_utility(role_spec, mechanism.outcome_space) for role, role_spec in roles.items()}
