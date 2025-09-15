"""Configuration schema for the negotiation simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

__all__ = [
    "IssueSpec",
    "ValueFunctionSpec",
    "RoleSpec",
    "DomainSpec",
    "OpponentSpec",
    "PolicySpec",
    "TrainingSpec",
    "WealthSpec",
    "EconomySpec",
    "NegotiationConfig",
    "load_config",
]


@dataclass(slots=True)
class IssueSpec:
    """Specification of a single negotiation issue."""

    name: str
    kind: str = "discrete"
    values: Optional[List[int]] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    step: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IssueSpec":
        return cls(
            name=data["name"],
            kind=data.get("kind", "discrete"),
            values=list(data["values"]) if "values" in data else None,
            minimum=data.get("min"),
            maximum=data.get("max"),
            step=int(data.get("step", 1)),
        )


@dataclass(slots=True)
class ValueFunctionSpec:
    """Configuration of a value function for a single issue."""

    kind: str
    weight: float = 1.0
    bias: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValueFunctionSpec":
        return cls(
            kind=data.get("kind", "identity"),
            weight=float(data.get("weight", data.get("slope", 1.0))),
            bias=float(data.get("bias", data.get("intercept", data.get("offset", 0.0)))),
        )


@dataclass(slots=True)
class RoleSpec:
    """Definition of a role (buyer, seller, etc.)."""

    name: str
    utility: Dict[str, ValueFunctionSpec]

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, Any]) -> "RoleSpec":
        utility_cfg = {
            issue: ValueFunctionSpec.from_dict(v)
            for issue, v in (data.get("utility") or {}).items()
        }
        return cls(name=name, utility=utility_cfg)


@dataclass(slots=True)
class DomainSpec:
    """Domain description used to instantiate NegMAS mechanisms."""

    issues: List[IssueSpec]
    n_steps: int
    time_limit: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainSpec":
        return cls(
            issues=[IssueSpec.from_dict(i) for i in data.get("issues", [])],
            n_steps=int(data.get("n_steps", 40)),
            time_limit=data.get("time_limit"),
        )


@dataclass(slots=True)
class OpponentSpec:
    """Description of an opponent negotiator."""

    role: str
    kind: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OpponentSpec":
        return cls(
            role=data["role"],
            kind=data["type"],
            weight=float(data.get("weight", 1.0)),
            params=dict(data.get("params", {})),
        )


@dataclass(slots=True)
class PolicySpec:
    """Hyper-parameters of the reinforcement learning policy."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 128
    rollout_size: int = 512
    hidden_size: int = 128

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicySpec":
        kwargs = {k: data.get(k, getattr(cls, k)) for k in cls.__dataclass_fields__}
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(slots=True)
class TrainingSpec:
    """Training run configuration."""

    total_episodes: int = 1000
    eval_interval: int = 100
    log_interval: int = 10

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingSpec":
        kwargs = {k: data.get(k, getattr(cls, k)) for k in cls.__dataclass_fields__}
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(slots=True)
class WealthSpec:
    """Parameters controlling the wealth process and ergodic rewards."""

    initial: float = 1.0
    time_cost: float = 0.01
    ruin_threshold: float = 1e-3
    ruin_penalty: float = -50.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WealthSpec":
        kwargs = {k: data.get(k, getattr(cls, k)) for k in cls.__dataclass_fields__}
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(slots=True)
class EconomySpec:
    """Parameters describing the mapping from outcomes to cash-flow."""

    value_per_unit: float = 0.3
    delivery_bonus: float = 0.05
    delivery_reference: int = 10
    seller_cost: float = 0.1
    seller_delivery_cost: float = 0.02

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EconomySpec":
        kwargs = {k: data.get(k, getattr(cls, k)) for k in cls.__dataclass_fields__}
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(slots=True)
class NegotiationConfig:
    """Top-level configuration bundle."""

    domain: DomainSpec
    roles: Dict[str, RoleSpec]
    opponents: List[OpponentSpec]
    policy: PolicySpec
    training: TrainingSpec
    wealth: Dict[str, WealthSpec]
    economy: EconomySpec

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NegotiationConfig":
        domain = DomainSpec.from_dict(data["domain"])
        roles = {name: RoleSpec.from_dict(name, cfg) for name, cfg in data.get("roles", {}).items()}
        opponents = [OpponentSpec.from_dict(op) for op in data.get("opponents", [])]
        policy = PolicySpec.from_dict(data.get("policy", {}))
        training = TrainingSpec.from_dict(data.get("training", {}))
        wealth = {name: WealthSpec.from_dict(cfg) for name, cfg in data.get("wealth", {}).items()}
        economy = EconomySpec.from_dict(data.get("economy", {}))
        return cls(
            domain=domain,
            roles=roles,
            opponents=opponents,
            policy=policy,
            training=training,
            wealth=wealth,
            economy=economy,
        )


def load_config(path: str | Path | Mapping[str, Any]) -> NegotiationConfig:
    """Load the negotiation configuration from YAML or dictionary."""

    if isinstance(path, Mapping):
        return NegotiationConfig.from_dict(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise TypeError("Configuration file must contain a YAML mapping")
    return NegotiationConfig.from_dict(data)
