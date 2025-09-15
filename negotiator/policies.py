"""Lightweight policy controller implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

import numpy as np

from .config import TrainingConfig


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class PolicyAction:
    mode: str
    target: float | None = None
    candidate: int | None = None
    accept: int | None = None


@dataclass
class StepRecord:
    mode: str
    obs: np.ndarray
    target_probs: np.ndarray | None
    target_index: int | None
    candidate_probs: np.ndarray | None
    candidate_index: int | None
    accept_prob: float | None
    accept_action: int | None
    reward: float = 0.0


class PolicyController:
    """Maintains linear-softmax policies and learns with REINFORCE."""

    def __init__(self, obs_dim: int, k_candidates: int, cfg: TrainingConfig):
        self.obs_dim = obs_dim
        self.k_candidates = k_candidates
        self.cfg = cfg
        self.n_bins = 11
        rng = np.random.default_rng(cfg.seed)
        self.target_weights = rng.normal(scale=0.1, size=(self.n_bins, obs_dim))
        self.candidate_weights = rng.normal(scale=0.1, size=(k_candidates, obs_dim))
        self.accept_weights = rng.normal(scale=0.1, size=obs_dim)
        self.records: List[StepRecord] = []
        self.baseline = 0.0

    def to(self, device):  # compatibility shim for agent API
        return None

    def start_episode(self) -> None:
        self.records = []

    def act(self, obs: np.ndarray, mode: str) -> Tuple[PolicyAction, int]:
        if mode == "propose":
            target_logits = self.target_weights @ obs
            target_probs = softmax(target_logits)
            target_index = int(np.random.choice(self.n_bins, p=target_probs))
            candidate_logits = self.candidate_weights @ obs
            candidate_probs = softmax(candidate_logits)
            candidate_index = int(np.random.choice(self.k_candidates, p=candidate_probs))
            rel = target_index / (self.n_bins - 1)
            action = PolicyAction(mode=mode, target=rel, candidate=candidate_index)
            record = StepRecord(
                mode=mode,
                obs=obs.copy(),
                target_probs=target_probs,
                target_index=target_index,
                candidate_probs=candidate_probs,
                candidate_index=candidate_index,
                accept_prob=None,
                accept_action=None,
            )
        elif mode == "respond":
            logit = float(self.accept_weights @ obs)
            prob = sigmoid(logit)
            action_value = int(np.random.binomial(1, prob))
            action = PolicyAction(mode=mode, accept=action_value)
            record = StepRecord(
                mode=mode,
                obs=obs.copy(),
                target_probs=None,
                target_index=None,
                candidate_probs=None,
                candidate_index=None,
                accept_prob=prob,
                accept_action=action_value,
            )
        else:
            raise ValueError(f"Unknown mode {mode}")
        index = len(self.records)
        self.records.append(record)
        return action, index

    def add_reward(self, index: int, reward: float) -> None:
        if 0 <= index < len(self.records):
            self.records[index].reward += float(reward)

    def finish_episode(self) -> dict:
        if not self.records:
            return {"loss": 0.0, "return": 0.0, "steps": 0}
        gamma = self.cfg.gamma
        returns: List[float] = []
        g = 0.0
        for record in reversed(self.records):
            g = record.reward + gamma * g
            returns.insert(0, g)
        returns = np.array(returns, dtype=float)
        baseline = self.baseline
        advantages = returns - baseline
        lr = self.cfg.learning_rate

        entropy_total = 0.0
        for record, adv in zip(self.records, advantages):
            obs = record.obs
            if record.mode == "propose":
                # Target gradients
                target_probs = record.target_probs
                target_index = record.target_index
                if target_probs is not None and target_index is not None:
                    one_hot = np.zeros_like(target_probs)
                    one_hot[target_index] = 1.0
                    grad = np.outer(one_hot - target_probs, obs)
                    self.target_weights += lr * adv * grad
                    entropy_total -= float(np.sum(target_probs * np.log(np.clip(target_probs, 1e-8, 1.0))))
                candidate_probs = record.candidate_probs
                candidate_index = record.candidate_index
                if candidate_probs is not None and candidate_index is not None:
                    one_hot = np.zeros_like(candidate_probs)
                    one_hot[candidate_index] = 1.0
                    grad = np.outer(one_hot - candidate_probs, obs)
                    self.candidate_weights += lr * adv * grad
                    entropy_total -= float(np.sum(candidate_probs * np.log(np.clip(candidate_probs, 1e-8, 1.0))))
            else:
                prob = record.accept_prob or 0.5
                action = record.accept_action or 0
                grad = (action - prob) * obs
                self.accept_weights += lr * adv * grad
                entropy = -(prob * math.log(max(prob, 1e-8)) + (1 - prob) * math.log(max(1 - prob, 1e-8)))
                entropy_total += entropy

        self.baseline = 0.9 * baseline + 0.1 * returns[0]
        episode_return = float(returns[0])
        return {
            "return": episode_return,
            "steps": len(self.records),
            "entropy": entropy_total,
        }
