"""Policy networks and PPO training logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from ..config.schema import PolicySpec

__all__ = ["PolicyOutput", "Policy", "PPOPolicy"]


def _to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


@dataclass
class PolicyOutput:
    """Container returned by :meth:`Policy.act`."""

    action: np.ndarray
    log_prob: float
    value: float


class Policy:
    """Abstract interface implemented by RL policies."""

    def act(self, observation: np.ndarray, training: bool) -> PolicyOutput:  # pragma: no cover - interface
        raise NotImplementedError

    def store(self, record) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self) -> Tuple[float, float, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def reset_episode(self) -> None:  # pragma: no cover - interface
        pass

    def ready(self) -> bool:  # pragma: no cover - interface
        return False


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.alpha = nn.Linear(hidden_size, action_dim)
        self.beta = nn.Linear(hidden_size, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        base = self.layers(obs)
        alpha = F.softplus(self.alpha(base)) + 1.0
        beta = F.softplus(self.beta(base)) + 1.0
        return alpha, beta


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(obs).squeeze(-1)


class RolloutBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.ptr = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError("Rollout buffer overflow")
        self.obs[self.ptr] = _to_tensor(observation, self.device)
        self.actions[self.ptr] = _to_tensor(action, self.device)
        self.logprobs[self.ptr] = float(log_prob)
        self.values[self.ptr] = float(value)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.ptr += 1

    def __len__(self) -> int:
        return self.ptr

    def ready(self) -> bool:
        return self.ptr >= self.capacity

    def reset(self) -> None:
        self.ptr = 0

    def compute_returns(self) -> Tuple[torch.Tensor, torch.Tensor]:
        length = self.ptr
        returns = torch.zeros(length, dtype=torch.float32, device=self.device)
        advantages = torch.zeros(length, dtype=torch.float32, device=self.device)
        next_value = torch.tensor(0.0, device=self.device)
        next_advantage = torch.tensor(0.0, device=self.device)
        for step in reversed(range(length)):
            mask = 1.0 - self.dones[step]
            delta = self.rewards[step] + self.gamma * next_value * mask - self.values[step]
            next_advantage = delta + self.gamma * self.gae_lambda * mask * next_advantage
            advantages[step] = next_advantage
            returns[step] = advantages[step] + self.values[step]
            next_value = self.values[step]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return returns.detach(), advantages.detach()

    def batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        length = self.ptr
        indices = torch.randperm(length, device=self.device)
        for start in range(0, length, batch_size):
            yield indices[start : start + batch_size]


class PPOPolicy(Policy, nn.Module):
    """Actor-critic policy with PPO updates and Beta action distribution."""

    def __init__(self, obs_dim: int, action_dim: int, config: PolicySpec, device: str | torch.device = "cpu") -> None:
        super().__init__()
        nn.Module.__init__(self)
        self.config = config
        self.device = torch.device(device)
        self.actor = ActorNetwork(obs_dim, config.hidden_size, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
        )
        self.buffer = RolloutBuffer(
            capacity=config.rollout_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=self.device,
        )

    # Policy API -------------------------------------------------------
    def act(self, observation: np.ndarray, training: bool) -> PolicyOutput:
        obs_tensor = _to_tensor(observation, self.device)
        with torch.no_grad():
            alpha, beta = self.actor(obs_tensor.unsqueeze(0))
            dist = Beta(alpha, beta)
            if training:
                action = dist.sample().squeeze(0)
            else:
                action = (alpha / (alpha + beta)).squeeze(0)
            action = torch.clamp(action, 1e-5, 1.0 - 1e-5)
            log_prob = dist.log_prob(action).sum(-1)
            value = self.critic(obs_tensor.unsqueeze(0)).squeeze(0)
        return PolicyOutput(
            action=action.cpu().numpy().astype(np.float32),
            log_prob=float(log_prob.cpu()),
            value=float(value.cpu()),
        )

    def store(self, record) -> None:
        self.buffer.add(
            observation=record.observation,
            action=record.action,
            log_prob=record.log_prob,
            value=record.value,
            reward=record.reward,
            done=record.done,
        )

    def ready(self) -> bool:
        return self.buffer.ready()

    def update(self) -> Tuple[float, float, float]:
        returns, advantages = self.buffer.compute_returns()
        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        total_batches = 0
        for _ in range(self.config.update_epochs):
            for batch_idx in self.buffer.batches(self.config.minibatch_size):
                obs_batch = self.buffer.obs[batch_idx]
                act_batch = self.buffer.actions[batch_idx]
                old_log = self.buffer.logprobs[batch_idx]
                ret_batch = returns[batch_idx]
                adv_batch = advantages[batch_idx]

                alpha, beta = self.actor(obs_batch)
                dist = Beta(alpha, beta)
                log_prob = dist.log_prob(act_batch.clamp(1e-5, 1.0 - 1e-5)).sum(-1)
                entropy = dist.entropy().sum(-1)
                ratio = (log_prob - old_log).exp()
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef)
                policy_loss = -(torch.min(ratio * adv_batch, clipped_ratio * adv_batch)).mean()
                value_estimate = self.critic(obs_batch)
                value_loss = 0.5 * (ret_batch - value_estimate).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                policy_loss_total += float(policy_loss.detach().cpu())
                value_loss_total += float(value_loss.detach().cpu())
                entropy_total += float(entropy.mean().detach().cpu())
                total_batches += 1
        self.buffer.reset()
        if total_batches == 0:
            return 0.0, 0.0, 0.0
        return (
            policy_loss_total / total_batches,
            value_loss_total / total_batches,
            entropy_total / total_batches,
        )
