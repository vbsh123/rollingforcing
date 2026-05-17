from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Optional

import torch
from torch import Tensor


class DriftDecision(str, Enum):
    """Outcome of a TokenTrim drift check."""

    WARMUP = "warmup"
    ACCEPT = "accept"
    PRUNE_AND_REROLL = "prune_and_reroll"


@dataclass(frozen=True)
class TokenTrimConfig:
    """Configuration for inference-time drift-triggered token suppression."""

    pruning_fraction: float = 0.10
    lambda_threshold: float = 2.0
    warmup_steps: int = 2
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 < self.pruning_fraction < 1.0:
            raise ValueError("pruning_fraction must be in (0, 1)")
        if self.lambda_threshold <= 0.0:
            raise ValueError("lambda_threshold must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass
class DriftStats:
    """Running mean and population variance for accepted drift severities."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @property
    def std(self) -> float:
        if self.count == 0:
            return 0.0
        return (self.m2 / self.count) ** 0.5

    def threshold(self, lambda_threshold: float) -> float:
        return self.mean + lambda_threshold * self.std

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2


@dataclass(frozen=True)
class TokenTrimStepResult:
    """Result of evaluating one candidate autoregressive chunk."""

    decision: DriftDecision
    token_indices: Tensor
    token_mask: Tensor
    drift: Tensor
    severity: float
    threshold: Optional[float]
    stats_count: int

    @property
    def should_prune(self) -> bool:
        return self.decision == DriftDecision.PRUNE_AND_REROLL


def latent_summary(latents: Tensor, frame_dim: int = 1) -> Tensor:
    """Average frame latents into a spatial-token summary."""

    if latents.ndim < 3:
        raise ValueError("latents must have at least frame, token, and channel dimensions")
    return latents.mean(dim=frame_dim)


def compute_drift(previous_summary: Tensor, current_summary: Tensor) -> Tensor:
    """Compute per-token L2 drift between consecutive latent summaries."""

    if previous_summary.shape != current_summary.shape:
        raise ValueError(
            "previous_summary and current_summary must have identical shapes; "
            f"got {tuple(previous_summary.shape)} and {tuple(current_summary.shape)}"
        )
    if previous_summary.ndim < 2:
        raise ValueError("summaries must end with [tokens, channels]")
    return torch.linalg.vector_norm(current_summary - previous_summary, ord=2, dim=-1)


def select_unstable_tokens(drift: Tensor, pruning_fraction: float) -> tuple[Tensor, Tensor, float]:
    """Select the top-p drifting spatial tokens."""

    if drift.ndim not in (1, 2):
        raise ValueError("drift must have shape [tokens] or [batch, tokens]")

    token_count = drift.shape[-1]
    top_k = max(1, ceil(pruning_fraction * token_count))
    _, indices = torch.topk(drift, k=top_k, dim=-1, largest=True, sorted=False)

    mask = torch.ones_like(drift, dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=False)

    severity_tensor = drift.gather(dim=-1, index=indices).mean()
    return indices, mask, float(severity_tensor.detach().cpu().item())


@dataclass
class TokenTrimState:
    """Stateful evaluator for drift-triggered pruning decisions."""

    config: TokenTrimConfig = field(default_factory=TokenTrimConfig)
    stats: DriftStats = field(default_factory=DriftStats)

    def evaluate_summaries(self, previous_summary: Tensor, current_summary: Tensor) -> TokenTrimStepResult:
        drift = compute_drift(previous_summary, current_summary)
        token_indices, token_mask, severity = select_unstable_tokens(
            drift=drift,
            pruning_fraction=self.config.pruning_fraction,
        )

        threshold = None
        if self.stats.count < self.config.warmup_steps:
            decision = DriftDecision.WARMUP
        else:
            threshold = self.stats.threshold(self.config.lambda_threshold)
            decision = (
                DriftDecision.PRUNE_AND_REROLL
                if severity > threshold + self.config.eps
                else DriftDecision.ACCEPT
            )

        return TokenTrimStepResult(
            decision=decision,
            token_indices=token_indices,
            token_mask=token_mask,
            drift=drift,
            severity=severity,
            threshold=threshold,
            stats_count=self.stats.count,
        )

    def evaluate_latents(
        self,
        previous_latents: Tensor,
        current_latents: Tensor,
        frame_dim: int = 1,
    ) -> TokenTrimStepResult:
        previous_summary = latent_summary(previous_latents, frame_dim=frame_dim)
        current_summary = latent_summary(current_latents, frame_dim=frame_dim)
        return self.evaluate_summaries(previous_summary, current_summary)

    def accept(self, result: TokenTrimStepResult) -> None:
        """Record the severity of a finalized chunk."""

        self.stats.update(result.severity)
