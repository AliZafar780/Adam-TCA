"""
Adam-TCA: Curvature-Aware Adam with Gradient Cosine Similarity Modulation.

A PyTorch optimizer that extends Adam by dynamically modulating the learning
rate based on the cosine similarity between the current gradient and the
running momentum estimate. This geometric signal serves as a proxy for local
curvature, accelerating convergence in low-curvature regions and improving
stability in high-curvature regions.

Features:
  - Curvature-aware LR modulation (cosine similarity + optional variance)
  - Gradient clipping by norm and value
  - Linear learning rate warmup
  - Skippable NaN/Inf parameter steps
  - Per-parameter curvature introspection
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer


__all__ = ["AdamTCA"]


class AdamTCA(Optimizer):
    """Adam-TCA: Curvature-Aware Adam with cosine-similarity LR modulation.

    This optimizer extends the Adam algorithm by incorporating a curvature-aware
    learning rate adjustment mechanism. It tracks the cosine similarity between
    the current gradient and the running momentum estimate, and optionally the
    gradient variance over a history window, to modulate the effective learning
    rate for each parameter.

    When the gradient and momentum are highly aligned (cosine similarity near 1),
    the optimizer increases the learning rate, exploiting consistent gradient
    directions. When they are misaligned (similarity near 0 or negative), the
    learning rate is reduced, indicating a region of high curvature or a
    changing gradient landscape.

    Three curvature modes are supported:
      - ``'cosine'`` (default): modulation based on cosine similarity only.
      - ``'variance'``: modulation based on gradient variance (needs
        ``curvature_window``).
      - ``'hybrid'``: combines cosine similarity and variance signals.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter
            groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient and
            its square (default: (0.9, 0.999)).
        eps: Term added to denominators to improve numerical stability
            (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) (default: 0).
        curvature_window: Number of recent gradient vectors to retain for
            curvature estimation (default: 100). Only used when
            ``curvature_mode`` is ``'variance'`` or ``'hybrid'``.
        curvature_mode: Curvature estimation mode. One of ``'cosine'``,
            ``'variance'``, or ``'hybrid'`` (default: ``'cosine'``).
        grad_clip_norm: Maximum gradient norm for clipping (0 = disabled)
            (default: 0).
        grad_clip_value: Maximum gradient value for clipping (0 = disabled)
            (default: 0).
        warmup_steps: Number of linear warmup steps (0 = disabled)
            (default: 0).
        skip_nan: If ``True``, skip the step entirely for parameters whose
            gradients contain NaN/Inf instead of zeroing them out
            (default: ``False``).

    .. note::
        The curvature modulation factor :math:`\\alpha_t` is computed as:

        .. math::
            \\alpha_t = \\frac{1 + \\cos(\\theta_t)}{2}

        where :math:`\\cos(\\theta_t)` is the cosine similarity between the
        current gradient :math:`g_t` and the running momentum estimate
        :math:`m_t`. The effective learning rate becomes:

        .. math::
            \\eta_t^{\\text{eff}} = \\eta_t^{\\text{base}} \\cdot \\alpha_t
    """

    def __init__(
        self,
        params: Iterable[Union[torch.Tensor, Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        curvature_window: int = 100,
        curvature_mode: str = "cosine",
        grad_clip_norm: float = 0,
        grad_clip_value: float = 0,
        warmup_steps: int = 0,
        skip_nan: bool = False,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(curvature_window, int) or curvature_window < 1:
            raise ValueError(
                f"curvature_window must be a positive integer, got {curvature_window}"
            )
        if curvature_mode not in ("cosine", "variance", "hybrid"):
            raise ValueError(
                f"curvature_mode must be 'cosine', 'variance', or 'hybrid', "
                f"got {curvature_mode!r}"
            )
        if not 0.0 <= grad_clip_norm:
            raise ValueError(f"Invalid grad_clip_norm: {grad_clip_norm}")
        if not 0.0 <= grad_clip_value:
            raise ValueError(f"Invalid grad_clip_value: {grad_clip_value}")
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be a non-negative integer, got {warmup_steps}"
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "curvature_window": curvature_window,
            "curvature_mode": curvature_mode,
            "grad_clip_norm": grad_clip_norm,
            "grad_clip_value": grad_clip_value,
            "warmup_steps": warmup_steps,
            "skip_nan": skip_nan,
        }
        super().__init__(params, defaults)

    def __repr__(self) -> str:
        d = self.defaults
        return (
            f"AdamTCA(lr={d['lr']}, betas={d['betas']}, eps={d['eps']}, "
            f"weight_decay={d['weight_decay']}, "
            f"curvature_window={d['curvature_window']}, "
            f"curvature_mode={d['curvature_mode']!r}, "
            f"grad_clip_norm={d['grad_clip_norm']}, "
            f"grad_clip_value={d['grad_clip_value']}, "
            f"warmup_steps={d['warmup_steps']}, "
            f"skip_nan={d['skip_nan']})"
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_learning_rate(self, group_index: int = 0) -> float:
        """Return the current effective learning rate for a parameter group,
        accounting for warmup and curvature modulation (if available).

        Args:
            group_index: Index of the parameter group (default: 0).

        Returns:
            The effective learning rate as a float.
        """
        group = self.param_groups[group_index]
        base_lr = group["lr"]
        warmup_steps = group["warmup_steps"]
        # Estimate warmup factor (use global step from first param if available)
        step_t = 1
        for p in group["params"]:
            if p in self.state and self.state[p]["step"] > 0:
                step_t = self.state[p]["step"]
                break
        if warmup_steps > 0 and step_t < warmup_steps:
            warmup_factor = step_t / warmup_steps
        else:
            warmup_factor = 1.0
        return base_lr * warmup_factor

    def get_curvature(self, param: torch.Tensor) -> Dict[str, float]:
        """Get curvature information for a specific parameter tensor.

        Args:
            param: The parameter tensor to query.

        Returns:
            A dictionary containing:
                - ``'cosine_similarity'``: Current cosine similarity value.
                - ``'curvature_alpha'``: The modulation factor alpha.
                - ``'gradient_norm'``: L2 norm of the gradient.
                - ``'momentum_norm'``: L2 norm of the momentum estimate.
                - ``'history_length'``: Number of gradients in the curvature
                  window.
                - ``'gradient_variance'``: Variance of gradients in history
                  (0 if mode is ``'cosine'``).
        """
        state = self.state[param]
        if len(state) == 0:
            return {
                "cosine_similarity": 0.0,
                "curvature_alpha": 0.5,
                "gradient_norm": 0.0,
                "momentum_norm": 0.0,
                "history_length": 0,
                "gradient_variance": 0.0,
            }

        exp_avg = state["exp_avg"]
        grad = param.grad
        grad_history = state.get("grad_history", [])

        if grad is None:
            return {
                "cosine_similarity": 0.0,
                "curvature_alpha": 0.5,
                "gradient_norm": 0.0,
                "momentum_norm": exp_avg.norm(2).item(),
                "history_length": len(grad_history),
                "gradient_variance": self._compute_grad_variance(grad_history)
                if grad_history else 0.0,
            }

        grad_flat = grad.view(-1)
        exp_avg_flat = exp_avg.view(-1)
        grad_norm = grad_flat.norm(2)
        exp_avg_norm = exp_avg_flat.norm(2)

        if grad_norm.item() == 0 or exp_avg_norm.item() == 0:
            cos_sim = 0.0
        else:
            cos_sim = torch.dot(grad_flat, exp_avg_flat) / (grad_norm * exp_avg_norm)
            cos_sim = cos_sim.clamp(-1.0, 1.0).item()

        alpha = (1.0 + cos_sim) / 2.0

        return {
            "cosine_similarity": cos_sim,
            "curvature_alpha": alpha,
            "gradient_norm": grad_norm.item(),
            "momentum_norm": exp_avg_norm.item(),
            "history_length": len(grad_history),
            "gradient_variance": self._compute_grad_variance(grad_history)
            if grad_history else 0.0,
        }

    def get_global_curvature(self) -> Dict[str, float]:
        """Get average curvature information across all parameters.

        Returns:
            A dictionary containing the mean curvature statistics across
            all parameter groups.
        """
        stats: Dict[str, List[float]] = {
            "cosine_similarity": [],
            "curvature_alpha": [],
            "gradient_norm": [],
            "momentum_norm": [],
            "history_length": [],
            "gradient_variance": [],
        }
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.state:
                    info = self.get_curvature(p)
                    for k in stats:
                        stats[k].append(info[k])

        if not stats["cosine_similarity"]:
            return {k: 0.0 for k in stats}

        return {
            k: float(torch.tensor(v, dtype=torch.float32).mean())
            for k, v in stats.items()
        }

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The computed loss if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            curvature_window = group["curvature_window"]
            curvature_mode = group["curvature_mode"]
            grad_clip_norm = group["grad_clip_norm"]
            grad_clip_value = group["grad_clip_value"]
            warmup_steps = group["warmup_steps"]
            skip_nan = group["skip_nan"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # ------------------------------------------------------------------
                # Gradient clipping
                # ------------------------------------------------------------------
                if not grad.is_sparse:
                    if grad_clip_norm > 0:
                        grad_norm_clip = grad.norm(2)
                        if grad_norm_clip > grad_clip_norm:
                            grad.mul_(grad_clip_norm / grad_norm_clip)

                    if grad_clip_value > 0:
                        grad.clamp_(-grad_clip_value, grad_clip_value)

                # ------------------------------------------------------------------
                # NaN / Inf handling
                # ------------------------------------------------------------------
                if not grad.is_sparse:
                    has_naninf = torch.isnan(grad).any() or torch.isinf(grad).any()
                    if has_naninf:
                        if skip_nan:
                            # Skip this parameter entirely
                            continue
                        else:
                            # Zero out NaN/Inf to prevent crashing
                            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                            p.grad.copy_(grad)

                state = self.state[p]

                # ------------------------------------------------------------------
                # State initialization
                # ------------------------------------------------------------------
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if curvature_mode in ("variance", "hybrid"):
                        state["grad_history"] = []

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                # ------------------------------------------------------------------
                # Learning rate warmup
                # ------------------------------------------------------------------
                if warmup_steps > 0 and step_t < warmup_steps:
                    warmup_factor = step_t / warmup_steps
                else:
                    warmup_factor = 1.0

                # ------------------------------------------------------------------
                # Weight decay
                # ------------------------------------------------------------------
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # ------------------------------------------------------------------
                # Momentum updates
                # ------------------------------------------------------------------
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t

                step_size = lr / bias_correction1

                # ------------------------------------------------------------------
                # Curvature-aware modulation
                # ------------------------------------------------------------------
                grad_flat = grad.view(-1)
                exp_avg_flat = exp_avg.view(-1)
                grad_norm_val = grad_flat.norm(2)
                exp_avg_norm_val = exp_avg_flat.norm(2)

                if grad_norm_val.item() == 0 or exp_avg_norm_val.item() == 0:
                    cos_sim = 0.0
                else:
                    cos_sim = torch.dot(grad_flat, exp_avg_flat) / (grad_norm_val * exp_avg_norm_val)
                    cos_sim = cos_sim.clamp(-1.0, 1.0).item()

                # Base alpha: maps [-1, 1] -> [0, 1]
                alpha = (1.0 + cos_sim) / 2.0

                # Variance-based curvature signal (if applicable)
                if curvature_mode in ("variance", "hybrid"):
                    grad_history: List[torch.Tensor] = state["grad_history"]
                    # Store flattened grad on the same device (no CPU transfer)
                    grad_history.append(grad.detach().clone().flatten())
                    if len(grad_history) > curvature_window:
                        grad_history.pop(0)

                    if len(grad_history) >= 2:
                        # Compute normalized variance across history
                        stacked = torch.stack(grad_history)  # (N, D)
                        var_est = stacked.var(dim=0).mean().item()  # scalar
                        # Normalize variance to [0, 1] via sigmoid-like scaling
                        var_signal = 1.0 - (var_est / (var_est + 1.0 + eps))
                    else:
                        var_signal = 1.0  # neutral when not enough history

                    if curvature_mode == "hybrid":
                        # Average cosine and variance signals
                        alpha = 0.5 * alpha + 0.5 * var_signal
                    else:
                        alpha = var_signal

                # Apply curvature modulation: full [0, 1] range
                modulated_step_size = step_size * alpha

                # Apply warmup to the modulated step size
                modulated_step_size *= warmup_factor

                # ------------------------------------------------------------------
                # Parameter update
                # ------------------------------------------------------------------
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-modulated_step_size)

        return loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_grad_variance(
        grad_history: List[torch.Tensor],
    ) -> float:
        """Compute normalized variance from a list of stored gradient vectors."""
        if len(grad_history) < 2:
            return 0.0
        try:
            stacked = torch.stack(grad_history)
            var_est = stacked.var(dim=0).mean().item()
            return float(var_est)
        except Exception:
            return 0.0
