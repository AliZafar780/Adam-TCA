"""
Adam-TCA: Taylor-Centric Adam with Curvature-Aware Learning Rate Modulation.

A novel PyTorch optimizer that applies concepts from differential geometry
to dynamically modulate the learning rate based on the cosine similarity
between the current gradient and the historical momentum vector.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer


__all__ = ["AdamTCA"]


class AdamTCA(Optimizer):
    """Adam-TCA: Taylor-Centric Adam with curvature-aware learning rate modulation.

    This optimizer extends the Adam algorithm by incorporating a curvature-aware
    learning rate adjustment mechanism. It tracks the cosine similarity between
    the current gradient and the running momentum estimate, and uses this
    geometric signal to modulate the effective learning rate for each parameter.

    When the gradient and momentum are highly aligned (cosine similarity near 1),
    the optimizer increases the learning rate, exploiting consistent gradient
    directions. When they are misaligned (similarity near 0 or negative), the
    learning rate is reduced, indicating a region of high curvature or a
    changing gradient landscape.

    The curvature window controls how many recent gradient vectors are retained
    for computing the running statistics used in the modulation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages of gradient and
            its square (default: (0.9, 0.999)).
        eps: Term added to denominators to improve numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) (default: 0).
        curvature_window: Number of recent gradient vectors to use for
            curvature estimation (default: 100).

    .. note::
        The curvature modulation factor :math:`\\alpha_t` is computed as:

        .. math::
            \\alpha_t = \\frac{1 + \\cos(\\theta_t)}{2}

        where :math:`\\cos(\\theta_t)` is the cosine similarity between the
        current gradient :math:`g_t` and the running momentum estimate
        :math:`m_t`. The effective learning rate becomes:

        .. math::
            \\eta_t^{\\text{eff}} = \\eta \\cdot \\alpha_t
    """

    def __init__(
        self,
        params: Iterable[Union[torch.Tensor, Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        curvature_window: int = 100,
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

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "curvature_window": curvature_window,
        }
        super().__init__(params, defaults)

    def __repr__(self) -> str:
        return (
            f"AdamTCA(lr={self.defaults['lr']}, "
            f"betas={self.defaults['betas']}, "
            f"eps={self.defaults['eps']}, "
            f"weight_decay={self.defaults['weight_decay']}, "
            f"curvature_window={self.defaults['curvature_window']})"
        )

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

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if not grad.is_sparse:
                    # Handle NaN/Inf gradients
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        # Replace NaN/Inf with zero to prevent crashing
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                        p.grad.copy_(grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # Gradient history for curvature estimation
                    state["grad_history"] = []

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Decay the first and second moment running average coefficients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t

                # Compute step size
                step_size = lr / bias_correction1

                # Curvature-aware modulation using cosine similarity
                # between current gradient and momentum estimate
                grad_flat = grad.view(-1)
                exp_avg_flat = exp_avg.view(-1)

                grad_norm = grad_flat.norm(2)
                exp_avg_norm = exp_avg_flat.norm(2)

                # Compute cosine similarity
                # Handle edge case where either norm is zero
                if grad_norm.item() == 0 or exp_avg_norm.item() == 0:
                    cos_sim = 0.0
                else:
                    cos_sim = torch.dot(grad_flat, exp_avg_flat) / (grad_norm * exp_avg_norm)
                    cos_sim = cos_sim.clamp(-1.0, 1.0).item()

                # Modulation factor: maps [-1, 1] -> [0, 1]
                # When gradient aligns with momentum (cos~1), alpha~1 (full speed)
                # When gradient opposes momentum (cos~-1), alpha~0 (brake)
                alpha = (1.0 + cos_sim) / 2.0

                # Apply curvature modulation to step size
                modulated_step_size = step_size * (0.5 + 0.5 * alpha)

                # Update gradient history for curvature tracking
                grad_history: List[torch.Tensor] = state["grad_history"]
                grad_history.append(grad.detach().clone().cpu())
                if len(grad_history) > curvature_window:
                    grad_history.pop(0)

                # Compute the denom
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-modulated_step_size)

        return loss

    def get_curvature(self, param: torch.Tensor) -> Dict[str, float]:
        """Get curvature information for a specific parameter tensor.

        Args:
            param: The parameter tensor to query.

        Returns:
            A dictionary containing:
                - 'cosine_similarity': The current cosine similarity value.
                - 'curvature_alpha': The modulation factor alpha.
                - 'gradient_norm': The L2 norm of the gradient.
                - 'momentum_norm': The L2 norm of the momentum estimate.
                - 'history_length': Number of gradients in the curvature window.
        """
        state = self.state[param]
        if len(state) == 0:
            return {
                "cosine_similarity": 0.0,
                "curvature_alpha": 0.5,
                "gradient_norm": 0.0,
                "momentum_norm": 0.0,
                "history_length": 0,
            }

        exp_avg = state["exp_avg"]
        grad = param.grad

        if grad is None:
            return {
                "cosine_similarity": 0.0,
                "curvature_alpha": 0.5,
                "gradient_norm": 0.0,
                "momentum_norm": exp_avg.norm(2).item(),
                "history_length": len(state.get("grad_history", [])),
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
            "history_length": len(state.get("grad_history", [])),
        }

    def get_global_curvature(self) -> Dict[str, float]:
        """Get average curvature information across all parameters.

        Returns:
            A dictionary containing the mean curvature statistics across
            all parameter groups.
        """
        stats = {
            "cosine_similarity": [],
            "curvature_alpha": [],
            "gradient_norm": [],
            "momentum_norm": [],
            "history_length": [],
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
