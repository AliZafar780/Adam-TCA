"""
Adam-TCA Comprehensive Example Suite.

Demonstrates:
  1. Simple quadratic function optimization
  2. Linear model training with convergence tracking
  3. Three curvature modes compared (cosine, variance, hybrid)
  4. Advanced features: gradient clipping, warmup, skip_nan
  5. Curvature introspection via get_curvature() / get_global_curvature()
  6. Parameter group configuration
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adam_tca import AdamTCA


# ===================================================================
# 1. Simple quadratic function optimization
# ===================================================================

def demo_quadratic() -> None:
    """Optimize a simple quadratic function f(x) = (x - 3)^2 + 5.

    This is the simplest test: one scalar parameter, convex loss.
    """
    print("=" * 65)
    print("DEMO 1: Quadratic function optimization")
    print("=" * 65)

    x = torch.tensor([0.0], requires_grad=True)
    optimizer = AdamTCA([x], lr=0.5)

    print(f"{'Step':<6} {'x':<10} {'f(x)':<12} {'Curvature a':<12}")
    print("-" * 42)

    for step in range(1, 31):
        optimizer.zero_grad()
        loss = (x - 3.0) ** 2 + 5.0  # minimum at x=3, f(x)=5
        loss.backward()
        optimizer.step()

        if step <= 5 or step % 5 == 0:
            info = optimizer.get_curvature(x)
            alpha = info["curvature_alpha"]
            print(f"{step:<6} {x.item():<10.6f} {loss.item():<12.6f} {alpha:<12.4f}")

    print(f"\nFinal x = {x.item():.6f} (target: 3.0)")
    print(f"Final f(x) = {((x - 3.0)**2 + 5.0).item():.6f} (target: 5.0)\n")


# ===================================================================
# 2. Linear model with convergence tracking
# ===================================================================

def demo_linear_model() -> None:
    """Train a linear model on synthetic data with all three curvature modes."""
    print("=" * 65)
    print("DEMO 2: Linear regression - comparing curvature modes")
    print("=" * 65)

    torch.manual_seed(42)
    n_samples = 200
    n_features = 5

    X = torch.randn(n_samples, n_features)
    true_w = torch.tensor([[1.5], [-2.0], [0.0], [3.0], [-1.0]])
    y = X @ true_w + 0.1 * torch.randn(n_samples, 1)

    modes = ["cosine", "variance", "hybrid"]
    results: Dict[str, List[float]] = {}

    for mode in modes:
        torch.manual_seed(42)
        model = nn.Linear(n_features, 1)
        optimizer = AdamTCA(
            model.parameters(),
            lr=0.01,
            curvature_mode=mode,
            curvature_window=20,
        )
        loss_fn = nn.MSELoss()

        losses: List[float] = []
        alphas: List[float] = []

        for _ in range(100):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            info = optimizer.get_global_curvature()
            alphas.append(info["curvature_alpha"])

        results[mode] = losses

        print(f"\n  Mode: {mode!r}")
        print(f"    Initial loss: {losses[0]:.6f}")
        print(f"    Final loss:   {losses[-1]:.6f}")
        print(f"    Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
        print(f"    Alpha range:   {min(alphas):.4f} - {max(alphas):.4f}")

    print()


# ===================================================================
# 3. CNN on synthetic image data
# ===================================================================

def demo_cnn() -> None:
    """Train a small CNN on synthetic image-like data (simulating MNIST)."""
    print("=" * 65)
    print("DEMO 3: Small CNN on synthetic image data")
    print("=" * 65)

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 10),
    )

    optimizer = AdamTCA(
        model.parameters(),
        lr=0.001,
        curvature_mode="hybrid",
        curvature_window=30,
        grad_clip_norm=1.0,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Generate synthetic 28x28 "images"
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))

    print(f"{'Epoch':<8} {'Loss':<14} {'Curvature a':<14} {'Grad Norm':<14}")
    print("-" * 52)

    for epoch in range(1, 21):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            info = optimizer.get_global_curvature()
            print(
                f"{epoch:<8} {loss.item():<14.6f} "
                f"{info['curvature_alpha']:<14.4f} "
                f"{info['gradient_norm']:<14.4f}"
            )

    print()


# ===================================================================
# 4. Gradient clipping and warmup demo
# ===================================================================

def demo_advanced_features() -> None:
    """Demonstrate gradient clipping, warmup, and skip_nan features."""
    print("=" * 65)
    print("DEMO 4: Advanced features - warmup + gradient clipping")
    print("=" * 65)

    torch.manual_seed(42)
    model = nn.Linear(10, 1)
    optimizer = AdamTCA(
        model.parameters(),
        lr=0.01,
        warmup_steps=20,
        grad_clip_norm=0.5,
        grad_clip_value=2.0,
    )
    loss_fn = nn.MSELoss()

    X = torch.randn(50, 10)
    y = torch.randn(50, 1)

    print(f"{'Step':<8} {'Loss':<14} {'Effective LR':<14} {'Grad Norm':<14}")
    print("-" * 52)

    for step in range(1, 41):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

        if step <= 5 or step % 5 == 0:
            eff_lr = optimizer.get_learning_rate()
            info = optimizer.get_global_curvature()
            print(
                f"{step:<8} {loss.item():<14.6f} "
                f"{eff_lr:<14.8f} "
                f"{info['gradient_norm']:<14.4f}"
            )

    print()


# ===================================================================
# 5. Parameter group demo
# ===================================================================

def demo_param_groups() -> None:
    """Use different settings for different parameter groups."""
    print("=" * 65)
    print("DEMO 5: Multiple parameter groups with different settings")
    print("=" * 65)

    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    # Different LR and curvature settings for each layer
    optimizer = AdamTCA([
        {
            "params": model[0].parameters(),  # first layer
            "lr": 0.01,
            "curvature_mode": "cosine",
        },
        {
            "params": model[2].parameters(),  # second layer
            "lr": 0.001,
            "curvature_mode": "hybrid",
            "curvature_window": 50,
        },
    ])

    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    loss_fn = nn.MSELoss()

    print(f"{'Epoch':<8} {'Loss':<14} {'LR Group 0':<14} {'LR Group 1':<14}")
    print("-" * 52)

    for epoch in range(1, 21):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            lr0 = optimizer.get_learning_rate(0)
            lr1 = optimizer.get_learning_rate(1)
            print(
                f"{epoch:<8} {loss.item():<14.6f} "
                f"{lr0:<14.8f} {lr1:<14.8f}"
            )

    print()


# ===================================================================
# 6. Quick comparison: Adam vs Adam-TCA
# ===================================================================

def demo_adam_comparison() -> None:
    """Run a quick A/B comparison between Adam and Adam-TCA."""
    print("=" * 65)
    print("DEMO 6: Adam vs Adam-TCA quick comparison")
    print("=" * 65)

    torch.manual_seed(42)
    n_samples = 500
    n_features = 20

    X = torch.randn(n_samples, n_features)
    true_w = torch.randn(n_features, 1)
    y = X @ true_w + 0.05 * torch.randn(n_samples, 1)
    loss_fn = nn.MSELoss()

    def train_with(opt_class: type, opt_kwargs: dict, label: str) -> float:
        torch.manual_seed(42)
        model = nn.Linear(n_features, 1)
        optimizer = opt_class(model.parameters(), **opt_kwargs)

        start = time.perf_counter()
        for _ in range(200):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - start

        final_loss = loss_fn(model(X), y).item()
        print(f"  {label:20s}  Final loss: {final_loss:.6f}  Time: {elapsed:.3f}s")
        return final_loss

    # Adam baseline (PyTorch)
    loss_adam = train_with(torch.optim.Adam, {"lr": 0.01}, "Adam (PyTorch)")

    # Adam-TCA cosine
    loss_tca_cos = train_with(
        AdamTCA, {"lr": 0.01, "curvature_mode": "cosine"}, "Adam-TCA (cosine)"
    )

    # Adam-TCA hybrid
    loss_tca_hybrid = train_with(
        AdamTCA, {"lr": 0.01, "curvature_mode": "hybrid"}, "Adam-TCA (hybrid)"
    )

    print()


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Run all demos."""
    print()
    print("  Adam-TCA Comprehensive Example Suite")
    print("  ====================================")
    print()

    demo_quadratic()
    demo_linear_model()
    demo_cnn()
    demo_advanced_features()
    demo_param_groups()
    demo_adam_comparison()

    print("=" * 65)
    print("All demos completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
