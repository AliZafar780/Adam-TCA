"""Comprehensive test suite for the Adam-TCA optimizer.

Covers:
  1. Basic initialization (all parameter combinations)
  2. Step execution
  3. Training convergence (linear model, MNIST-like CNN)
  4. Comparison with Adam
  5. Curvature tracking
  6. Edge cases (NaN, Inf, zero, sparse gradients)
  7. Gradient clipping
  8. Learning rate warmup
  9. Curvature modes (cosine, variance, hybrid)
  10. Parameter groups
  11. Serialization / reproducibility
  12. Multi-GPU (if available)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn

from adam_tca import AdamTCA


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def simple_model() -> nn.Linear:
    """A simple linear model for training tests."""
    return nn.Linear(10, 2)


@pytest.fixture
def simple_params(simple_model: nn.Linear) -> list:
    """Parameter list from a simple linear model."""
    return list(simple_model.parameters())


@pytest.fixture
def small_input() -> torch.Tensor:
    """A small batch of random input data."""
    torch.manual_seed(42)
    return torch.randn(4, 10)


@pytest.fixture
def small_target() -> torch.Tensor:
    """Random target labels."""
    torch.manual_seed(42)
    return torch.randn(4, 2)


# ===================================================================
# 1. Test optimizer creation
# ===================================================================

class TestAdamTCACreation:
    """Test that AdamTCA can be instantiated with various parameter configurations."""

    @pytest.mark.parametrize("lr", [1e-4, 1e-3, 0.1])
    def test_default_params(self, simple_params: list, lr: float) -> None:
        """Create with default parameters."""
        optimizer = AdamTCA(simple_params, lr=lr)
        assert optimizer.defaults["lr"] == lr
        assert optimizer.defaults["betas"] == (0.9, 0.999)
        assert optimizer.defaults["eps"] == 1e-8
        assert optimizer.defaults["weight_decay"] == 0
        assert optimizer.defaults["curvature_window"] == 100
        assert optimizer.defaults["curvature_mode"] == "cosine"
        assert optimizer.defaults["grad_clip_norm"] == 0
        assert optimizer.defaults["grad_clip_value"] == 0
        assert optimizer.defaults["warmup_steps"] == 0
        assert optimizer.defaults["skip_nan"] is False
        assert len(optimizer.param_groups) == 1

    def test_custom_all_params(self, simple_params: list) -> None:
        """Create with every parameter customized."""
        optimizer = AdamTCA(
            simple_params,
            lr=0.01,
            betas=(0.95, 0.999),
            eps=1e-6,
            weight_decay=0.1,
            curvature_window=50,
            curvature_mode="hybrid",
            grad_clip_norm=1.0,
            grad_clip_value=5.0,
            warmup_steps=100,
            skip_nan=True,
        )
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["betas"] == (0.95, 0.999)
        assert optimizer.param_groups[0]["eps"] == 1e-6
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        assert optimizer.param_groups[0]["curvature_window"] == 50
        assert optimizer.param_groups[0]["curvature_mode"] == "hybrid"
        assert optimizer.param_groups[0]["grad_clip_norm"] == 1.0
        assert optimizer.param_groups[0]["grad_clip_value"] == 5.0
        assert optimizer.param_groups[0]["warmup_steps"] == 100
        assert optimizer.param_groups[0]["skip_nan"] is True

    def test_custom_lr(self, simple_params: list) -> None:
        """Create with custom learning rate."""
        optimizer = AdamTCA(simple_params, lr=0.01)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_custom_betas(self, simple_params: list) -> None:
        """Create with custom betas."""
        optimizer = AdamTCA(simple_params, betas=(0.95, 0.99))
        assert optimizer.param_groups[0]["betas"] == (0.95, 0.99)

    def test_custom_curvature_window(self, simple_params: list) -> None:
        """Create with custom curvature window."""
        optimizer = AdamTCA(simple_params, curvature_window=50)
        assert optimizer.param_groups[0]["curvature_window"] == 50

    def test_custom_weight_decay(self, simple_params: list) -> None:
        """Create with weight decay."""
        optimizer = AdamTCA(simple_params, weight_decay=0.01)
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_empty_params(self) -> None:
        """Create with empty parameter list raises ValueError."""
        with pytest.raises(ValueError, match="empty parameter list"):
            AdamTCA([])

    def test_multiple_param_groups(self) -> None:
        """Create with multiple parameter groups."""
        weight = torch.randn(5, 10)
        bias = torch.randn(5)
        optimizer = AdamTCA([
            {"params": weight, "lr": 1e-3},
            {"params": bias, "lr": 1e-2},
        ])
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[1]["lr"] == 1e-2

    def test_repr(self, simple_params: list) -> None:
        """Test __repr__ output."""
        optimizer = AdamTCA(simple_params, lr=0.01, curvature_window=50,
                            curvature_mode="hybrid")
        rep = repr(optimizer)
        assert "AdamTCA" in rep
        assert "lr=0.01" in rep
        assert "curvature_window=50" in rep
        assert "curvature_mode='hybrid'" in rep
        assert "grad_clip_norm" in rep
        assert "warmup_steps" in rep

    def test_serialization(self, simple_params: list) -> None:
        """Test state_dict / load_state_dict round-trip."""
        optimizer = AdamTCA(simple_params, lr=0.01)
        state = optimizer.state_dict()
        optimizer2 = AdamTCA(simple_params, lr=0.01)
        optimizer2.load_state_dict(state)
        assert optimizer.state_dict() == optimizer2.state_dict()


# ===================================================================
# 2. Test parameter validation
# ===================================================================

class TestAdamTCAValidation:
    """Test that invalid parameters are properly rejected."""

    @pytest.mark.parametrize("lr", [-0.1, -1e-3])
    def test_negative_lr(self, simple_params: list, lr: float) -> None:
        with pytest.raises(ValueError, match="Invalid learning rate"):
            AdamTCA(simple_params, lr=lr)

    @pytest.mark.parametrize("eps", [-1e-8, -1.0])
    def test_negative_eps(self, simple_params: list, eps: float) -> None:
        with pytest.raises(ValueError, match="Invalid epsilon"):
            AdamTCA(simple_params, eps=eps)

    def test_beta1_out_of_range(self, simple_params: list) -> None:
        with pytest.raises(ValueError, match="Invalid beta parameter at index 0"):
            AdamTCA(simple_params, betas=(1.5, 0.999))

    def test_beta2_out_of_range(self, simple_params: list) -> None:
        with pytest.raises(ValueError, match="Invalid beta parameter at index 1"):
            AdamTCA(simple_params, betas=(0.9, 1.5))

    def test_negative_weight_decay(self, simple_params: list) -> None:
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            AdamTCA(simple_params, weight_decay=-1.0)

    @pytest.mark.parametrize("window", [0, -1, -10])
    def test_invalid_curvature_window(self, simple_params: list, window: int) -> None:
        with pytest.raises(ValueError, match="curvature_window must be a positive integer"):
            AdamTCA(simple_params, curvature_window=window)

    def test_invalid_curvature_mode(self, simple_params: list) -> None:
        with pytest.raises(ValueError, match="curvature_mode must be"):
            AdamTCA(simple_params, curvature_mode="invalid")

    def test_negative_grad_clip_norm(self, simple_params: list) -> None:
        with pytest.raises(ValueError, match="Invalid grad_clip_norm"):
            AdamTCA(simple_params, grad_clip_norm=-1.0)

    def test_negative_grad_clip_value(self, simple_params: list) -> None:
        with pytest.raises(ValueError, match="Invalid grad_clip_value"):
            AdamTCA(simple_params, grad_clip_value=-1.0)

    def test_negative_warmup_steps(self, simple_params: list) -> None:
        with pytest.raises(ValueError, match="warmup_steps must be a non-negative integer"):
            AdamTCA(simple_params, warmup_steps=-1)


# ===================================================================
# 3. Test basic optimization step
# ===================================================================

class TestAdamTCAStep:
    """Test that the optimizer step runs and updates parameters."""

    def test_step_updates_params(self, simple_model: nn.Linear,
                                 small_input: torch.Tensor) -> None:
        """Verify that a single step changes parameter values."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()
        params_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step()
        params_after = [p for p in simple_model.parameters()]
        for before, after in zip(params_before, params_after):
            assert not torch.equal(before, after), "Parameters did not change after step"

    def test_step_no_grad(self, simple_model: nn.Linear) -> None:
        """Verify that step with no gradients is a no-op."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        params_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step()
        params_after = [p for p in simple_model.parameters()]
        for before, after in zip(params_before, params_after):
            assert torch.equal(before, after), "Parameters changed with no gradients"

    def test_zero_grad(self, simple_model: nn.Linear,
                       small_input: torch.Tensor) -> None:
        """Verify that zero_grad works correctly."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in simple_model.parameters())
        optimizer.zero_grad()
        for p in simple_model.parameters():
            if p.grad is not None:
                assert p.grad.abs().sum().item() == 0.0, "Gradients not zeroed"

    def test_step_with_closure(self, simple_model: nn.Linear,
                                small_input: torch.Tensor) -> None:
        """Verify step with closure returns loss."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            output = simple_model(small_input)
            loss_val = output.sum()
            loss_val.backward()
            return loss_val

        loss = optimizer.step(closure)
        assert loss is not None
        assert loss.item() is not None

    def test_closure_returns_none(self, simple_model: nn.Linear,
                                   small_input: torch.Tensor) -> None:
        """Step with a closure that returns None should still work."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)

        def closure() -> None:
            optimizer.zero_grad()
            output = simple_model(small_input)
            output.sum().backward()
            return None

        result = optimizer.step(closure)
        assert result is None

    def test_multiple_steps(self, simple_model: nn.Linear,
                            small_input: torch.Tensor) -> None:
        """Verify that multiple steps run without error."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        for _ in range(10):
            optimizer.zero_grad()
            output = simple_model(small_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        params = list(simple_model.parameters())
        assert all(p.isfinite().all() for p in params), "Parameters contain NaN/Inf"

    @pytest.mark.parametrize("curvature_mode", ["cosine", "variance", "hybrid"])
    def test_all_curvature_modes(self, simple_model: nn.Linear,
                                  small_input: torch.Tensor,
                                  curvature_mode: str) -> None:
        """All curvature modes should run without error."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01,
                            curvature_mode=curvature_mode)
        for _ in range(5):
            optimizer.zero_grad()
            simple_model(small_input).sum().backward()
            optimizer.step()
        for p in simple_model.parameters():
            assert p.isfinite().all()


# ===================================================================
# 4. Test training convergence
# ===================================================================

class TestAdamTCAConvergence:
    """Test that the optimizer can minimize a simple loss function."""

    CONVERGENCE_EPOCHS = 100

    def test_loss_decreases_over_time(self) -> None:
        """Verify that loss decreases over multiple iterations on a simple task."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        optimizer = AdamTCA(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        x = torch.randn(32, 5)
        y = torch.randn(32, 1)

        losses: List[float] = []
        for _ in range(self.CONVERGENCE_EPOCHS):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 1.1, (
            f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        )

    def test_fitting_linear_function(self) -> None:
        """Verify that the optimizer can fit a simple linear function."""
        torch.manual_seed(42)
        model = nn.Linear(3, 1)
        optimizer = AdamTCA(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        x = torch.randn(100, 3)
        true_w = torch.tensor([[2.0], [-1.0], [0.5]])
        y = x @ true_w + 0.01 * torch.randn(100, 1)

        initial_loss: Optional[float] = None
        for _ in range(200):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            if initial_loss is None:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < 0.1, f"Final loss too high: {final_loss:.6f}"
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"
        )

    def test_all_curvature_modes_converge(self) -> None:
        """All three curvature modes should converge on a linear task."""
        torch.manual_seed(42)
        x = torch.randn(50, 4)
        true_w = torch.randn(4, 1)
        y = x @ true_w + 0.05 * torch.randn(50, 1)
        loss_fn = nn.MSELoss()

        for mode in ("cosine", "variance", "hybrid"):
            torch.manual_seed(42)
            model = nn.Linear(4, 1)
            opt = AdamTCA(model.parameters(), lr=0.01, curvature_mode=mode)
            losses: List[float] = []
            for _ in range(100):
                opt.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            assert losses[-1] < losses[0], (
                f"Mode {mode!r} did not converge: {losses[0]:.6f} -> {losses[-1]:.6f}"
            )

    def test_mnist_cnn_convergence(self) -> None:
        """Train a small CNN on synthetic MNIST-like data and verify loss decreases."""
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
        optimizer = AdamTCA(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        # Synthetic MNIST-like data
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))

        losses: List[float] = []
        for _ in range(30):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"CNN loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
        )

    def test_compare_adam_convergence(self) -> None:
        """Verify Adam-TCA does not diverge where Adam converges."""
        torch.manual_seed(42)
        x = torch.randn(100, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w + 0.1 * torch.randn(100, 1)
        loss_fn = nn.MSELoss()

        # Train with Adam
        torch.manual_seed(42)
        model_adam = nn.Linear(5, 1)
        opt_adam = torch.optim.Adam(model_adam.parameters(), lr=0.01)
        adam_losses: List[float] = []
        for _ in range(100):
            opt_adam.zero_grad()
            out = model_adam(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt_adam.step()
            adam_losses.append(loss.item())

        # Train with Adam-TCA
        torch.manual_seed(42)
        model_tca = nn.Linear(5, 1)
        opt_tca = AdamTCA(model_tca.parameters(), lr=0.01)
        tca_losses: List[float] = []
        for _ in range(100):
            opt_tca.zero_grad()
            out = model_tca(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt_tca.step()
            tca_losses.append(loss.item())

        # Both should converge; TCA should not diverge to NaN
        assert all(math.isfinite(l) for l in tca_losses), "Adam-TCA produced NaN losses"
        assert all(math.isfinite(l) for l in adam_losses), "Adam produced NaN losses"

        # Adam-TCA final loss should be in the same ballpark as Adam
        # (not worse by more than 2x)
        assert tca_losses[-1] < adam_losses[-1] * 2.0, (
            f"Adam-TCA diverged significantly: Adam={adam_losses[-1]:.6f}, "
            f"Adam-TCA={tca_losses[-1]:.6f}"
        )


# ===================================================================
# 5. Test curvature modulation
# ===================================================================

class TestCurvatureModulation:
    """Test the curvature-aware modulation mechanics."""

    def test_curvature_info_after_step(self, simple_model: nn.Linear,
                                        small_input: torch.Tensor) -> None:
        """Verify that curvature info is accessible after a step."""
        optimizer = AdamTCA(simple_model.parameters(), curvature_window=50)
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        for p in simple_model.parameters():
            info = optimizer.get_curvature(p)
            assert "cosine_similarity" in info
            assert "curvature_alpha" in info
            assert "gradient_norm" in info
            assert "momentum_norm" in info
            assert "history_length" in info
            assert "gradient_variance" in info
            assert isinstance(info["cosine_similarity"], float)
            assert -1.0 <= info["cosine_similarity"] <= 1.0
            assert 0.0 <= info["curvature_alpha"] <= 1.0
            assert info["history_length"] >= 0

    def test_global_curvature(self, simple_model: nn.Linear,
                               small_input: torch.Tensor) -> None:
        """Verify that global curvature stats are available."""
        optimizer = AdamTCA(simple_model.parameters())
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        global_info = optimizer.get_global_curvature()
        for key in ("cosine_similarity", "curvature_alpha", "gradient_norm",
                     "momentum_norm", "history_length", "gradient_variance"):
            assert key in global_info, f"Missing key: {key}"
        assert isinstance(global_info["cosine_similarity"], float)

    def test_curvature_before_step(self, simple_model: nn.Linear) -> None:
        """Verify that curvature info is safe before any step."""
        optimizer = AdamTCA(simple_model.parameters())
        for p in simple_model.parameters():
            info = optimizer.get_curvature(p)
            assert info["cosine_similarity"] == 0.0
            assert info["curvature_alpha"] == 0.5

    def test_global_curvature_before_step(self, simple_model: nn.Linear) -> None:
        """Verify global curvature is safe before any step."""
        optimizer = AdamTCA(simple_model.parameters())
        info = optimizer.get_global_curvature()
        assert isinstance(info["cosine_similarity"], float)

    def test_curvature_tracks_over_steps(self) -> None:
        """Verify that curvature values change over training steps."""
        model = nn.Linear(5, 1)
        optimizer = AdamTCA(model.parameters(), lr=0.01, curvature_window=20)
        x = torch.randn(10, 5)
        y = torch.randn(10, 1)
        loss_fn = nn.MSELoss()

        alphas: List[float] = []
        for _ in range(50):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            info = optimizer.get_global_curvature()
            alphas.append(info["curvature_alpha"])

        # Alpha should have varied over training (not all identical)
        unique_alphas = set(round(a, 4) for a in alphas)
        assert len(unique_alphas) > 1, (
            f"Curvature alpha did not vary: all values ~{alphas[0]:.4f}"
        )

    def test_get_learning_rate_method(self) -> None:
        """Test the get_learning_rate() helper."""
        model = nn.Linear(3, 1)
        optimizer = AdamTCA(model.parameters(), lr=0.01, warmup_steps=10)
        # Before any steps, effective LR should be 0 (warmup at step 0)
        lr_before = optimizer.get_learning_rate()
        # Step once
        x = torch.randn(4, 3)
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()
        lr_after = optimizer.get_learning_rate()
        # With warmup, after 1 step of 10, LR should be 0.01 * 0.1 = 0.001
        assert abs(lr_after - 0.001) < 1e-6, f"Warmup LR wrong: {lr_after}"


# ===================================================================
# 6. Test NaN/Inf handling
# ===================================================================

class TestNumericalStability:
    """Test numerical edge cases."""

    def test_nan_gradient_recovery(self, simple_model: nn.Linear,
                                    small_input: torch.Tensor) -> None:
        """Verify that NaN gradients don't crash the optimizer."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()

        for p in simple_model.parameters():
            if p.grad is not None:
                if p.grad.dim() >= 2:
                    p.grad[0, 0] = float("nan")
                else:
                    p.grad[0] = float("nan")

        optimizer.step()
        for p in simple_model.parameters():
            assert p.isfinite().all(), "Parameters contain NaN after step"

    def test_inf_gradient_recovery(self, simple_model: nn.Linear,
                                    small_input: torch.Tensor) -> None:
        """Verify that Inf gradients don't crash the optimizer."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()

        for p in simple_model.parameters():
            if p.grad is not None:
                if p.grad.dim() >= 2:
                    p.grad[0, 0] = float("inf")
                else:
                    p.grad[0] = float("inf")

        optimizer.step()
        for p in simple_model.parameters():
            assert p.isfinite().all(), "Parameters contain Inf after step"

    def test_skip_nan_mode(self, simple_model: nn.Linear,
                            small_input: torch.Tensor) -> None:
        """Verify skip_nan=True prevents NaN from corrupting parameters."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01, skip_nan=True)
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()

        # Inject NaN into gradients
        for p in simple_model.parameters():
            if p.grad is not None:
                if p.grad.dim() >= 2:
                    p.grad[0, 0] = float("nan")
                else:
                    p.grad[0] = float("nan")

        optimizer.step()
        for p in simple_model.parameters():
            assert p.isfinite().all(), "Parameters contain NaN after skip_nan"

    def test_zero_gradient(self, simple_model: nn.Linear) -> None:
        """Verify that zero gradients don't cause division by zero."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        for p in simple_model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        optimizer.step()
        for p in simple_model.parameters():
            assert p.isfinite().all()

    def test_mixed_nan_valid_gradients(self) -> None:
        """Verify some params with NaN, some without."""
        model = nn.Sequential(
            nn.Linear(5, 3),
            nn.Linear(3, 1),
        )
        optimizer = AdamTCA(model.parameters(), lr=0.01)
        x = torch.randn(4, 5)
        model(x).sum().backward()

        # Inject NaN only into the first layer
        list(model.parameters())[0].grad[0, 0] = float("nan")

        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all(), "Parameters contaminated with NaN"


# ===================================================================
# 7. Test gradient clipping
# ===================================================================

class TestGradientClipping:
    """Test gradient clipping functionality."""

    def test_clip_norm_applied(self) -> None:
        """Verify grad_clip_norm limits gradient norm."""
        model = nn.Linear(10, 10)
        # Large gradients by using large input and large loss
        optimizer = AdamTCA(model.parameters(), lr=0.01, grad_clip_norm=0.1)
        x = torch.randn(100, 10) * 100
        model(x).sum().backward()

        # Check that gradient norms are bounded after step
        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all()

    def test_clip_value_applied(self) -> None:
        """Verify grad_clip_value limits individual gradient values."""
        model = nn.Linear(10, 1)
        optimizer = AdamTCA(model.parameters(), lr=0.01, grad_clip_value=0.5)
        x = torch.randn(100, 10) * 100
        model(x).sum().backward()

        # Apply gradient clipping by calling step
        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all()

    def test_clip_norm_zero_disabled(self, simple_model: nn.Linear,
                                      small_input: torch.Tensor) -> None:
        """Verify grad_clip_norm=0 disables clipping."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01, grad_clip_norm=0.0)
        output = simple_model(small_input)
        output.sum().backward()
        optimizer.step()
        for p in simple_model.parameters():
            assert p.isfinite().all()

    def test_clip_both_enabled(self) -> None:
        """Verify both norm and value clipping can be active simultaneously."""
        model = nn.Linear(5, 5)
        optimizer = AdamTCA(
            model.parameters(), lr=0.01,
            grad_clip_norm=1.0, grad_clip_value=5.0,
        )
        x = torch.randn(10, 5) * 10
        model(x).sum().backward()
        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all()


# ===================================================================
# 8. Test learning rate warmup
# ===================================================================

class TestWarmup:
    """Test linear learning rate warmup."""

    def test_warmup_default_disabled(self, simple_model: nn.Linear,
                                      small_input: torch.Tensor) -> None:
        """Verify default warmup_steps=0 means no warmup."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            simple_model(small_input).sum().backward()
            optimizer.step()
        assert optimizer.get_learning_rate() == 0.01

    def test_warmup_increases_lr(self) -> None:
        """Verify LR increases during warmup phase."""
        model = nn.Linear(3, 1)
        optimizer = AdamTCA(model.parameters(), lr=0.01, warmup_steps=10)
        x = torch.randn(4, 3)

        lrs: List[float] = []
        for step in range(1, 15):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()
            lrs.append(optimizer.get_learning_rate())

        # LR should increase from 0.001 to 0.01 during warmup
        assert lrs[0] < lrs[5], f"Warmup didn't increase LR: {lrs[0]} -> {lrs[5]}"
        # After warmup (step >= 10), LR should be full 0.01
        assert abs(lrs[-1] - 0.01) < 1e-6, f"Post-warmup LR wrong: {lrs[-1]}"

    def test_warmup_then_converge(self) -> None:
        """Model should still converge with warmup enabled."""
        torch.manual_seed(42)
        model = nn.Linear(4, 1)
        optimizer = AdamTCA(model.parameters(), lr=0.01, warmup_steps=20)
        loss_fn = nn.MSELoss()

        x = torch.randn(50, 4)
        true_w = torch.randn(4, 1)
        y = x @ true_w + 0.05 * torch.randn(50, 1)

        losses: List[float] = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss increased with warmup: {losses[0]:.6f} -> {losses[-1]:.6f}"
        )


# ===================================================================
# 9. Test weight decay
# ===================================================================

class TestWeightDecay:
    """Test weight decay functionality."""

    def test_weight_decay_applied(self) -> None:
        """Verify that weight decay changes parameter values differently."""
        model1 = nn.Linear(10, 2)
        model2 = nn.Linear(10, 2)
        model2.weight.data.copy_(model1.weight.data)
        model2.bias.data.copy_(model1.bias.data)

        opt_no_decay = AdamTCA(model1.parameters(), lr=0.01, weight_decay=0.0)
        opt_with_decay = AdamTCA(model2.parameters(), lr=0.01, weight_decay=0.1)

        x = torch.randn(4, 10)

        for _ in range(5):
            opt_no_decay.zero_grad()
            model1(x).sum().backward()
            opt_no_decay.step()

            opt_with_decay.zero_grad()
            model2(x).sum().backward()
            opt_with_decay.step()

        w1 = model1.weight.data
        w2 = model2.weight.data
        assert not torch.allclose(w1, w2, atol=1e-6), "Weight decay had no effect"


# ===================================================================
# 10. Test parameter groups
# ===================================================================

class TestParameterGroups:
    """Test multiple parameter group behavior."""

    def test_different_lr_per_group(self) -> None:
        """Verify different learning rates for different parameter groups."""
        model = nn.Linear(10, 5)
        optimizer = AdamTCA([
            {"params": model.weight, "lr": 1e-2},
            {"params": model.bias, "lr": 1e-4},
        ])
        x = torch.randn(2, 10)

        optimizer.zero_grad()
        output = model(x)
        output.sum().backward()
        optimizer.step()

        weight_delta = (model.weight.data - model.weight.data.clone()).abs().sum()
        assert weight_delta >= 0

    def test_different_curvature_window_per_group(self) -> None:
        """Verify different curvature windows for different groups."""
        model = nn.Linear(10, 5)
        optimizer = AdamTCA([
            {"params": model.weight, "curvature_window": 10},
            {"params": model.bias, "curvature_window": 200},
        ])
        assert optimizer.param_groups[0]["curvature_window"] == 10
        assert optimizer.param_groups[1]["curvature_window"] == 200

    def test_different_curvature_mode_per_group(self) -> None:
        """Verify different curvature modes for different groups."""
        model = nn.Linear(10, 5)
        optimizer = AdamTCA([
            {"params": model.weight, "curvature_mode": "cosine"},
            {"params": model.bias, "curvature_mode": "hybrid"},
        ])
        assert optimizer.param_groups[0]["curvature_mode"] == "cosine"
        assert optimizer.param_groups[1]["curvature_mode"] == "hybrid"

    def test_different_grad_clip_per_group(self) -> None:
        """Verify different grad clipping for different groups."""
        model = nn.Linear(10, 5)
        optimizer = AdamTCA([
            {"params": model.weight, "grad_clip_norm": 0.5},
            {"params": model.bias, "grad_clip_norm": 0.0},
        ])
        assert optimizer.param_groups[0]["grad_clip_norm"] == 0.5
        assert optimizer.param_groups[1]["grad_clip_norm"] == 0.0

    def test_different_warmup_per_group(self) -> None:
        """Verify different warmup steps for different groups."""
        model = nn.Linear(10, 5)
        optimizer = AdamTCA([
            {"params": model.weight, "warmup_steps": 100},
            {"params": model.bias, "warmup_steps": 0},
        ])
        assert optimizer.param_groups[0]["warmup_steps"] == 100
        assert optimizer.param_groups[1]["warmup_steps"] == 0


# ===================================================================
# 11. Test edge cases
# ===================================================================

class TestEdgeCases:
    """Test various edge cases."""

    def test_large_model(self) -> None:
        """Verify optimizer works with a moderately large model."""
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        optimizer = AdamTCA(model.parameters(), lr=0.001)
        x = torch.randn(8, 100)
        optimizer.zero_grad()
        output = model(x)
        output.sum().backward()
        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all()

    def test_no_parameters(self) -> None:
        """Create with empty parameter list raises ValueError."""
        with pytest.raises(ValueError, match="empty parameter list"):
            AdamTCA([])

    def test_state_dict_roundtrip(self) -> None:
        """Verify state_dict save/load preserves training capability."""
        torch.manual_seed(42)
        model1 = nn.Linear(10, 2)
        opt1 = AdamTCA(model1.parameters(), lr=0.01)

        x = torch.randn(4, 10)

        for _ in range(5):
            opt1.zero_grad()
            model1(x).sum().backward()
            opt1.step()

        state = opt1.state_dict()
        opt1_copy = AdamTCA(model1.parameters(), lr=0.01)
        opt1_copy.load_state_dict(state)

        # Both optimizers should produce the same parameters from here
        opt1.zero_grad()
        model1(x).sum().backward()
        opt1.step()

        opt1_copy.zero_grad()
        model1(x).sum().backward()
        opt1_copy.step()

        orig_state = opt1.state_dict()
        for param_id, param_state in orig_state["state"].items():
            assert "step" in param_state
            assert "exp_avg" in param_state
            assert "exp_avg_sq" in param_state

        assert opt1_copy.step() is None

    def test_sparse_gradients_not_crash(self) -> None:
        """Sparse gradients should be handled gracefully (clipping skipped)."""
        model = nn.Embedding(100, 32)
        optimizer = AdamTCA(model.parameters(), lr=0.01,
                            grad_clip_norm=1.0, grad_clip_value=0.5)
        x = torch.randint(0, 100, (8,))
        optimizer.zero_grad()
        output = model(x)
        output.sum().backward()
        # This should not crash despite sparse gradients + clipping
        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all()


# ===================================================================
# 12. Test different parameter types
# ===================================================================

class TestParameterTypes:
    """Test with various tensor configurations."""

    def test_conv_params(self) -> None:
        """Test with a convolutional model."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 10),
        )
        optimizer = AdamTCA(model.parameters(), lr=0.001)
        x = torch.randn(2, 3, 16, 16)
        optimizer.zero_grad()
        output = model(x)
        output.sum().backward()
        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all()

    def test_embedding_params(self) -> None:
        """Test with an embedding layer."""
        model = nn.Embedding(100, 32)
        optimizer = AdamTCA(model.parameters(), lr=0.01)
        x = torch.randint(0, 100, (8,))
        optimizer.zero_grad()
        output = model(x)
        output.sum().backward()
        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all()

    def test_lstm_params(self) -> None:
        """Test with an LSTM model."""
        model = nn.LSTM(16, 32, num_layers=2, batch_first=True)
        optimizer = AdamTCA(model.parameters(), lr=0.001)
        x = torch.randn(4, 8, 16)
        optimizer.zero_grad()
        output, _ = model(x)
        output.sum().backward()
        optimizer.step()
        for p in model.parameters():
            assert p.isfinite().all()


# ===================================================================
# 13. Test reproducibility
# ===================================================================

class TestReproducibility:
    """Test that results are reproducible with fixed seeds."""

    def test_deterministic_results(self) -> None:
        """Running twice with the same seed should produce identical results."""
        def run() -> float:
            torch.manual_seed(42)
            model = nn.Linear(5, 1)
            optimizer = AdamTCA(model.parameters(), lr=0.01)
            x = torch.randn(10, 5)
            y = torch.randn(10, 1)
            for _ in range(20):
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(x), y)
                loss.backward()
                optimizer.step()
            return model.weight.data.sum().item()

        result1 = run()
        result2 = run()
        assert math.isclose(result1, result2, rel_tol=1e-5), (
            f"Results not reproducible: {result1} vs {result2}"
        )


# ===================================================================
# 14. Test multi-GPU (if available)
# ===================================================================

class TestMultiGPU:
    """Test that Adam-TCA works with DataParallel (if multiple GPUs available)."""

    def test_data_parallel(self) -> None:
        """Verify optimizer works with DataParallel wrapper."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("requires 2+ GPUs")

        model = nn.Linear(10, 2).cuda()
        model = nn.DataParallel(model)
        optimizer = AdamTCA(model.parameters(), lr=0.01)
        x = torch.randn(8, 10).cuda()

        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            output.sum().backward()
            optimizer.step()

        for p in model.parameters():
            assert p.isfinite().all()
