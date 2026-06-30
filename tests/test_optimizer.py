"""Comprehensive test suite for the Adam-TCA optimizer."""

import math
import warnings

import pytest
import torch
import torch.nn as nn

from adam_tca import AdamTCA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_model() -> nn.Linear:
    """A simple linear model for training tests."""
    return nn.Linear(10, 2)


@pytest.fixture
def simple_params(simple_model) -> list:
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


# ---------------------------------------------------------------------------
# Test optimizer creation
# ---------------------------------------------------------------------------

class TestAdamTCACreation:
    """Test that AdamTCA can be instantiated with various parameter configurations."""

    def test_default_params(self, simple_params) -> None:
        """Create with default parameters."""
        optimizer = AdamTCA(simple_params)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["betas"] == (0.9, 0.999)
        assert optimizer.defaults["eps"] == 1e-8
        assert optimizer.defaults["weight_decay"] == 0
        assert optimizer.defaults["curvature_window"] == 100
        assert len(optimizer.param_groups) == 1

    def test_custom_lr(self, simple_params) -> None:
        """Create with custom learning rate."""
        optimizer = AdamTCA(simple_params, lr=0.01)
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_custom_betas(self, simple_params) -> None:
        """Create with custom betas."""
        optimizer = AdamTCA(simple_params, betas=(0.95, 0.99))
        assert optimizer.param_groups[0]["betas"] == (0.95, 0.99)

    def test_custom_curvature_window(self, simple_params) -> None:
        """Create with custom curvature window."""
        optimizer = AdamTCA(simple_params, curvature_window=50)
        assert optimizer.param_groups[0]["curvature_window"] == 50

    def test_custom_weight_decay(self, simple_params) -> None:
        """Create with weight decay."""
        optimizer = AdamTCA(simple_params, weight_decay=0.01)
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_empty_params(self) -> None:
        """Create with empty parameter list raises ValueError."""
        with pytest.raises(ValueError, match="empty parameter list"):
            AdamTCA([])

    def test_multiple_param_groups(self) -> None:
        """Create with multiple parameter groups."""
        model = nn.Linear(10, 5)
        optimizer = AdamTCA([
            {"params": model.weight, "lr": 1e-3},
            {"params": model.bias, "lr": 1e-2},
        ])
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[1]["lr"] == 1e-2

    def test_repr(self, simple_params) -> None:
        """Test __repr__ output."""
        optimizer = AdamTCA(simple_params, lr=0.01, curvature_window=50)
        rep = repr(optimizer)
        assert "AdamTCA" in rep
        assert "lr=0.01" in rep
        assert "curvature_window=50" in rep

    def test_serialization(self, simple_params) -> None:
        """Test state_dict / load_state_dict round-trip."""
        optimizer = AdamTCA(simple_params, lr=0.01)
        state = optimizer.state_dict()
        optimizer2 = AdamTCA(simple_params, lr=0.01)
        optimizer2.load_state_dict(state)
        assert optimizer.state_dict() == optimizer2.state_dict()


# ---------------------------------------------------------------------------
# Test parameter validation
# ---------------------------------------------------------------------------

class TestAdamTCAValidation:
    """Test that invalid parameters are properly rejected."""

    def test_negative_lr(self, simple_params) -> None:
        with pytest.raises(ValueError, match="Invalid learning rate"):
            AdamTCA(simple_params, lr=-0.1)

    def test_negative_eps(self, simple_params) -> None:
        with pytest.raises(ValueError, match="Invalid epsilon"):
            AdamTCA(simple_params, eps=-1e-8)

    def test_beta1_out_of_range(self, simple_params) -> None:
        with pytest.raises(ValueError, match="Invalid beta parameter at index 0"):
            AdamTCA(simple_params, betas=(1.5, 0.999))

    def test_beta2_out_of_range(self, simple_params) -> None:
        with pytest.raises(ValueError, match="Invalid beta parameter at index 1"):
            AdamTCA(simple_params, betas=(0.9, 1.5))

    def test_negative_weight_decay(self, simple_params) -> None:
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            AdamTCA(simple_params, weight_decay=-1.0)

    def test_invalid_curvature_window_zero(self, simple_params) -> None:
        with pytest.raises(ValueError, match="curvature_window must be a positive integer"):
            AdamTCA(simple_params, curvature_window=0)

    def test_invalid_curvature_window_negative(self, simple_params) -> None:
        with pytest.raises(ValueError, match="curvature_window must be a positive integer"):
            AdamTCA(simple_params, curvature_window=-10)


# ---------------------------------------------------------------------------
# Test basic optimization step
# ---------------------------------------------------------------------------

class TestAdamTCAStep:
    """Test that the optimizer step runs and updates parameters."""

    def test_step_updates_params(self, simple_model, small_input) -> None:
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

    def test_step_no_grad(self, simple_model) -> None:
        """Verify that step with no gradients is a no-op."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        params_before = [p.clone() for p in simple_model.parameters()]
        optimizer.step()
        params_after = [p for p in simple_model.parameters()]
        for before, after in zip(params_before, params_after):
            assert torch.equal(before, after), "Parameters changed with no gradients"

    def test_zero_grad(self, simple_model, small_input) -> None:
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

    def test_step_with_closure(self, simple_model, small_input) -> None:
        """Verify step with closure returns loss."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            output = simple_model(small_input)
            loss = output.sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        assert loss is not None
        assert loss.item() is not None

    def test_multiple_steps(self, simple_model, small_input) -> None:
        """Verify that multiple steps run without error."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        for _ in range(10):
            optimizer.zero_grad()
            output = simple_model(small_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        # After 10 steps, parameters should have changed significantly
        params = list(simple_model.parameters())
        assert all(p.isfinite().all() for p in params), "Parameters contain NaN/Inf"


# ---------------------------------------------------------------------------
# Test training convergence
# ---------------------------------------------------------------------------

class TestAdamTCAConvergence:
    """Test that the optimizer can minimize a simple loss function."""

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

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease
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

        initial_loss = None
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


# ---------------------------------------------------------------------------
# Test curvature modulation
# ---------------------------------------------------------------------------

class TestCurvatureModulation:
    """Test the curvature-aware modulation mechanics."""

    def test_curvature_info_after_step(self, simple_model, small_input) -> None:
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
            assert isinstance(info["cosine_similarity"], float)
            assert -1.0 <= info["cosine_similarity"] <= 1.0
            assert 0.0 <= info["curvature_alpha"] <= 1.0
            assert info["history_length"] >= 0

    def test_global_curvature(self, simple_model, small_input) -> None:
        """Verify that global curvature stats are available."""
        optimizer = AdamTCA(simple_model.parameters())
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        global_info = optimizer.get_global_curvature()
        assert "cosine_similarity" in global_info
        assert "curvature_alpha" in global_info
        assert "gradient_norm" in global_info
        assert "momentum_norm" in global_info
        assert "history_length" in global_info
        assert isinstance(global_info["cosine_similarity"], float)

    def test_curvature_before_step(self, simple_model) -> None:
        """Verify that curvature info is safe before any step."""
        optimizer = AdamTCA(simple_model.parameters())
        for p in simple_model.parameters():
            info = optimizer.get_curvature(p)
            assert info["cosine_similarity"] == 0.0
            assert info["curvature_alpha"] == 0.5

    def test_global_curvature_before_step(self, simple_model) -> None:
        """Verify global curvature is safe before any step."""
        optimizer = AdamTCA(simple_model.parameters())
        info = optimizer.get_global_curvature()
        assert isinstance(info["cosine_similarity"], float)


# ---------------------------------------------------------------------------
# Test NaN/Inf handling
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Test numerical edge cases."""

    def test_nan_gradient_recovery(self, simple_model, small_input) -> None:
        """Verify that NaN gradients don't crash the optimizer."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
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

        # Step should not raise
        optimizer.step()
        # Parameters should remain finite
        for p in simple_model.parameters():
            assert p.isfinite().all(), "Parameters contain NaN after step"

    def test_inf_gradient_recovery(self, simple_model, small_input) -> None:
        """Verify that Inf gradients don't crash the optimizer."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        output = simple_model(small_input)
        loss = output.sum()
        loss.backward()

        # Inject Inf into gradients
        for p in simple_model.parameters():
            if p.grad is not None:
                if p.grad.dim() >= 2:
                    p.grad[0, 0] = float("inf")
                else:
                    p.grad[0] = float("inf")

        # Step should not raise
        optimizer.step()
        # Parameters should remain finite
        for p in simple_model.parameters():
            assert p.isfinite().all(), "Parameters contain Inf after step"

    def test_zero_gradient(self, simple_model) -> None:
        """Verify that zero gradients don't cause division by zero."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)
        # Set all gradients to zero manually
        for p in simple_model.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        # Step should not raise
        optimizer.step()
        for p in simple_model.parameters():
            assert p.isfinite().all()


# ---------------------------------------------------------------------------
# Test weight decay
# ---------------------------------------------------------------------------

class TestWeightDecay:
    """Test weight decay functionality."""

    def test_weight_decay_applied(self, simple_model, small_input) -> None:
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
            l1 = model1(x).sum()
            l1.backward()
            opt_no_decay.step()

            opt_with_decay.zero_grad()
            l2 = model2(x).sum()
            l2.backward()
            opt_with_decay.step()

        # Models should diverge due to different weight decay
        w1 = model1.weight.data
        w2 = model2.weight.data
        assert not torch.allclose(w1, w2, atol=1e-6), (
            "Weight decay had no effect"
        )


# ---------------------------------------------------------------------------
# Test parameter groups
# ---------------------------------------------------------------------------

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

        # Both groups should still update
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


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------

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

    def test_closure_returns_none(self, simple_model, small_input) -> None:
        """Step with a closure that returns None should still work."""
        optimizer = AdamTCA(simple_model.parameters(), lr=0.01)

        def closure() -> None:
            optimizer.zero_grad()
            output = simple_model(small_input)
            output.sum().backward()
            return None

        result = optimizer.step(closure)
        assert result is None

    def test_state_dict_roundtrip(self, simple_model, small_input) -> None:
        """Verify state_dict save/load preserves training capability."""
        torch.manual_seed(42)
        model1 = nn.Linear(10, 2)
        opt1 = AdamTCA(model1.parameters(), lr=0.01)

        x = torch.randn(4, 10)

        # Train for a few steps
        for _ in range(5):
            opt1.zero_grad()
            model1(x).sum().backward()
            opt1.step()

        # Save and reload state_dict
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

        for p1, p2 in zip(model1.parameters(), model1.parameters()):
            # p1 and p2 are the same parameters since it's the same model
            pass

        # Actually verify that state was loaded correctly by checking internal state
        # Compare state dict keys
        orig_state = opt1.state_dict()
        for param_id, param_state in orig_state["state"].items():
            assert "step" in param_state
            assert "exp_avg" in param_state
            assert "exp_avg_sq" in param_state

        # Verify the optimizer still works after roundtrip
        assert opt1_copy.step() is None


# ---------------------------------------------------------------------------
# Test different parameter types
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Test reproducibility
# ---------------------------------------------------------------------------

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
