"""
Basic usage example for the Adam-TCA optimizer.

Demonstrates how to use AdamTCA to train a simple neural network
on synthetic data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import the Adam-TCA optimizer
from adam_tca import AdamTCA


def main() -> None:
    """Run a simple training demo using Adam-TCA."""

    # ------------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    n_samples = 500
    n_features = 10

    X = torch.randn(n_samples, n_features)
    # True weights: w1=2.0, w2=-1.5, w3=0.5, rest zero
    true_w = torch.zeros(n_features, 1)
    true_w[0] = 2.0
    true_w[1] = -1.5
    true_w[2] = 0.5
    y = X @ true_w + 0.1 * torch.randn(n_samples, 1)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ------------------------------------------------------------------
    # 2. Define a simple model
    # ------------------------------------------------------------------
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    # ------------------------------------------------------------------
    # 3. Create Adam-TCA optimizer
    # ------------------------------------------------------------------
    optimizer = AdamTCA(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
        curvature_window=100,  # TCA window: how many gradients to track
    )

    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    num_epochs = 10
    print("Training with Adam-TCA...")
    print(f"{'Epoch':<6} {'Loss':<12} {'Curvature (avg)':<18}")
    print("-" * 40)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Query the global curvature statistics for insight
        curvature_info = optimizer.get_global_curvature()
        avg_alpha = curvature_info["curvature_alpha"]

        print(
            f"{epoch:<6} {avg_loss:<12.6f} {avg_alpha:<18.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        test_pred = model(X)
        final_loss = criterion(test_pred, y).item()

    print("-" * 40)
    print(f"Final MSE: {final_loss:.6f}")
    print("Training complete!")


if __name__ == "__main__":
    main()
