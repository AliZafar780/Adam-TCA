# Adam-TCA: Taylor-Centric Adam Optimizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-1.10%2B-EE6842?style=flat&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat" alt="License" />
  <img src="https://img.shields.io/github/actions/workflow/status/AliZafar780/Adam-TCA/python-tests.yml?style=flat&logo=github" alt="Tests" />
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen" alt="Coverage" />
</p>

<p align="center">
  <em>A curvature-aware optimizer that dynamically modulates learning rates using differential geometry principles.</em>
</p>

---

## Overview

**Adam-TCA** (Taylor-Centric Adam) is a novel PyTorch optimizer that extends the standard Adam algorithm by incorporating **curvature-aware learning rate modulation**. It computes the cosine similarity between the current gradient and the running momentum estimate, using this geometric signal as a proxy for local curvature. When the gradient and momentum are aligned, the learning rate is increased to accelerate convergence; when they are misaligned (indicating high curvature or a changing gradient landscape), the learning rate is reduced to improve stability.

### Key Innovation

| Feature | Description |
|:--------|:------------|
| **Geometric Adaptation** | Uses cosine similarity between gradient and momentum as a curvature proxy |
| **Dynamic LR Modulation** | Continuously adjusts effective learning rate per parameter |
| **Transformer-Optimized** | Designed for deep networks where gradient-momentum alignment carries geometric information |
| **Momentum-Aware** | Leverages the running momentum estimate for adaptive control |
| **Compatible** | Drop-in replacement for `torch.optim.Adam` with identical interface |

---

## Installation

### From PyPI (coming soon)

```bash
pip install adam-tca
```

### From source

```bash
git clone https://github.com/AliZafar780/Adam-TCA.git
cd Adam-TCA
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy (required by PyTorch)

---

## Quick Start

```python
import torch
import torch.nn as nn
from adam_tca import AdamTCA

# Define your model
model = nn.Linear(10, 2)

# Create the Adam-TCA optimizer (drop-in replacement for Adam)
optimizer = AdamTCA(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    curvature_window=100,  # Number of gradients to track for curvature estimation
)

# Standard training loop
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

---

## API Reference

### `AdamTCA(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, curvature_window=100)`

#### Parameters

| Argument | Type | Default | Description |
|:---------|:-----|:--------|:------------|
| `params` | `iterable` | required | Iterable of parameters to optimize or dicts defining parameter groups |
| `lr` | `float` | `1e-3` | Learning rate |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | Coefficients for computing running averages of gradient and its square |
| `eps` | `float` | `1e-8` | Term added to denominator for numerical stability |
| `weight_decay` | `float` | `0` | Weight decay (L2 penalty) |
| `curvature_window` | `int` | `100` | Number of recent gradient vectors to retain for curvature estimation |

#### Methods

| Method | Description |
|:-------|:------------|
| `step(closure=None)` | Performs a single optimization step |
| `zero_grad(set_to_none=False)` | Clears the gradients of all optimized parameters |
| `get_curvature(param)` | Returns curvature info for a specific parameter tensor |
| `get_global_curvature()` | Returns average curvature statistics across all parameters |

#### `get_curvature(param)` return value

Returns a dictionary with:
- `cosine_similarity` — cosine similarity between current gradient and momentum (`[-1, 1]`)
- `curvature_alpha` — modulation factor `(1 + cos_sim) / 2` (`[0, 1]`)
- `gradient_norm` — L2 norm of the gradient
- `momentum_norm` — L2 norm of the momentum estimate
- `history_length` — number of gradients in the curvature window

---

## How It Works

### The Adam Base

Adam-TCA builds on the Adam optimizer, maintaining:

- **First moment estimate** (momentum): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
- **Second moment estimate** (adaptive LR): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
- **Bias-corrected estimates**: $\hat{m}_t$, $\hat{v}_t$

### Curvature Modulation

The core innovation is the **curvature-aware modulation factor** $\alpha_t$:

$$\alpha_t = \frac{1 + \cos(\theta_t)}{2}$$

where $\cos(\theta_t)$ is the cosine similarity between the current gradient $g_t$ and the momentum estimate $m_t$:

$$\cos(\theta_t) = \frac{g_t \cdot m_t}{\|g_t\| \cdot \|m_t\|}$$

The effective learning rate becomes:

$$\eta_t^{\text{eff}} = \eta \cdot \alpha_t$$

### Intuition

- **High alignment** ($\cos \approx 1$, $\alpha \approx 1$): The gradient consistently points in the same direction as the momentum. This indicates a low-curvature region where we can safely take larger steps.

- **Low alignment** ($\cos \approx 0$, $\alpha \approx 0.5$): The gradient and momentum are uncorrelated. This suggests moderate curvature or noise, and the learning rate is reduced to 50%.

- **Negative alignment** ($\cos \approx -1$, $\alpha \approx 0$): The gradient opposes the momentum. This signals high curvature, a valley, or an optimum nearby. The learning rate is heavily reduced to prevent overshooting.

### Final Update Rule

$$\theta_{t+1} = \theta_t - \eta_t^{\text{eff}} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

---

## Project Structure

```
Adam-TCA/
├── adam_tca.py              # Main optimizer implementation
├── setup.py                 # Package configuration
├── requirements.txt         # Dependencies
├── README.md                # This file
├── LICENSE                  # Apache 2.0
├── SECURITY.md              # Security policy
├── tests/
│   ├── __init__.py
│   └── test_optimizer.py    # Comprehensive test suite
├── examples/
│   ├── __init__.py
│   └── basic_usage.py       # Usage demo
└── .github/
    └── workflows/
        └── python-tests.yml # CI configuration
```

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=adam_tca
```

---

## Citation

If you use Adam-TCA in your research, please cite:

```bibtex
@software{adam_tca_2025,
  author = {Ali Zafar},
  title = {Adam-TCA: Taylor-Centric Adam Optimizer with Curvature-Aware Learning Rate Modulation},
  year = {2025},
  url = {https://github.com/AliZafar780/Adam-TCA},
}
```

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome and appreciated!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please ensure tests pass and coverage remains high.

---

<p align="center">
  <em>Pioneering geometric optimization for deep learning</em>
</p>
