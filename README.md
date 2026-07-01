# Adam-TCA: Curvature-Aware Adam Optimizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-1.10%2B-EE6842?style=flat&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat" alt="License" />
  <img src="https://img.shields.io/github/actions/workflow/status/AliZafar780/Adam-TCA/python-tests.yml?style=flat&logo=github" alt="Tests" />
</p>

<p align="center">
  <em>A curvature-aware optimizer that dynamically modulates learning rates using gradient-momentum cosine similarity and gradient variance signals.</em>
</p>

---

## Overview

**Adam-TCA** is a PyTorch optimizer that extends the standard Adam algorithm by incorporating **curvature-aware learning rate modulation**. It computes geometric signals — cosine similarity between the current gradient and running momentum estimate, and optionally gradient variance — and uses them as a proxy for local curvature.

When the gradient and momentum are aligned (low curvature), the learning rate is increased to accelerate convergence. When they are misaligned (high curvature or changing landscape), the learning rate is reduced to improve stability.

### Key Features

| Feature | Description |
|:--------|:------------|
| **3 Curvature Modes** | `cosine` (cosine similarity), `variance` (gradient variance), `hybrid` (combined) |
| **Gradient Clipping** | Clip by norm (`grad_clip_norm`) and/or value (`grad_clip_value`) |
| **LR Warmup** | Linear warmup from 0 to `lr` over `warmup_steps` |
| **NaN/Inf Handling** | Zero-out or skip (`skip_nan=True`) parameters with corrupt gradients |
| **Curvature Introspection** | Per-parameter and global curvature statistics via `get_curvature()` and `get_global_curvature()` |
| **Effective LR Query** | `get_learning_rate()` returns the current LR accounting for warmup |
| **Drop-in Compatible** | Same interface as `torch.optim.Adam` |

---

## Installation

### From source

```bash
git clone https://github.com/AliZafar780/Adam-TCA.git
cd Adam-TCA
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 1.10+

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
    curvature_window=100,
    curvature_mode='hybrid',       # cosine | variance | hybrid
    grad_clip_norm=0.0,            # 0 = disabled
    grad_clip_value=0.0,           # 0 = disabled
    warmup_steps=0,                # 0 = disabled
    skip_nan=False,                # False = zero out, True = skip step
)

# Standard training loop
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Inspect curvature stats
    alpha = optimizer.get_global_curvature()["curvature_alpha"]
    eff_lr = optimizer.get_learning_rate()
```

---

## API Reference

### `AdamTCA(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, curvature_window=100, curvature_mode='cosine', grad_clip_norm=0, grad_clip_value=0, warmup_steps=0, skip_nan=False)`

#### Parameters

| Argument | Type | Default | Description |
|:---------|:-----|:--------|:------------|
| `params` | `iterable` | required | Iterable of parameters to optimize or dicts defining parameter groups |
| `lr` | `float` | `1e-3` | Learning rate |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | Coefficients for running averages of gradient and its square |
| `eps` | `float` | `1e-8` | Term added to denominator for numerical stability |
| `weight_decay` | `float` | `0` | Weight decay (L2 penalty) |
| `curvature_window` | `int` | `100` | Number of recent gradients to retain (used in `variance` / `hybrid` modes) |
| `curvature_mode` | `str` | `'cosine'` | Curvature estimation: `'cosine'`, `'variance'`, or `'hybrid'` |
| `grad_clip_norm` | `float` | `0` | Max gradient norm (0 = disabled) |
| `grad_clip_value` | `float` | `0` | Max gradient value (0 = disabled) |
| `warmup_steps` | `int` | `0` | Number of linear warmup steps (0 = disabled) |
| `skip_nan` | `bool` | `False` | If `True`, skip params with NaN/Inf gradients instead of zeroing |

#### Methods

| Method | Description |
|:-------|:------------|
| `step(closure=None)` | Performs a single optimization step |
| `zero_grad(set_to_none=False)` | Clears the gradients of all optimized parameters |
| `get_curvature(param)` | Returns curvature info for a specific parameter tensor |
| `get_global_curvature()` | Returns average curvature across all parameters |
| `get_learning_rate(group_index=0)` | Returns current effective LR (with warmup applied) |

#### `get_curvature(param)` return value

Returns a dictionary with:
- `cosine_similarity` — cosine similarity between current gradient and momentum (`[-1, 1]`)
- `curvature_alpha` — modulation factor `(1 + cos_sim) / 2` (`[0, 1]`)
- `gradient_norm` — L2 norm of the gradient
- `momentum_norm` — L2 norm of the momentum estimate
- `history_length` — number of gradients in the curvature window
- `gradient_variance` — variance of gradients in history (0 if mode is `'cosine'`)

---

## Curvature Modes

### Cosine Mode (default)

Uses cosine similarity between the current gradient $g_t$ and the momentum estimate $m_t$:

$$\alpha_t = \frac{1 + \cos(\theta_t)}{2}, \quad
\cos(\theta_t) = \frac{g_t \cdot m_t}{\|g_t\| \cdot \|m_t\|}$$

### Variance Mode

Uses the variance of recent gradients as a curvature signal. High variance suggests a noisy/high-curvature region. The normalized variance signal is:

$$\text{var\_signal} = 1 - \frac{\sigma^2}{\sigma^2 + 1 + \epsilon}$$

where $\sigma^2$ is the mean variance across all dimensions in the gradient history.

### Hybrid Mode

Averages cosine similarity and variance signals:

$$\alpha_t^{\text{hybrid}} = \frac{1}{2}\alpha_t^{\text{cosine}} + \frac{1}{2}\alpha_t^{\text{variance}}$$

---

## Parameter Groups

Different layers can have different settings:

```python
optimizer = AdamTCA([
    {"params": model.features.parameters(), "lr": 0.01, "curvature_mode": "cosine"},
    {"params": model.classifier.parameters(), "lr": 0.001, "curvature_mode": "hybrid"},
])
```

---

## Migration Guide from Adam

Adam-TCA is a **drop-in replacement** for `torch.optim.Adam`. Simply change:

```python
# Before
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# After
from adam_tca import AdamTCA
optimizer = AdamTCA(model.parameters(), lr=1e-3)
```

No other code changes are needed. All existing training loops, LR schedulers, and checkpointing work identically.

### Optional enhancements

```python
# Add curvature modulation
optimizer = AdamTCA(model.parameters(), lr=1e-3, curvature_mode="hybrid")

# Add gradient clipping
optimizer = AdamTCA(model.parameters(), lr=1e-3, grad_clip_norm=1.0)

# Add warmup
optimizer = AdamTCA(model.parameters(), lr=1e-3, warmup_steps=1000)
```

---

## How It Works

### The Adam Base

Adam-TCA builds on the Adam optimizer, maintaining:
- **First moment** (momentum): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
- **Second moment** (adaptive LR): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
- **Bias-corrected estimates**: $\hat{m}_t$, $\hat{v}_t$

### Curvature Modulation

The effective learning rate is:

$$\eta_t^{\text{eff}} = \eta \cdot \alpha_t$$

where $\alpha_t$ depends on the selected curvature mode.

### Intuition

- **High alignment** ($\cos \approx 1$, $\alpha \approx 1$): Low-curvature region — safe to take larger steps.
- **Low alignment** ($\cos \approx 0$, $\alpha \approx 0.5$): Moderate curvature or noise — reduce LR.
- **Negative alignment** ($\cos \approx -1$, $\alpha \approx 0$): High curvature, valley, or optimum nearby — reduce LR heavily.

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
│   └── test_optimizer.py    # Comprehensive test suite (~60 tests)
├── examples/
│   ├── __init__.py
│   └── basic_usage.py       # 6 demos: quadratic, linear, CNN, advanced features, param groups, Adam comparison
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

The test suite covers:
- All parameter combinations and validation
- Step execution and convergence (linear model, CNN)
- All three curvature modes
- Gradient clipping (norm and value)
- Learning rate warmup
- NaN/Inf handling (zero-out and skip modes)
- Weight decay
- Multiple parameter groups
- Serialization round-trip
- Deterministic reproducibility
- Multi-GPU (if available)
- Sparse gradients
- Various model architectures (CNN, LSTM, Embedding)

---

## Citation

If you use Adam-TCA in your research, please cite:

```bibtex
@software{adam_tca_2025,
  author = {Ali Zafar},
  title = {Adam-TCA: Curvature-Aware Adam Optimizer with Learning Rate Modulation},
  year = {2025},
  url = {https://github.com/AliZafar780/Adam-TCA},
}
```

---

## License

This project is licensed under the **Apache License 2.0**.

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
