# 🧮 Adam-TCA - Novel PyTorch Optimizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE6842?style=flat&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License" />
</p>

---

## ✨ Overview

Adam-TCA is a novel PyTorch optimizer that applies **Differential Geometry** to the training of Transformer models. It dynamically modulates the learning rate based on the cosine similarity between the current gradient and the historical momentum vector.

## 🔬 Key Innovation

| Feature | Description |
|:--------|:------------|
| 🎯 **Geometric Adaptation** | Uses differential geometry concepts |
| 📈 **Dynamic LR Modulation** | Cosine similarity-based adjustment |
| ⚡ **Transformer-Optimized** | Specifically designed for Transformers |
| 🧠 **Momentum-Aware** | Tracks gradient momentum |

## 🚀 Quick Start

```python
import torch
from adam_tca import AdamTCA

# Create optimizer
optimizer = AdamTCA(
    model.parameters(),
    lr=1e-4,
    beta1=0.9,
    beta2=0.999,
    curvature_window=100  # TCA window size
)

# Training loop
for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy

## 🏗️ How It Works

Adam-TCA computes the Tensor Product Contraction (TPC) between the gradient manifold and the momentum manifold, using a curvature-aware window to modulate learning rates adaptively.

## 📁 Project Structure

```
Adam-TCA/
├── adam_tca.py      # Main optimizer implementation
├── tests/           # Unit tests
├── examples/        # Usage examples
└── README.md        # This file
```

## 🤝 Contributing

Contributions are welcome! Please submit issues and pull requests.

## 📜 License

MIT License

---

*Pioneering geometric optimization for deep learning 🧮*
