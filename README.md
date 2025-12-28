# ScratchML (CPU-only): Logistic Regression from Scratch in PyTorch

A small, reproducible CPU-only project implementing **binary logistic regression from scratch** using PyTorch tensors, including:
- **Manual full-batch Gradient Descent** (no `torch.optim`)
- **Newton / IRLS-style second-order optimization**
- Numerically stable loss from logits (`softplus`)
- A verification baseline using **PyTorch `nn.Linear` + LBFGS**

Artifacts (plots + results JSON/MD) are generated in one command.

## Things actually built from scratch here
- Logistic regression objective implemented directly from logits (stable form)
- Closed-form gradients for GD updates
- Closed-form Hessian for Newton/IRLS updates (with damping for stability)
- End-to-end experiment scripts producing reproducible artifacts

## Quickstart (Windows / CPU)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib

python -m scripts.benchmark_all
