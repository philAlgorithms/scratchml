# ScratchML (PyTorch): Logistic Regression From Scratch: GD vs Newton

## Goal
Implement binary logistic regression from scratch using PyTorch tensors and manual optimization, then validate correctness and convergence behavior against a standard PyTorch baseline.

## Model
Given features \(x \in \mathbb{R}^d\), the model outputs a logit
\[
z = w^\top x + b
\]
and a probability
\[
p(y=1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}.
\]

## Objective (stable form)
For labels \(y \in \{0,1\}\), the average negative log-likelihood can be written stably from logits:
\[
\mathcal{L}(w,b) = \frac{1}{n}\sum_{i=1}^n \Big(\log(1+e^{z_i}) - y_i z_i\Big) + \lambda \|w\|_2^2.
\]
In code, \(\log(1+e^{z})\) is computed using `softplus(z)` to avoid overflow.

## Optimization
### Gradient Descent
\[
\nabla_w = \frac{1}{n}X^\top(\sigma(z)-y) + 2\lambda w,\quad
\nabla_b = \frac{1}{n}\sum_i(\sigma(z_i)-y_i).
\]
Parameters are updated with a fixed learning rate.

### Newton’s Method (IRLS-style)
Newton updates use the Hessian
\[
H = \frac{1}{n}X^\top R X + 2\lambda I,
\quad R = \text{diag}(\sigma(z)(1-\sigma(z))).
\]
A small damping term is added to improve numerical stability.

## Experiment (CPU-only)
- Dataset: synthetic binary classification
- Preprocessing: standardization (mean 0, std 1)
- Metrics: validation accuracy; training loss curve
- Reproducibility: fixed seed

## Results
See `results/summary.md` and `results/benchmark.json`.

![GD vs Newton](figures/gd_vs_newton.png)

## Notes / Takeaways
- Newton converges in far fewer iterations than GD on this problem.
- From-scratch Newton implementation matches a PyTorch baseline closely on validation accuracy.

## How to reproduce
```bash
python -m scripts.benchmark_all
