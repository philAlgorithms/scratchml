# scripts/compare_with_torch_baseline.py
import numpy as np
import torch
import torch.nn as nn

from scratchml.utils import set_seed, train_val_split, StandardScaler
from scratchml.linear_model import LogisticRegression


def make_synthetic(n=2000, d=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=(d,))
    logits = X @ true_w + 0.3 * rng.normal(size=(n,))
    probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
    y = (probs > 0.5).astype(np.int64)
    return X, y


def acc_from_logits(logits, y_true):
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).long().cpu().numpy().reshape(-1)
    y_true = y_true.cpu().numpy().reshape(-1)
    return float((pred == y_true).mean())


def main():
    set_seed(42)

    X, y = make_synthetic(n=2000, d=10, seed=42)
    Xtr, Xva, ytr, yva = train_val_split(X, y, val_size=0.2, seed=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    # -----------------------
    # From-scratch Newton
    # -----------------------
    scratch = LogisticRegression(
        solver="newton",
        reg_lambda=1e-3,
        max_iter=50,
        verbose=False,
        device="cpu",
    ).fit(Xtr, ytr)

    scratch_pred = scratch.predict(Xva).cpu().numpy().reshape(-1)
    scratch_acc = float((scratch_pred == yva.reshape(-1)).mean())

    # -----------------------
    # PyTorch baseline
    # -----------------------
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)

    model = nn.Linear(Xtr_t.shape[1], 1, bias=True)
    loss_fn = nn.BCEWithLogitsLoss()

    # LBFGS converges fast for logistic regression (CPU-friendly)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = model(Xtr_t).squeeze(1)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        val_logits = model(Xva_t).squeeze(1)
        torch_acc = acc_from_logits(val_logits, yva_t)

    print(f"Scratch (Newton) val acc: {scratch_acc:.4f}")
    print(f"Torch baseline   val acc: {torch_acc:.4f}")


if __name__ == "__main__":
    main()
