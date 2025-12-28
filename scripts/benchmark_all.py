# scripts/benchmark_all.py
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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


def eval_acc_scratch(model, X, y):
    pred = model.predict(X).cpu().numpy().reshape(-1)
    y = y.reshape(-1)
    return float((pred == y).mean())


@torch.no_grad()
def eval_acc_torch_linear(model, X, y):
    logits = model(X).squeeze(1)
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).long().cpu().numpy().reshape(-1)
    y = y.long().cpu().numpy().reshape(-1)
    return float((pred == y).mean())


def ensure_dirs():
    Path("figures").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)


def main():
    ensure_dirs()
    set_seed(42)

    # Data
    X, y = make_synthetic(n=2000, d=10, seed=42)
    Xtr, Xva, ytr, yva = train_val_split(X, y, val_size=0.2, seed=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    results = {"dataset": "synthetic", "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0]), "d": int(Xtr.shape[1])}

    # -----------------------
    # Scratch: GD
    # -----------------------
    t0 = time.perf_counter()
    gd = LogisticRegression(
        solver="gd",
        lr=0.2,
        reg_lambda=1e-3,
        max_iter=300,
        tol=1e-12,
        verbose=False,
        device="cpu",
    ).fit(Xtr, ytr)
    t1 = time.perf_counter()

    gd_acc = eval_acc_scratch(gd, Xva, yva)
    results["scratch_gd"] = {
        "iters": len(gd.loss_history_),
        "final_loss": float(gd.loss_history_[-1]),
        "val_acc": gd_acc,
        "train_seconds": float(t1 - t0),
    }

    # -----------------------
    # Scratch: Newton
    # -----------------------
    t0 = time.perf_counter()
    newton = LogisticRegression(
        solver="newton",
        reg_lambda=1e-3,
        max_iter=50,
        tol=1e-12,
        verbose=False,
        device="cpu",
    ).fit(Xtr, ytr)
    t1 = time.perf_counter()

    nt_acc = eval_acc_scratch(newton, Xva, yva)
    results["scratch_newton"] = {
        "iters": len(newton.loss_history_),
        "final_loss": float(newton.loss_history_[-1]),
        "val_acc": nt_acc,
        "train_seconds": float(t1 - t0),
    }

    # -----------------------
    # PyTorch baseline: nn.Linear + BCEWithLogitsLoss + LBFGS
    # -----------------------
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.float32)

    torch_model = nn.Linear(Xtr_t.shape[1], 1, bias=True)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS(torch_model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = torch_model(Xtr_t).squeeze(1)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        return loss

    t0 = time.perf_counter()
    opt.step(closure)
    t1 = time.perf_counter()

    base_acc = eval_acc_torch_linear(torch_model, Xva_t, yva_t)
    results["torch_baseline"] = {
        "val_acc": base_acc,
        "train_seconds": float(t1 - t0),
    }

    # -----------------------
    # Save plot: GD vs Newton loss curves
    # -----------------------
    plt.figure()
    plt.plot(gd.loss_history_, label="Gradient Descent (scratch)")
    plt.plot(newton.loss_history_, label="Newton (scratch)")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Logistic Regression: GD vs Newton (from scratch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/gd_vs_newton.png", dpi=200)
    plt.close()

    # -----------------------
    # Save results.json
    # -----------------------
    with open("results/benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also write a tiny markdown summary table for the report
    md = []
    md.append("# Benchmark Summary (Synthetic)\n")
    md.append("| Method | Iters | Final Loss | Val Acc | Train (s) |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| Scratch GD | {results['scratch_gd']['iters']} | {results['scratch_gd']['final_loss']:.6f} | {results['scratch_gd']['val_acc']:.4f} | {results['scratch_gd']['train_seconds']:.3f} |")
    md.append(f"| Scratch Newton | {results['scratch_newton']['iters']} | {results['scratch_newton']['final_loss']:.6f} | {results['scratch_newton']['val_acc']:.4f} | {results['scratch_newton']['train_seconds']:.3f} |")
    md.append(f"| PyTorch baseline (LBFGS) | — | — | {results['torch_baseline']['val_acc']:.4f} | {results['torch_baseline']['train_seconds']:.3f} |")
    md.append("")
    Path("results/summary.md").write_text("\n".join(md), encoding="utf-8")

    # Print console summary
    print("Saved: figures/gd_vs_newton.png")
    print("Saved: results/benchmark.json")
    print("Saved: results/summary.md\n")
    print(md[2])  # header row
    print(md[3])  # separator
    print(md[4])
    print(md[5])
    print(md[6])


if __name__ == "__main__":
    main()
