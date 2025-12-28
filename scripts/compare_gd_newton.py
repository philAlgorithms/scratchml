# scripts/compare_gd_newton.py
import numpy as np
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


def eval_acc(model, X, y):
    pred = model.predict(X).cpu().numpy().reshape(-1)
    y = y.reshape(-1)
    return float((pred == y).mean())


def main():
    set_seed(42)

    X, y = make_synthetic(n=2000, d=10, seed=42)
    Xtr, Xva, ytr, yva = train_val_split(X, y, val_size=0.2, seed=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    # --- Gradient Descent ---
    gd = LogisticRegression(
        solver="gd",
        lr=0.2,            # you can tune this
        reg_lambda=1e-3,
        max_iter=300,
        tol=1e-10,
        verbose=False,
        device="cpu",
    ).fit(Xtr, ytr)

    # --- Newton ---
    newton = LogisticRegression(
        solver="newton",
        reg_lambda=1e-3,
        max_iter=50,
        tol=1e-10,
        verbose=False,
        device="cpu",
    ).fit(Xtr, ytr)

    gd_acc = eval_acc(gd, Xva, yva)
    nt_acc = eval_acc(newton, Xva, yva)

    print(f"GD:     iters={len(gd.loss_history_):3d}  final_loss={gd.loss_history_[-1]:.6f}  val_acc={gd_acc:.4f}")
    print(f"Newton: iters={len(newton.loss_history_):3d}  final_loss={newton.loss_history_[-1]:.6f}  val_acc={nt_acc:.4f}")

    plt.figure()
    plt.plot(gd.loss_history_, label="Gradient Descent")
    plt.plot(newton.loss_history_, label="Newton")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Logistic Regression: GD vs Newton (from scratch)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
