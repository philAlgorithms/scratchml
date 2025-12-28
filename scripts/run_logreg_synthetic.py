# scripts/run_logreg_synthetic.py
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


def main():
    set_seed(42)

    X, y = make_synthetic()

    Xtr, Xva, ytr, yva = train_val_split(X, y, val_size=0.2, seed=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)

    model = LogisticRegression(
        solver="newton",      # try "gd" too
        lr=0.2,
        reg_lambda=1e-3,
        max_iter=100,
        verbose=True,
        device="cpu",
    ).fit(Xtr, ytr)

    pred = model.predict(Xva).cpu().numpy()
    acc = (pred == yva).mean()
    print("val accuracy:", acc)

    plt.figure()
    plt.plot(model.loss_history_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss (Logistic Regression)")
    plt.show()


if __name__ == "__main__":
    main()
