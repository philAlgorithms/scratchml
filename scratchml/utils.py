from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np

try:
    import torch
except Exception:
    torch = None

def set_seed(seed: int = 42, deterministic_torch: bool = True) -> None:
    """
    Seeds Python, Numpy and PyTorch(if installed)
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def train_val_split(
        X: np.ndarray,
        y: np.ndarray,
        val_size: float = 0.2,
        seed: int = 42,
        shuffle: bool = True,
):
    n = X.shape[0]
    idx = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_val = int(np.floor(n * val_size))
    val_idx =idx[:n_val]
    tr_idx = idx[n_val:]

    return X[tr_idx], X[val_idx], y[tr_idx], y[val_idx]

@dataclass
class StandardScaler:
    """
    (X - mean) / std
    """
    eps: float = 1e-12
    mean_: np.ndarray|None = None
    scale_: np.ndarray|None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_ = np.where(self.scale_ < self.eps, 1, self.scale_)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler must fit before transform()")
        return (X - self.mean_)/self.scale_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

