from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

class LogisticRegression:
    """
    Binary Logistic Regression from scratch

    y mst be {0,1}
    L2 regularization applies to w (not intercept)
    """

    def __init__(
        self,
        reg_lambda: float = 0.0,
        fit_intercept: bool = True,
        solver: str = "gd",         # "gd" or "newton"
        lr: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        damping: float = 1e-8,      # Newtonstability
        verbose: bool = False,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        self.reg_lambda = float(reg_lambda)
        self.fit_intercept = bool(fit_intercept)
        self.solver = solver
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.damping = float(damping)
        self.verbose = bool(verbose)
        self.device = torch.device(device)
        self.dtype = dtype

        self.w: torch.Tensor|None = None
        self.b: torch.Tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.loss_history_: list[float] = []

    def _ensure_torch(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device, self.dtype)
        else:
            X = X.to(self.device, self.dtype)

        if y is None:
            return X
        
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).to(self.device, self.dtype)
        else:
            y = torch.to(self.device, self.dtype)

        return X, y
    
    def __init_params(self, d: int) -> None:
        self.w = torch.zeros(d, device=self.device, dtype=self.dtype)
        self.b = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def _logits(self, X: torch.Tensor) -> torch.Tensor:
        z = X @ self.w
        if self.fit_intercept:
            z = z + self.b
        return z
    
    def _loss(self, X: torch.tensor, y: torch.Tensor) -> torch.Tensor:
        z = self._logits(X)
        # stable: mean(softplus(z) - y*z)
        data_loss = torch.mean(F.softplus(z) - y * z)
        reg = self.reg_lambda * torch.sum(self.w * self.w)
        return data_loss + reg
    
    @torch.no_grad()
    def fit(self, X, y):
        X, y = self._ensure_torch(X, y)
        y = y.float()

        yu = torch.unique(y)
        if not torch.all((yu == 0) | (yu == 1)):
            raise ValueError(f"y must be in {{0, 1}}. Got {yu.detach().cpu().tolist()}")
        
        n, d = X.shape
        self.__init_params(d)
        self.loss_history_ = []

        prev_loss = None

        for it in range(self.max_iter):
            z = self._logits(X)
            p = torch.sigmoid(z)
            diff = p - y    # (n,)

            grad_w = (X.t() @ diff) / n + 2.0 * self.reg_lambda * self.w
            grad_b = diff.mean() if self.fit_intercept else torch.tensor(0.0, device=self.device, dtype=self.dtype)

            if self.solver == "gd":
                self.w -= self.lr * grad_w
                if self.fit_intercept:
                    self.b -= self.lr * grad_b

            elif self.solver == "newton":
                r = p * (1.0 - p)            # (n,)
                XR = X * r.unsqueeze(1)      # (n,d)
                H = (X.t() @ XR) / n         #(d,d)
                H = H + (2.0 *self.reg_lambda + self.damping) * torch.eye(
                    d, device=self.device, dtype=self.dtype
                )

                step_w = torch.linalg.solve(H, grad_w)
                self.w -= step_w

                if self.fit_intercept:
                    Hb = r.mean() + self.damping
                    self.b -= grad_b / Hb

            else:
                raise ValueError("solver must be 'gd' or 'newton'")
            
            loss_val = float(self._loss(X, y).item())
            self.loss_history_.append(loss_val)

            if self.verbose and (it % 50 == 0 or it == self.max_iter - 1):
                print(f"iter={it:4d} loss={loss_val:.6f}")

            if prev_loss is not None and abs(prev_loss - loss_val) < self.tol:
                break
            prev_loss = loss_val

        return self


    @torch.no_grad()
    def predict_prob(self, X):
        X = self._ensure_torch(X)
        p1 = torch.sigmoid(self._logits(X))
        return torch.stack([1.0 - p1, p1], dim=1)
    
    @torch.no_grad()
    def predict(self, X, threshold: float = 0.5):
        p1 = self.predict_prob(X)[:,1]
        return (p1 >= threshold).long()

