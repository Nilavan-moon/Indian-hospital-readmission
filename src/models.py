"""Model definitions, training and evaluation helpers."""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

SEED = 42


def best_threshold(y, p):
    """Threshold maximising F1 on the validation set."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return float(th[np.argmax(f1[:-1])])


def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh), f1=f1_score(y, yh), roc_auc=roc_auc_score(y, p),
                pr_auc=average_precision_score(y, p), threshold=thr, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                specificity=tn / (tn + fp))


class MLP(nn.Module):
    def __init__(self, d, hidden=(128, 64, 32), drop=0.3):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(drop)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLP:
    def __init__(self, hidden=(128, 64, 32), drop=0.3, lr=1e-3, batch=512, epochs=60, patience=6, seed=SEED):
        self.hidden, self.drop, self.lr, self.batch, self.epochs, self.patience, self.seed = hidden, drop, lr, batch, epochs, patience, seed
        self.history = {"train_loss": [], "val_loss": []}

    def fit(self, X, y, Xv, yv):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        X, Xv = torch.tensor(np.asarray(X), dtype=torch.float32), torch.tensor(np.asarray(Xv), dtype=torch.float32)
        y, yv = torch.tensor(y, dtype=torch.float32), torch.tensor(yv, dtype=torch.float32)
        self.m = MLP(X.shape[1], self.hidden, self.drop)
        pw = (1 - y.mean()) / y.mean()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(self.m.parameters(), lr=self.lr)
        best, bad, state = 1e9, 0, None
        for ep in range(self.epochs):
            self.m.train(); perm = torch.randperm(len(X)); tl = 0
            for i in range(0, len(X), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad(); l = loss_fn(self.m(X[idx]), y[idx]); l.backward(); opt.step(); tl += l.item() * len(idx)
            self.m.eval()
            with torch.no_grad():
                vl = loss_fn(self.m(Xv), yv).item()
            self.history["train_loss"].append(tl / len(X)); self.history["val_loss"].append(vl)
            if vl < best - 1e-4:
                best, bad, state = vl, 0, {k: v.clone() for k, v in self.m.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.m.load_state_dict(state); return self

    def predict_proba1(self, X):
        self.m.eval()
        with torch.no_grad():
            return torch.sigmoid(self.m(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy()


def make_sklearn_models(pos_weight):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="auc",
                                 n_jobs=-1, random_state=SEED, early_stopping_rounds=30),
    }


def run_all(Xtr, ytr, Xva, yva, Xte, yte, Xtr_s, Xva_s, Xte_s):
    """Train every model; return dict name -> results, plus fitted objects."""
    pw = (1 - ytr.mean()) / ytr.mean()
    res, fitted = {}, {}
    for name, m in make_sklearn_models(pw).items():
        scaled = name == "Logistic Regression"
        A, B, C = (Xtr_s, Xva_s, Xte_s) if scaled else (Xtr, Xva, Xte)
        t0 = time.time()
        if name == "XGBoost":
            m.fit(A, ytr, eval_set=[(B, yva)], verbose=False)
        else:
            m.fit(A, ytr)
        tt = time.time() - t0
        pv = m.predict_proba(B)[:, 1]; thr = best_threshold(yva, pv)
        t1 = time.time(); pt = m.predict_proba(C)[:, 1]; tp = time.time() - t1
        r = metrics(yte, pt, thr); r.update(train_time_s=tt, predict_time_s=tp, val_roc_auc=roc_auc_score(yva, pv))
        res[name], fitted[name] = r, (m, pt)
    t0 = time.time(); mlp = TorchMLP().fit(Xtr_s, ytr, Xva_s, yva); tt = time.time() - t0
    pv = mlp.predict_proba1(Xva_s); thr = best_threshold(yva, pv)
    t1 = time.time(); pt = mlp.predict_proba1(Xte_s); tp = time.time() - t1
    r = metrics(yte, pt, thr); r.update(train_time_s=tt, predict_time_s=tp, val_roc_auc=roc_auc_score(yva, pv),
                                        epochs_run=len(mlp.history["val_loss"]))
    res["MLP (DL)"], fitted["MLP (DL)"] = r, (mlp, pt)
    return res, fitted
