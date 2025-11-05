import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

class BinaryMetrics:
    def __init__(self):
        self.preds = []; self.targets = []
    def update(self, prob, y):
        self.preds.extend(list(prob)); self.targets.extend(list(y))
    def auroc(self):
        y = np.array(self.targets); p = np.array(self.preds)
        if len(np.unique(y))<2: return 0.5
        return float(roc_auc_score(y, p))
    def auprc(self):
        y = np.array(self.targets); p = np.array(self.preds)
        return float(average_precision_score(y, p))
    def f1(self, thr=0.5):
        y = np.array(self.targets); p = (np.array(self.preds)>=thr).astype(int)
        if len(np.unique(y))<2: return 0.0
        return float(f1_score(y, p))
    def acc(self, thr=0.5):
        y = np.array(self.targets); p = (np.array(self.preds)>=thr).astype(int)
        return float(accuracy_score(y, p))
    def compute(self):
        return {"auroc": self.auroc(), "auprc": self.auprc(), "f1": self.f1(), "acc": self.acc()}
    def summary_str(self):
        d = self.compute()
        return f"AUC {d['auroc']:.3f} | AUPRC {d['auprc']:.3f} | F1 {d['f1']:.3f} | Acc {d['acc']:.3f}"
