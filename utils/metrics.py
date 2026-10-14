"""Dataset-level binary confusion matrix; class 1 is plant."""
import torch
import numpy as np
from .surface_metrics import surface_scores


class SegmentationMetrics:
    def __init__(self):
        self.cm = torch.zeros(2, 2, dtype=torch.int64)
        self.surface_sums = np.zeros(3, dtype=np.float64)
        self.image_count = 0

    def update(self, logits, target):
        prediction = logits.argmax(1)
        if target.min() < 0 or target.max() > 1:
            raise ValueError("Expected binary class IDs")
        bins = (2 * target + prediction).detach().reshape(-1).cpu()
        self.cm += torch.bincount(bins, minlength=4).reshape(2, 2)
        for p, t in zip(prediction.detach().cpu().numpy(), target.detach().cpu().numpy()):
            self.surface_sums += surface_scores(p == 1, t == 1)
            self.image_count += 1

    def compute(self):
        cm = self.cm.double()
        diag = cm.diag()
        counts = cm.sum(1)
        union = cm.sum(0) + counts - diag
        present = union > 0
        iou = diag / union.clamp_min(1)
        accuracy = diag / counts.clamp_min(1)
        tn, fp, fn, tp = [cm[i, j].item() for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        def ratio(a, b):
            return a / b if b else None
        return {
            "Foreground IoU": ratio(tp, tp + fp + fn),
            "Background IoU": ratio(tn, tn + fp + fn),
            "Mean IoU": iou[present].mean().item() if present.any() else None,
            "Dice": ratio(2 * tp, 2 * tp + fp + fn),
            "Precision": ratio(tp, tp + fp),
            "Recall": ratio(tp, tp + fn),
            "Specificity": ratio(tn, tn + fp),
            "HD95": float(self.surface_sums[0] / self.image_count) if self.image_count else None,
            "ASSD": float(self.surface_sums[1] / self.image_count) if self.image_count else None,
            "Boundary F1": float(self.surface_sums[2] / self.image_count) if self.image_count else None,
            "Pixel Accuracy": ratio(tp + tn, cm.sum().item()),
            "Mean Accuracy": accuracy[counts > 0].mean().item() if (counts > 0).any() else None,
            "Frequency Weighted IoU": ((counts * iou).sum() / cm.sum()).item() if cm.sum() else None,
            "confusion_matrix": self.cm.tolist(),
        }
