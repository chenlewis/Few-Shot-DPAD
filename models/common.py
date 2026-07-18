import os.path as osp
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.test_per_class_result:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)
        self.output_dir = osp.join(cfg.cv_dir, cfg.name)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]
        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                self._per_class_res[label].append(int(matches[i].item()))

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true, self._y_pred, average="macro", labels=np.unique(self._y_true)
        )
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = sorted(self._per_class_res.keys())
            print("=> per-class result")
            accs = []
            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                class_acc = 100.0 * correct / total
                accs.append(class_acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\tcorrect: {correct:,}\tacc: {class_acc:.1f}%"
                )
            mean_acc = float(np.mean(accs))
            print(f"* average: {mean_acc:.1f}%")
            results["perclass_accuracy"] = mean_acc

        if self.cfg.test_compute_cmat:
            cmat = confusion_matrix(self._y_true, self._y_pred, normalize="true")
            save_path = osp.join(self.output_dir, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
