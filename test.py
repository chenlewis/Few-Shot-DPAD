"""Standalone evaluation for FE-CLIP checkpoints."""

import argparse
import glob
from os.path import join as ospj

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

cudnn.benchmark = True

from data.meta_dataset import MetaDataset
from flags import parser as train_parser
from models.common import Classification
from models.fecilp import FECLIP
from utils.utils import load_args_test


def compute_auc_eer(scores, labels):
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    P = (labels == 1).sum()
    N = (labels != 1).sum()
    if P == 0 or N == 0:
        return float("nan"), float("nan")

    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]
    tp = fp = prev_tpr = prev_fpr = auc = 0.0
    fprs, tprs = [0.0], [0.0]
    for i in range(len(scores)):
        if labels[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr, fpr = tp / P, fp / N
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
        fprs.append(fpr)
        tprs.append(tpr)
    if prev_fpr < 1.0 or prev_tpr < 1.0:
        fprs.append(1.0)
        tprs.append(1.0)
        auc += (1.0 - prev_fpr) * (1.0 + prev_tpr) / 2.0
    fprs, tprs = np.array(fprs), np.array(tprs)
    fnrs = 1.0 - tprs
    idx = np.argmin(np.abs(fprs - fnrs))
    eer = (fprs[idx] + fnrs[idx]) / 2.0
    return float(auc), float(eer)


def run_eval(model, dataloader, evaluator, device, subset="all"):
    evaluator.reset()
    model.eval()
    all_scores, all_labels = [], []
    for data in tqdm(dataloader, desc=f"Testing {subset}"):
        data = [d.to(device) for d in data]
        with torch.inference_mode():
            _, predictions = model(data, subset=subset.lower())
        predictions = predictions.cpu()
        labels = data[-1].cpu()
        evaluator.process(predictions, labels)
        pos = (
            predictions[:, 1].float()
            if predictions.ndim == 2 and predictions.size(1) >= 2
            else predictions.view(-1).float()
        )
        all_scores.append(pos)
        all_labels.append(labels.view(-1).float())

    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    auc, eer = compute_auc_eer(all_scores, all_labels)
    stats = evaluator.evaluate()
    stats["auc"] = auc
    stats["eer"] = eer
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logpath", type=str, required=True)
    parser.add_argument("--test-dataset-json-paths", type=str, default=None)
    cli_args = parser.parse_args()

    logpath = cli_args.logpath
    args = train_parser.parse_args([])
    load_args_test(ospj(logpath, "args_all.yaml"), args)
    args.logpath = logpath
    print(f"Config: {ospj(logpath, 'args_all.yaml')} | Dataset: {args.dataset}")

    device = torch.device(getattr(args, "device", "cuda"))
    test_json = (
        cli_args.test_dataset_json_paths
        or getattr(args, "dataset_json_paths2", None)
        or args.dataset_json_paths
    )
    test_set = MetaDataset(
        phase="test",
        dataset=args.dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
        num_template=args.num_text_template,
        rand_aug=args.rand_aug,
        few_shot=False,
        dataset_json_paths=test_json,
    )
    print(f"Test set: {len(test_set)} samples")

    model = FECLIP(test_set, {"all": test_set.classnames}, args, model=None, tokenizer=None, few_shot=False)
    model.to(device)

    ckpt_candidates = sorted(glob.glob(ospj(logpath, "ckpt_best_*.t7")))
    if not ckpt_candidates:
        ckpt_candidates = sorted(glob.glob(ospj(logpath, "ckpt_last_*.t7")))
    if not ckpt_candidates:
        raise FileNotFoundError(f"No ckpt_best_*.t7 or ckpt_last_*.t7 found in {logpath}")
    ckpt_path = ckpt_candidates[-1]
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("net", ckpt)
    new_sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    model.eval()

    loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers)
    evaluator = Classification(args, test_set.idx2label)
    with torch.no_grad():
        stats = run_eval(model, loader, evaluator, device, subset="all")
    print("\n===== Final =====")
    print(f"Acc: {stats['accuracy']:.4f}  |  AUC: {stats['auc']:.4f}  |  EER: {stats['eer']:.4f}")


if __name__ == "__main__":
    main()
