import os
import shutil
import sys
from os.path import join as ospj

import torch
import torch.distributed as dist
import yaml


def get_norm_values(norm_family="imagenet"):
    if norm_family == "imagenet":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if norm_family == "clip":
        return [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    raise ValueError(f"Incorrect normalization family: {norm_family}")


def save_args(args, log_path, argfile):
    try:
        shutil.copy(argfile, log_path)
    except Exception:
        print("Config exists")
    try:
        shutil.copytree("models/", ospj(log_path, "models"))
    except Exception:
        print("Already exists")
    with open(ospj(log_path, "args_all.yaml"), "w") as f:
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
    with open(ospj(log_path, "args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))


def load_args(filename, args):
    with open(filename, "r") as stream:
        data_loaded = yaml.safe_load(stream)
    for _, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def load_args_test(filename, args):
    with open(filename, "r") as stream:
        data_loaded = yaml.load(stream, Loader=yaml.UnsafeLoader)
    for key, val in data_loaded.__dict__.items():
        setattr(args, key, val)


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
