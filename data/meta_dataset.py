import json
import os
import random
from argparse import Namespace
from os.path import join as ospj

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from data.randaugment import RandomAugment
from flags import DATA_FOLDER
from utils.utils import get_norm_values


CUSTOM_TEMPLATES = {
    "spoof_detection": "a photo of a {} document.",
}


class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        return Image.open(ospj(self.root_dir, img)).convert("RGB")


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


def dataset_transform(phase, norm_family="clip", rand_aug=False):
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == "train":
        if rand_aug:
            return transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                RandomAugment(
                    2, 7, isPIL=True,
                    augs=[
                        "Identity", "AutoContrast", "Equalize", "Brightness", "Sharpness",
                        "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    if phase in ("val", "test"):
        return transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    raise ValueError(f"Invalid transform phase: {phase}")


class SpoofDetection:
    """JSON-based spoof / recapture detection dataset."""

    def __init__(self, args):
        custom_json_path = getattr(args, "JSON_PATH", None)
        found_path = None

        if custom_json_path and os.path.exists(custom_json_path):
            found_path = custom_json_path
            print(f"[SpoofDetection] Using JSON: {found_path}")
        else:
            candidates = [
                os.path.join(DATA_FOLDER, "DM1", "spoof_detection_SRDID162.json"),
                os.path.join(DATA_FOLDER, "DM1", "spoof_detection.json"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    found_path = path
                    break

        if found_path is None:
            raise FileNotFoundError(
                f"Could not find spoof_detection JSON. Input path was: {custom_json_path}"
            )

        with open(found_path, "r") as f:
            config = json.load(f)

        # Fixed semantic names for CLIP text prompts (label order must match dataset)
        self.classnames = [
            "genuine original",
            "screen recaptured",
            "printed recaptured",
        ]
        if "classnames" in config and len(config["classnames"]) != len(self.classnames):
            print(
                f"[Warning] JSON has {len(config['classnames'])} classes, "
                f"code defines {len(self.classnames)}."
            )

        self.dataset_dir = config["data_dir"]
        self.lab2cname = {i: name for i, name in enumerate(self.classnames)}
        if not os.path.isabs(self.dataset_dir):
            self.dataset_dir = os.path.abspath(
                os.path.join(os.path.dirname(found_path), self.dataset_dir)
            )

        class DataSample:
            def __init__(self, impath, label, path):
                self.impath = impath
                self.label = label
                self.path = path

        self.train_x = [
            DataSample(os.path.join(self.dataset_dir, item["image"]), item["label"], item["image"])
            for item in config["train"]
        ]
        self.test = [
            DataSample(os.path.join(self.dataset_dir, item["image"]), item["label"], item["image"])
            for item in config["test"]
        ]
        print(
            f"[SpoofDetection] loaded: {len(self.train_x)} train, {len(self.test)} test | "
            f"dir={self.dataset_dir}"
        )


DATASET_CLASSMAP = {
    "spoof_detection": SpoofDetection,
}


class MetaDataset(Dataset):
    def __init__(
        self,
        phase,
        dataset=None,
        seed=1,
        return_images=False,
        num_shots=16,
        num_template=1,
        rand_aug=False,
        few_shot=False,
        dataset_json_paths=None,
    ):
        self.phase = phase
        self.return_images = return_images

        dataset_args = Namespace(
            SEED=seed,
            NUM_SHOTS=num_shots,
            SUBSAMPLE_CLASSES="all" if few_shot else ("new" if phase == "test" else "base"),
            JSON_PATH=dataset_json_paths,
        )
        self.dataset = DATASET_CLASSMAP[dataset](dataset_args)
        self.template = CUSTOM_TEMPLATES[dataset]
        self.classnames = self.dataset.classnames
        self.idx2label = self.dataset.lab2cname
        self.loader = ImageLoader("")
        self.transform = dataset_transform(self.phase, "clip", rand_aug=rand_aug)
        self.data_dir = self.dataset.dataset_dir
        self.dataset = self.dataset.train_x if phase == "train" else self.dataset.test

    def __getitem__(self, index):
        data_sample = self.dataset[index]
        pil_img = self.loader(data_sample.impath)
        img = self.transform(pil_img)
        data = [img, img, data_sample.label]
        if self.return_images:
            data.append(data_sample.path)
        return data

    def __len__(self):
        return len(self.dataset)
