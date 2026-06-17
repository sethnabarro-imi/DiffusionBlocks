import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import lightning as L
from datasets import load_dataset, DatasetDict


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def transforms(examples, transform):
    examples["pixel_values"] = [transform(image) for image in examples["image"]]
    return {
        "pixel_values": examples["pixel_values"],
        "labels": examples["label"],
    }


def add_gaussian_noise(x, std: float):
    return x + std * torch.randn_like(x)


class ImageDataModule(L.LightningDataModule):
    data_name = None
    image_size = None
    dataset_kwargs = {}
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(
        self,
        batch_size: int = 64,
        eval_batch_size: int | None = None,
        num_workers: int | None = None,
        add_rand_aug: bool = False,
        input_noise_std: float = 0.0,
    ):
        super().__init__()
        if input_noise_std < 0.0:
            raise ValueError("--input_noise_std must be non-negative")
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.input_noise_std = input_noise_std
        self.collate_fn = None
        train_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.RandomResizedCrop(self.image_size),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transformers = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ]
        post_transforms = self.post_transforms()
        self.train_transforms = T.Compose(train_transforms + post_transforms)
        self.val_transforms = T.Compose(val_transformers + post_transforms)
        self.datasets = {}
        self.train_key = "train"
        self.val_key = "validation"
        self.test_key = "test"

    def prepare_data(self):
        load_dataset(self.data_name, num_proc=os.cpu_count() // 2)

    def post_transforms(self):
        post_transforms = [T.ToTensor()]
        if self.input_noise_std > 0.0:
            post_transforms.append(
                T.Lambda(
                    partial(add_gaussian_noise, std=self.input_noise_std)
                )
            )
        post_transforms.append(T.Normalize(mean=self.mean, std=self.std))
        return post_transforms

    def setup_dataset(self, data: DatasetDict):
        return data

    def setup(self, stage=None):
        data = load_dataset(self.data_name)
        data = self.setup_dataset(data)
        train_data = data[self.train_key].with_transform(
            partial(transforms, transform=self.train_transforms)
        )
        train_eval_data = data[self.train_key].with_transform(
            partial(transforms, transform=self.val_transforms)
        )
        self.datasets["train"] = train_data
        self.datasets["train_eval"] = train_eval_data
        self.train_eval_dataloader = self._train_eval_dataloader
        if self.val_key is not None:
            val_data = data[self.val_key].with_transform(
                partial(transforms, transform=self.val_transforms)
            )
            self.datasets["val"] = val_data
            self.val_dataloader = self._val_dataloader
        if self.test_key is not None:
            test_data = data[self.test_key].with_transform(
                partial(transforms, transform=self.val_transforms)
            )
            self.datasets["test"] = test_data
            self.test_dataloader = self._test_dataloader

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def _train_eval_dataloader(self):
        return DataLoader(
            self.datasets["train_eval"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )


class CIFAR100DataModule(ImageDataModule):
    data_name = "uoft-cs/cifar100"
    image_size = 32
    num_labels = 100
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        add_rand_aug = kwargs.get("add_rand_aug", False)
        self.val_key = None
        self.test_key = "test"
        # ref: https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT/blob/main/data_loader.py#L62
        train_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize([self.image_size, self.image_size]),
            T.RandomCrop(self.image_size, padding=4),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize([self.image_size, self.image_size]),
            T.CenterCrop(self.image_size),
        ]
        post_transforms = self.post_transforms()
        self.train_transforms = T.Compose(train_transforms + post_transforms)
        self.val_transforms = T.Compose(val_transforms + post_transforms)

    def setup_dataset(self, data: DatasetDict):
        data = data.remove_columns(["coarse_label"])
        data = data.rename_columns({"img": "image", "fine_label": "label"})
        return data


class TinyImageNetDataModule(ImageDataModule):
    data_name = "zh-plus/tiny-imagenet"
    image_size = 64
    num_labels = 200
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        add_rand_aug = kwargs.get("add_rand_aug", False)
        self.val_key = "valid"
        self.test_key = "valid"
        train_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.RandomResizedCrop(self.image_size),
            T.RandomHorizontalFlip(),
        ]
        if add_rand_aug:
            train_transforms.extend([T.RandAugment()])
        val_transforms = [
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
        ]
        post_transforms = self.post_transforms()
        self.train_transforms = T.Compose(train_transforms + post_transforms)
        self.val_transforms = T.Compose(val_transforms + post_transforms)


def load_data(args):
    data_kwargs = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "add_rand_aug": args.add_rand_aug,
        "input_noise_std": args.input_noise_std,
    }
    if args.data_name == "cifar100":
        return CIFAR100DataModule(**data_kwargs)
    elif args.data_name == "tiny-imagenet":
        return TinyImageNetDataModule(**data_kwargs)
    else:
        raise ValueError(f"Invalid data name: {args.data_name}")
