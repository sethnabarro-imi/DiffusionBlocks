import os
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
    task_type = "classification"
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


def synthetic_teacher_forward(
    x: torch.Tensor,
    *,
    input_dim: int,
    output_dim: int,
    depth: int,
    width: int,
    seed: int,
    activation: str,
) -> torch.Tensor:
    if depth < 1:
        raise ValueError("--synthetic_teacher_depth must be at least 1")
    if width < 1:
        raise ValueError("--synthetic_teacher_width must be at least 1")

    generator = torch.Generator().manual_seed(seed)
    hidden = x
    in_dim = input_dim
    for _ in range(max(depth - 1, 0)):
        weight = torch.randn(width, in_dim, generator=generator) / (in_dim**0.5)
        bias = torch.randn(width, generator=generator) * 0.01
        hidden = F.linear(hidden, weight, bias)
        if activation == "relu":
            hidden = F.relu(hidden)
        elif activation == "gelu":
            hidden = F.gelu(hidden)
        elif activation == "tanh":
            hidden = torch.tanh(hidden)
        else:
            raise ValueError(f"Invalid synthetic teacher activation: {activation}")
        in_dim = width

    weight = torch.randn(output_dim, in_dim, generator=generator) / (in_dim**0.5)
    bias = torch.randn(output_dim, generator=generator) * 0.01
    return F.linear(hidden, weight, bias)


class SyntheticTeacherDataset(Dataset):
    def __init__(
        self,
        *,
        num_examples: int,
        input_dim: int,
        image_size: int,
        num_labels: int,
        target_type: str,
        teacher_depth: int,
        teacher_width: int,
        teacher_seed: int,
        data_seed: int,
        activation: str,
        label_noise: float,
        target_noise_std: float,
        input_noise_std: float,
    ):
        super().__init__()
        if num_examples < 1:
            raise ValueError("synthetic split sizes must be at least 1")
        flat_image_dim = 3 * image_size * image_size
        if input_dim < 1:
            raise ValueError("--synthetic_input_dim must be at least 1")
        if input_dim > flat_image_dim:
            raise ValueError(
                "--synthetic_input_dim cannot exceed 3 * image_size * image_size"
            )
        if target_type not in ["classification", "continuous"]:
            raise ValueError("--synthetic_target_type must be classification or continuous")
        if target_type == "classification" and num_labels < 2:
            raise ValueError("--synthetic_num_classes must be at least 2")
        if target_type == "continuous" and num_labels < 1:
            raise ValueError("--synthetic_target_dim must be at least 1")
        if label_noise < 0.0 or label_noise > 1.0:
            raise ValueError("--synthetic_label_noise must be between 0 and 1")
        if target_noise_std < 0.0:
            raise ValueError("--synthetic_target_noise_std must be non-negative")
        if input_noise_std < 0.0:
            raise ValueError("--input_noise_std must be non-negative")

        generator = torch.Generator().manual_seed(data_seed)
        features = torch.randn(num_examples, input_dim, generator=generator)
        logits = synthetic_teacher_forward(
            features,
            input_dim=input_dim,
            output_dim=num_labels,
            depth=teacher_depth,
            width=teacher_width,
            seed=teacher_seed,
            activation=activation,
        )
        if target_type == "classification":
            labels = logits.argmax(dim=-1)
        else:
            labels = logits

        if target_type == "classification" and label_noise > 0.0:
            flip_mask = torch.rand(num_examples, generator=generator) < label_noise
            noisy_labels = torch.randint(
                low=0,
                high=num_labels,
                size=(num_examples,),
                generator=generator,
            )
            labels = torch.where(flip_mask, noisy_labels, labels)
        elif target_type == "continuous" and target_noise_std > 0.0:
            labels = labels + target_noise_std * torch.randn(
                labels.shape,
                generator=generator,
            )

        if input_noise_std > 0.0:
            features = features + input_noise_std * torch.randn(
                features.shape,
                generator=generator,
            )

        flat_images = torch.zeros(num_examples, flat_image_dim)
        flat_images[:, :input_dim] = features
        self.pixel_values = flat_images.view(num_examples, 3, image_size, image_size)
        if target_type == "classification":
            self.labels = labels.long()
        else:
            self.labels = labels.float()

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx):
        return {
            "pixel_values": self.pixel_values[idx],
            "labels": self.labels[idx],
        }


class SyntheticTeacherDataModule(L.LightningDataModule):
    data_name = "synthetic-teacher"
    image_size = 32

    def __init__(
        self,
        *,
        batch_size: int = 64,
        eval_batch_size: int | None = None,
        num_workers: int | None = None,
        add_rand_aug: bool = False,
        input_noise_std: float = 0.0,
        num_train_examples: int = 4096,
        num_test_examples: int = 2048,
        input_dim: int = 128,
        num_classes: int = 10,
        target_type: str = "classification",
        target_dim: int = 1,
        teacher_depth: int = 3,
        teacher_width: int = 256,
        teacher_seed: int = 123,
        train_seed: int = 1000,
        test_seed: int = 2000,
        activation: str = "gelu",
        label_noise: float = 0.0,
        target_noise_std: float = 0.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.add_rand_aug = add_rand_aug
        self.input_noise_std = input_noise_std
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples
        self.input_dim = input_dim
        self.task_type = "regression" if target_type == "continuous" else "classification"
        self.target_type = target_type
        self.num_labels = target_dim if target_type == "continuous" else num_classes
        self.teacher_depth = teacher_depth
        self.teacher_width = teacher_width
        self.teacher_seed = teacher_seed
        self.train_seed = train_seed
        self.test_seed = test_seed
        self.activation = activation
        self.label_noise = label_noise
        self.target_noise_std = target_noise_std
        self.datasets = {}
        self.train_key = "train"
        self.val_key = None
        self.test_key = "test"

    def prepare_data(self):
        return None

    def _build_dataset(self, *, num_examples: int, data_seed: int):
        return SyntheticTeacherDataset(
            num_examples=num_examples,
            input_dim=self.input_dim,
            image_size=self.image_size,
            num_labels=self.num_labels,
            target_type=self.target_type,
            teacher_depth=self.teacher_depth,
            teacher_width=self.teacher_width,
            teacher_seed=self.teacher_seed,
            data_seed=data_seed,
            activation=self.activation,
            label_noise=self.label_noise,
            target_noise_std=self.target_noise_std,
            input_noise_std=self.input_noise_std,
        )

    def setup(self, stage=None):
        train_data = self._build_dataset(
            num_examples=self.num_train_examples,
            data_seed=self.train_seed,
        )
        test_data = self._build_dataset(
            num_examples=self.num_test_examples,
            data_seed=self.test_seed,
        )
        self.datasets["train"] = train_data
        self.datasets["train_eval"] = train_data
        self.datasets["test"] = test_data

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def train_eval_dataloader(self):
        return DataLoader(
            self.datasets["train_eval"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )


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
    elif args.data_name == "synthetic-teacher":
        return SyntheticTeacherDataModule(
            **data_kwargs,
            num_train_examples=args.synthetic_num_train,
            num_test_examples=args.synthetic_num_test,
            input_dim=args.synthetic_input_dim,
            num_classes=args.synthetic_num_classes,
            target_type=args.synthetic_target_type,
            target_dim=args.synthetic_target_dim,
            teacher_depth=args.synthetic_teacher_depth,
            teacher_width=args.synthetic_teacher_width,
            teacher_seed=args.synthetic_teacher_seed,
            train_seed=args.synthetic_train_seed,
            test_seed=args.synthetic_test_seed,
            activation=args.synthetic_teacher_activation,
            label_noise=args.synthetic_label_noise,
            target_noise_std=args.synthetic_target_noise_std,
        )
    else:
        raise ValueError(f"Invalid data name: {args.data_name}")
