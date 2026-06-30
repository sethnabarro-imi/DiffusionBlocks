import argparse
import csv
import datetime as dt
import json
import math
import os
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def target_function(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "sine_poly":
        return torch.sin(3.0 * x) + 0.25 * x.square()
    if name == "saw_sine":
        return 0.8 * torch.sin(5.0 * x) + 0.35 * torch.sign(torch.sin(1.5 * x))
    if name == "cubic":
        return 0.15 * x.pow(3) - 0.5 * x + torch.sin(2.0 * x)
    raise ValueError(f"Unknown target function: {name}")


@dataclass
class ToyConfig:
    num_train: int = 512
    num_test: int = 512
    num_blocks: int = 6
    latent_dim: int = 32
    hidden_dim: int = 128
    depth: int = 3
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    sigma_max: float = 4.0
    sigma_min: float = 0.02
    sigma_schedule: str = "log"
    prediction_loss_weight: float = 1.0
    latent_loss_weight: float = 0.1
    training_objective: str = "clean_latent"
    train_encoder_with_prediction_loss_only: bool = False
    observation_noise_std: float = 0.0
    function: str = "sine_poly"
    seed: int = 0
    output_dir: str = "results/toy_1d_regression"
    eval_every: int = 100
    prediction_grid_points: int = 256
    device: str = "auto"
    no_plots: bool = False


def make_dataset(num_examples: int, config: ToyConfig, seed_offset: int):
    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    x = -3.0 + 6.0 * torch.rand(num_examples, 1, generator=generator)
    y = target_function(x, config.function)
    if config.observation_noise_std > 0.0:
        y = y + config.observation_noise_std * torch.randn(
            y.shape,
            generator=generator,
        )
    return TensorDataset(x.float(), y.float())


def sigma_schedule(config: ToyConfig, device: torch.device) -> torch.Tensor:
    if config.sigma_schedule == "linear":
        return torch.linspace(
            config.sigma_max,
            config.sigma_min,
            config.num_blocks,
            device=device,
        )
    if config.sigma_schedule != "log":
        raise ValueError(f"Unknown sigma schedule: {config.sigma_schedule}")
    return torch.exp(
        torch.linspace(
            math.log(config.sigma_max),
            math.log(config.sigma_min),
            config.num_blocks,
            device=device,
        )
    )


class DenoisingBlock(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers = []
        in_dim = 1 + latent_dim + 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma = sigma.view(-1, 1)
        features = torch.cat([x, z, sigma.log(), sigma], dim=-1)
        return self.net(features)


class ToyDiffusionBlocks(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.target_encoder = nn.Sequential(
            nn.Linear(1, config.latent_dim),
            nn.Tanh(),
        )
        self.target_decoder = nn.Linear(config.latent_dim, 1)
        self.blocks = nn.ModuleList(
            [
                DenoisingBlock(config.latent_dim, config.hidden_dim, config.depth)
                for _ in range(config.num_blocks)
            ]
        )

    def clean_latent(self, y: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(y)

    def predict_from_latent(self, z_clean_pred: torch.Tensor) -> torch.Tensor:
        return self.target_decoder(z_clean_pred)

    def euler_update(
        self,
        z: torch.Tensor,
        z_clean_pred: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
    ) -> torch.Tensor:
        d = (z - z_clean_pred) / sigma[:, None]
        return z + (next_sigma - sigma)[:, None] * d

    def uses_residual_next_latent_objective(self) -> bool:
        return self.config.training_objective == "residual_next_latent"

    def prediction_loss_weights(self, reference: torch.Tensor) -> torch.Tensor:
        weights = torch.full(
            (self.config.num_blocks,),
            self.config.prediction_loss_weight,
            device=reference.device,
            dtype=reference.dtype,
        )
        weights[-1] = torch.maximum(weights[-1], weights.new_tensor(1.0))
        return weights

    def latent_target_for_loss(self, z_clean: torch.Tensor) -> torch.Tensor:
        if self.config.train_encoder_with_prediction_loss_only:
            return z_clean.detach()
        return z_clean

    def encoder_prediction_loss(self, z_clean: torch.Tensor, y: torch.Tensor):
        if not self.config.train_encoder_with_prediction_loss_only:
            return z_clean.new_zeros(())
        encoder_prediction = self.predict_from_latent(z_clean)
        return F.mse_loss(encoder_prediction, y)

    def sequential_predictions(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ):
        batch_size = x.shape[0]
        if noise is None:
            z = torch.randn(
                batch_size,
                self.config.latent_dim,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            z = noise.to(device=x.device, dtype=x.dtype)
        z = z * torch.sqrt(1.0 + sigmas[0] ** 2)

        predictions = []
        latents = []
        for block_index, block in enumerate(self.blocks):
            sigma = sigmas[block_index].expand(batch_size)
            block_output = block(x, z, sigma)
            if self.uses_residual_next_latent_objective():
                z_next_pred = z + block_output
                y_pred = self.predict_from_latent(z_next_pred)
                latent_prediction = z_next_pred
            else:
                y_pred = self.predict_from_latent(block_output)
                latent_prediction = block_output
            predictions.append(y_pred)
            latents.append(latent_prediction)
            if block_index < len(self.blocks) - 1:
                next_sigma = sigmas[block_index + 1].expand(batch_size)
                if self.uses_residual_next_latent_objective():
                    z = z_next_pred.detach()
                else:
                    z = self.euler_update(z, block_output, sigma, next_sigma).detach()
        return torch.stack(predictions), torch.stack(latents)

    def oracle_predictions(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ):
        z_clean = self.clean_latent(y)
        predictions = []
        latents = []
        for block_index, block in enumerate(self.blocks):
            sigma = sigmas[block_index].expand(x.shape[0])
            if noise is None:
                epsilon = torch.randn_like(z_clean)
            else:
                epsilon = noise.to(device=x.device, dtype=x.dtype)
            z_noisy = z_clean + sigma[:, None] * epsilon
            block_output = block(x, z_noisy, sigma)
            if self.uses_residual_next_latent_objective():
                z_next_pred = z_noisy + block_output
                y_pred = self.predict_from_latent(z_next_pred)
                latent_prediction = z_next_pred
            else:
                y_pred = self.predict_from_latent(block_output)
                latent_prediction = block_output
            predictions.append(y_pred)
            latents.append(latent_prediction)
        return torch.stack(predictions), torch.stack(latents)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor):
        if self.uses_residual_next_latent_objective():
            return self.residual_next_latent_training_loss(x, y, sigmas)

        z_clean = self.clean_latent(y)
        z_clean_target = self.latent_target_for_loss(z_clean)
        predictions, latents = self.sequential_predictions(x, sigmas)
        mse_losses = F.mse_loss(
            predictions,
            y.unsqueeze(0).expand_as(predictions),
            reduction="none",
        ).mean(dim=(1, 2))
        latent_losses = F.mse_loss(
            latents,
            z_clean_target.unsqueeze(0).expand_as(latents),
            reduction="none",
        ).mean(dim=(1, 2))
        encoder_prediction_loss = self.encoder_prediction_loss(z_clean, y)
        encoder_prediction_losses = torch.zeros_like(mse_losses)
        encoder_prediction_losses[-1] = encoder_prediction_loss
        per_block_losses = (
            self.prediction_loss_weights(mse_losses) * mse_losses
            + self.config.latent_loss_weight * latent_losses
            + encoder_prediction_losses
        )
        loss = per_block_losses.sum()
        return loss, {
            "mse_mean": mse_losses.mean().detach(),
            "latent_mse_mean": latent_losses.mean().detach(),
            "encoder_prediction_loss": encoder_prediction_loss.detach(),
            "per_block_loss": per_block_losses.detach(),
            "per_block_mse_loss": mse_losses.detach(),
            "per_block_latent_mse_loss": latent_losses.detach(),
            "per_block_encoder_prediction_loss": (
                encoder_prediction_losses.detach()
            ),
        }

    def residual_next_latent_training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
    ):
        batch_size = x.shape[0]
        z_clean = self.clean_latent(y)
        z = torch.randn(
            batch_size,
            self.config.latent_dim,
            device=x.device,
            dtype=x.dtype,
        )
        z = z * torch.sqrt(1.0 + sigmas[0] ** 2)

        predictions = []
        residual_losses = []
        for block_index, block in enumerate(self.blocks):
            sigma = sigmas[block_index].expand(batch_size)
            next_sigma = (
                sigmas[block_index + 1].expand(batch_size)
                if block_index < len(self.blocks) - 1
                else torch.zeros_like(sigma)
            )
            z_input = z.detach()
            delta_pred = block(x, z_input, sigma)
            z_next_pred = z_input + delta_pred
            z_clean_target = self.latent_target_for_loss(z_clean)
            z_next_target = z_clean_target + (next_sigma / sigma)[:, None] * (
                z_input - z_clean_target
            )
            delta_target = z_next_target - z_input
            residual_losses.append(
                F.mse_loss(delta_pred, delta_target, reduction="none").mean(dim=(0, 1))
            )
            predictions.append(self.predict_from_latent(z_next_pred))
            z = z_next_pred.detach()

        predictions = torch.stack(predictions)
        residual_losses = torch.stack(residual_losses)
        mse_losses = F.mse_loss(
            predictions,
            y.unsqueeze(0).expand_as(predictions),
            reduction="none",
        ).mean(dim=(1, 2))
        encoder_prediction_loss = self.encoder_prediction_loss(z_clean, y)
        encoder_prediction_losses = torch.zeros_like(mse_losses)
        encoder_prediction_losses[-1] = encoder_prediction_loss
        per_block_losses = (
            self.prediction_loss_weights(mse_losses) * mse_losses
            + self.config.latent_loss_weight * residual_losses
            + encoder_prediction_losses
        )
        loss = per_block_losses.sum()
        return loss, {
            "mse_mean": mse_losses.mean().detach(),
            "latent_mse_mean": residual_losses.mean().detach(),
            "encoder_prediction_loss": encoder_prediction_loss.detach(),
            "per_block_loss": per_block_losses.detach(),
            "per_block_mse_loss": mse_losses.detach(),
            "per_block_latent_mse_loss": residual_losses.detach(),
            "per_block_encoder_prediction_loss": (
                encoder_prediction_losses.detach()
            ),
        }


def evaluate_model(
    model: ToyDiffusionBlocks,
    dataloader: DataLoader,
    sigmas: torch.Tensor,
    split: str,
):
    model.eval()
    sequential_sse = torch.zeros(model.config.num_blocks, device=sigmas.device)
    oracle_sse = torch.zeros(model.config.num_blocks, device=sigmas.device)
    sequential_latent_sse = torch.zeros(model.config.num_blocks, device=sigmas.device)
    oracle_latent_sse = torch.zeros(model.config.num_blocks, device=sigmas.device)
    count = 0
    latent_count = 0
    final_disagreement_sse = torch.zeros(model.config.num_blocks, device=sigmas.device)
    previous_change_sse = torch.zeros(model.config.num_blocks, device=sigmas.device)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(sigmas.device)
            y = y.to(sigmas.device)
            z_clean = model.clean_latent(y)
            seq_preds, seq_latents = model.sequential_predictions(x, sigmas)
            oracle_preds, oracle_latents = model.oracle_predictions(x, y, sigmas)
            target = y.unsqueeze(0).expand_as(seq_preds)
            latent_target = z_clean.unsqueeze(0).expand_as(seq_latents)
            sequential_sse += (seq_preds - target).square().sum(dim=(1, 2))
            oracle_sse += (oracle_preds - target).square().sum(dim=(1, 2))
            sequential_latent_sse += (
                seq_latents - latent_target
            ).square().sum(dim=(1, 2))
            oracle_latent_sse += (
                oracle_latents - latent_target
            ).square().sum(dim=(1, 2))
            final = seq_preds[-1].unsqueeze(0).expand_as(seq_preds)
            final_disagreement_sse += (seq_preds - final).square().sum(dim=(1, 2))
            changes = torch.zeros_like(seq_preds)
            changes[1:] = seq_preds[1:] - seq_preds[:-1]
            previous_change_sse += changes.square().sum(dim=(1, 2))
            count += y.numel()
            latent_count += z_clean.numel()

    rows = []
    for block_index in range(model.config.num_blocks):
        rows.append(
            {
                "split": split,
                "block_index": block_index,
                "sigma": float(sigmas[block_index].detach().cpu()),
                "sequential_mse": float(
                    (sequential_sse[block_index] / count).detach().cpu()
                ),
                "sequential_rmse": float(
                    (sequential_sse[block_index] / count).sqrt().detach().cpu()
                ),
                "oracle_mse": float((oracle_sse[block_index] / count).detach().cpu()),
                "oracle_rmse": float(
                    (oracle_sse[block_index] / count).sqrt().detach().cpu()
                ),
                "sequential_latent_mse": float(
                    (sequential_latent_sse[block_index] / latent_count)
                    .detach()
                    .cpu()
                ),
                "sequential_latent_rmse": float(
                    (sequential_latent_sse[block_index] / latent_count)
                    .sqrt()
                    .detach()
                    .cpu()
                ),
                "oracle_latent_mse": float(
                    (oracle_latent_sse[block_index] / latent_count).detach().cpu()
                ),
                "oracle_latent_rmse": float(
                    (oracle_latent_sse[block_index] / latent_count)
                    .sqrt()
                    .detach()
                    .cpu()
                ),
                "final_mse": float(
                    (final_disagreement_sse[block_index] / count).detach().cpu()
                ),
                "changed_from_previous_mse": float(
                    (previous_change_sse[block_index] / count).detach().cpu()
                ),
            }
        )
    return rows


def evaluate_prediction_curves(
    model: ToyDiffusionBlocks,
    config: ToyConfig,
    sigmas: torch.Tensor,
    epoch: int,
) -> list[dict]:
    model.eval()
    x = torch.linspace(
        -3.2,
        3.2,
        config.prediction_grid_points,
        device=sigmas.device,
    ).view(-1, 1)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 50_000)
    noise = torch.randn(
        config.prediction_grid_points,
        config.latent_dim,
        generator=generator,
        device="cpu",
    ).to(sigmas.device)
    with torch.no_grad():
        target = target_function(x, config.function)
        predictions, _ = model.sequential_predictions(x, sigmas, noise=noise)

    rows = []
    cpu_x = x[:, 0].detach().cpu().tolist()
    cpu_target = target[:, 0].detach().cpu().tolist()
    cpu_predictions = predictions[:, :, 0].detach().cpu().tolist()
    for block_index, block_predictions in enumerate(cpu_predictions):
        for point_index, (x_value, target_value, prediction_value) in enumerate(
            zip(cpu_x, cpu_target, block_predictions)
        ):
            rows.append(
                {
                    "epoch": epoch,
                    "block_index": block_index,
                    "point_index": point_index,
                    "x": x_value,
                    "target_y": target_value,
                    "sequential_prediction": prediction_value,
                }
            )
    return rows


def write_rows_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        split_rows = sorted(split_rows, key=lambda row: row["block_index"])
        x = [row["block_index"] for row in split_rows]
        ax.plot(
            x,
            [row["sequential_rmse"] for row in split_rows],
            marker="o",
            label=f"{split}: sequential",
        )
        ax.plot(
            x,
            [row["oracle_rmse"] for row in split_rows],
            marker="s",
            linestyle="--",
            label=f"{split}: oracle",
        )
    ax.set_xlabel("Block index")
    ax.set_ylabel("RMSE")
    ax.set_title("Toy 1D DBlock Regression")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    return True


def write_latent_distance_plot(rows: list[dict], path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        split_rows = sorted(split_rows, key=lambda row: row["block_index"])
        x = [row["block_index"] for row in split_rows]
        ax.plot(
            x,
            [row["sequential_latent_rmse"] for row in split_rows],
            marker="o",
            label=f"{split}: sequential",
        )
        ax.plot(
            x,
            [row["oracle_latent_rmse"] for row in split_rows],
            marker="s",
            linestyle="--",
            label=f"{split}: oracle",
        )
    ax.set_xlabel("Block index")
    ax.set_ylabel("Latent RMSE to clean")
    ax.set_title("Toy 1D DBlock Latent Distance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    return True


def epoch_color(index: int, count: int) -> tuple[int, int, int]:
    # Blue-to-red continuous ramp for eval epoch ordering.
    t = index / max(count - 1, 1)
    return (
        int(37 * (1 - t) + 220 * t),
        int(99 * (1 - t) + 38 * t),
        int(235 * (1 - t) + 38 * t),
    )


def write_eval_cycle_rmse_csv(history: list[dict], path: str) -> None:
    rows = []
    for item in history:
        epoch = item["epoch"]
        for row in sorted(item["test_block_rows"], key=lambda r: r["block_index"]):
            rows.append(
                {
                    "epoch": epoch,
                    "block_index": row["block_index"],
                    "test_sequential_rmse": row["sequential_rmse"],
                }
            )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "block_index", "test_sequential_rmse"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_eval_cycle_latent_distance_csv(history: list[dict], path: str) -> None:
    rows = []
    for item in history:
        epoch = item["epoch"]
        for row in sorted(item["test_block_rows"], key=lambda r: r["block_index"]):
            rows.append(
                {
                    "epoch": epoch,
                    "block_index": row["block_index"],
                    "test_sequential_latent_rmse": row["sequential_latent_rmse"],
                    "test_oracle_latent_rmse": row["oracle_latent_rmse"],
                }
            )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "block_index",
                "test_sequential_latent_rmse",
                "test_oracle_latent_rmse",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_eval_cycle_train_loss_csv(history: list[dict], path: str) -> None:
    rows = [
        row
        for item in history
        for row in item.get("train_loss_rows", [])
    ]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "block_index",
                "sigma",
                "train_loss",
                "train_mse_loss",
                "train_latent_mse_loss",
                "train_encoder_prediction_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_curve_csv(history: list[dict], path: str) -> None:
    rows = [
        row
        for item in history
        for row in item.get("prediction_curve_rows", [])
    ]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "block_index",
                "point_index",
                "x",
                "target_y",
                "sequential_prediction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_eval_cycle_rmse_plot(
    history: list[dict],
    png_path: str,
    svg_path: str,
    *,
    y_max: float = 0.6,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    series = []
    for item in history:
        series.append(
            {
                "epoch": item["epoch"],
                "points": [
                    (int(row["block_index"]), float(row["sequential_rmse"]))
                    for row in sorted(
                        item["test_block_rows"], key=lambda row: row["block_index"]
                    )
                ],
            }
        )
    if not series:
        return False

    all_blocks = [block for item in series for block, _ in item["points"]]
    x_min, x_max = min(all_blocks), max(all_blocks)
    y_min = 0.0
    width, height = 1000, 650
    left, right, top, bottom = 90, 230, 58, 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        value = max(y_min, min(y_max, value))
        return top + (y_max - value) / (y_max - y_min) * plot_h

    def points_for(item: dict) -> list[tuple[float, float]]:
        return [(sx(block), sy(rmse)) for block, rmse in item["points"]]

    def epoch_tick_indices() -> list[int]:
        if len(series) == 1:
            return [0]
        return sorted({0, len(series) // 2, len(series) - 1})

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 13
        )
        font_bold = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16
        )
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22
        )
    except OSError:
        font = font_bold = title_font = ImageFont.load_default()

    title = f"Test RMSE by Block Across Eval Cycles (y-axis 0-{y_max:g})"
    draw.text((left, 24), title, font=title_font, fill=(17, 24, 39))
    for idx in range(7):
        value = y_min + (y_max - y_min) * idx / 6
        yy = sy(value)
        draw.line((left, yy, left + plot_w, yy), fill=(229, 231, 235), width=1)
        draw.text(
            (left - 14, yy - 7),
            f"{value:.2f}",
            font=font,
            fill=(17, 24, 39),
            anchor="ra",
        )
    for block in range(x_min, x_max + 1):
        xx = sx(block)
        draw.line((xx, top, xx, top + plot_h), fill=(229, 231, 235), width=1)
        draw.text(
            (xx, top + plot_h + 14),
            str(block),
            font=font,
            fill=(17, 24, 39),
            anchor="ma",
        )
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(17, 24, 39), width=2)
    draw.line((left, top, left, top + plot_h), fill=(17, 24, 39), width=2)
    draw.text(
        (left + plot_w / 2, height - 38),
        "Block id",
        font=font_bold,
        fill=(17, 24, 39),
        anchor="ma",
    )
    label = Image.new("RGBA", (260, 30), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text(
        (130, 15),
        "Test sequential RMSE",
        font=font_bold,
        fill=(17, 24, 39),
        anchor="mm",
    )
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, (18, int(top + plot_h / 2 - 130)), rotated)

    for idx, item in enumerate(series):
        color = epoch_color(idx, len(series))
        pts = points_for(item)
        draw.line(pts, fill=color, width=2, joint="curve")
        for x, y in pts:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    colorbar_x = left + plot_w + 48
    colorbar_y = top + 40
    colorbar_w = 24
    colorbar_h = plot_h - 80
    draw.text(
        (colorbar_x - 6, colorbar_y - 30),
        "Epoch",
        font=font_bold,
        fill=(17, 24, 39),
    )
    for offset in range(colorbar_h):
        t = offset / max(colorbar_h - 1, 1)
        color = epoch_color(round(t * (len(series) - 1)), len(series))
        y = colorbar_y + offset
        draw.line((colorbar_x, y, colorbar_x + colorbar_w, y), fill=color)
    draw.rectangle(
        (
            colorbar_x,
            colorbar_y,
            colorbar_x + colorbar_w,
            colorbar_y + colorbar_h,
        ),
        outline=(17, 24, 39),
        width=1,
    )
    for idx in epoch_tick_indices():
        t = idx / max(len(series) - 1, 1)
        y = colorbar_y + t * colorbar_h
        draw.line(
            (colorbar_x + colorbar_w, y, colorbar_x + colorbar_w + 7, y),
            fill=(17, 24, 39),
            width=1,
        )
        draw.text(
            (colorbar_x + colorbar_w + 12, y - 7),
            str(series[idx]["epoch"]),
            font=font,
            fill=(17, 24, 39),
        )
    image.save(png_path)

    def svg_color(idx: int) -> str:
        red, green, blue = epoch_color(idx, len(series))
        return f"rgb({red},{green},{blue})"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.axis{stroke:#111827;stroke-width:1.4}.grid{stroke:#e5e7eb;stroke-width:1}.tick{font-size:13px}.label{font-size:16px;font-weight:600}.title{font-size:22px;font-weight:700}.legend{font-size:12px}</style>',
        f'<text class="title" x="{left}" y="32">{title}</text>',
    ]
    for idx in range(7):
        value = y_min + (y_max - y_min) * idx / 6
        yy = sy(value)
        svg.append(
            f'<line class="grid" x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}"/>'
        )
        svg.append(
            f'<text class="tick" x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end">{value:.2f}</text>'
        )
    for block in range(x_min, x_max + 1):
        xx = sx(block)
        svg.append(
            f'<line class="grid" x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top + plot_h}"/>'
        )
        svg.append(
            f'<text class="tick" x="{xx:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{block}</text>'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<text class="label" x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle">Block id</text>',
            f'<text class="label" transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Test sequential RMSE</text>',
        ]
    )
    for idx, item in enumerate(series):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points_for(item))
        color = svg_color(idx)
        svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.0" opacity="0.82"/>'
        )
        for x, y in points_for(item):
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" opacity="0.9"/>'
            )
    colorbar_x = left + plot_w + 48
    colorbar_y = top + 40
    colorbar_w = 24
    colorbar_h = plot_h - 80
    gradient_id = "epoch-gradient"
    svg.append("<defs>")
    svg.append(
        f'<linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="0%" y2="100%">'
    )
    for idx in range(len(series)):
        offset = 100 * idx / max(len(series) - 1, 1)
        svg.append(
            f'<stop offset="{offset:.2f}%" stop-color="{svg_color(idx)}"/>'
        )
    svg.append("</linearGradient>")
    svg.append("</defs>")
    svg.append(
        f'<text class="label" x="{colorbar_x - 6}" y="{colorbar_y - 30}">Epoch</text>'
    )
    svg.append(
        f'<rect x="{colorbar_x}" y="{colorbar_y}" width="{colorbar_w}" height="{colorbar_h}" fill="url(#{gradient_id})" stroke="#111827" stroke-width="1"/>'
    )
    for idx in epoch_tick_indices():
        tick_y = colorbar_y + idx / max(len(series) - 1, 1) * colorbar_h
        svg.append(
            f'<line x1="{colorbar_x + colorbar_w}" y1="{tick_y:.2f}" x2="{colorbar_x + colorbar_w + 7}" y2="{tick_y:.2f}" stroke="#111827" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="legend" x="{colorbar_x + colorbar_w + 12}" y="{tick_y + 4:.2f}">{series[idx]["epoch"]}</text>'
        )
    svg.append("</svg>")
    with open(svg_path, "w") as f:
        f.write("\n".join(svg))
        f.write("\n")
    return True


def write_eval_cycle_latent_distance_plot(
    history: list[dict],
    png_path: str,
    svg_path: str,
) -> bool:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rows = []
    for item in history:
        for row in item.get("test_block_rows", []):
            rows.append(
                {
                    "epoch": int(item["epoch"]),
                    "block_index": int(row["block_index"]),
                    "sequential_latent_rmse": float(row["sequential_latent_rmse"]),
                    "oracle_latent_rmse": float(row["oracle_latent_rmse"]),
                }
            )
    if not rows:
        return False

    epochs = sorted({row["epoch"] for row in rows})
    norm = mpl.colors.Normalize(vmin=min(epochs), vmax=max(epochs))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(8, 5))
    for epoch in epochs:
        group = sorted(
            [row for row in rows if row["epoch"] == epoch],
            key=lambda row: row["block_index"],
        )
        ax.plot(
            [row["block_index"] for row in group],
            [row["sequential_latent_rmse"] for row in group],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=cmap(norm(epoch)),
        )
    ax.set_xlabel("Block index")
    ax.set_ylabel("Test sequential latent RMSE to clean")
    ax.set_title("Test Latent Distance by Block Across Eval Cycles")
    ax.grid(True, alpha=0.25)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Epoch")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return True


def write_eval_cycle_train_loss_plot(
    history: list[dict],
    png_path: str,
    svg_path: str,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    rows = [
        row
        for item in history
        for row in item.get("train_loss_rows", [])
    ]
    if not rows:
        return False

    epochs = sorted({int(row["epoch"]) for row in rows})
    blocks = sorted({int(row["block_index"]) for row in rows})
    by_block = {block: [] for block in blocks}
    for row in rows:
        by_block[int(row["block_index"])].append(
            (int(row["epoch"]), float(row["train_loss"]))
        )
    for block in blocks:
        by_block[block].sort(key=lambda point: point[0])

    x_min, x_max = min(epochs), max(epochs)
    y_min = 0.0
    y_max = max(loss for points in by_block.values() for _, loss in points)
    y_max = max(y_max * 1.08, 1e-6)

    width, height = 1000, 650
    left, right, top, bottom = 100, 230, 58, 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(value: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        value = max(y_min, min(y_max, value))
        return top + (y_max - value) / (y_max - y_min) * plot_h

    def points_for(block: int) -> list[tuple[float, float]]:
        return [(sx(epoch), sy(loss)) for epoch, loss in by_block[block]]

    def block_color(index: int) -> tuple[int, int, int]:
        return epoch_color(index, len(blocks))

    def block_index(block: int) -> int:
        return blocks.index(block)

    def x_tick_values() -> list[int]:
        if len(epochs) <= 6:
            return epochs
        return sorted({epochs[0], epochs[len(epochs) // 4], epochs[len(epochs) // 2], epochs[3 * len(epochs) // 4], epochs[-1]})

    def block_tick_indices() -> list[int]:
        if len(blocks) == 1:
            return [0]
        return sorted({0, len(blocks) // 2, len(blocks) - 1})

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 13
        )
        font_bold = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16
        )
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22
        )
    except OSError:
        font = font_bold = title_font = ImageFont.load_default()

    title = "Per-block Train Loss Across Eval Cycles"
    draw.text((left, 24), title, font=title_font, fill=(17, 24, 39))
    for idx in range(7):
        value = y_min + (y_max - y_min) * idx / 6
        yy = sy(value)
        draw.line((left, yy, left + plot_w, yy), fill=(229, 231, 235), width=1)
        draw.text(
            (left - 14, yy - 7),
            f"{value:.3g}",
            font=font,
            fill=(17, 24, 39),
            anchor="ra",
        )
    for epoch in x_tick_values():
        xx = sx(epoch)
        draw.line((xx, top, xx, top + plot_h), fill=(229, 231, 235), width=1)
        draw.text(
            (xx, top + plot_h + 14),
            str(epoch),
            font=font,
            fill=(17, 24, 39),
            anchor="ma",
        )
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(17, 24, 39), width=2)
    draw.line((left, top, left, top + plot_h), fill=(17, 24, 39), width=2)
    draw.text(
        (left + plot_w / 2, height - 38),
        "Epoch",
        font=font_bold,
        fill=(17, 24, 39),
        anchor="ma",
    )
    label = Image.new("RGBA", (260, 30), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text(
        (130, 15),
        "Train loss",
        font=font_bold,
        fill=(17, 24, 39),
        anchor="mm",
    )
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, (18, int(top + plot_h / 2 - 130)), rotated)

    for block in blocks:
        color = block_color(block_index(block))
        pts = points_for(block)
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2, joint="curve")
        for x, y in pts:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    colorbar_x = left + plot_w + 48
    colorbar_y = top + 40
    colorbar_w = 24
    colorbar_h = plot_h - 80
    draw.text(
        (colorbar_x - 6, colorbar_y - 30),
        "Block",
        font=font_bold,
        fill=(17, 24, 39),
    )
    for offset in range(colorbar_h):
        t = offset / max(colorbar_h - 1, 1)
        color = block_color(round(t * (len(blocks) - 1)))
        y = colorbar_y + offset
        draw.line((colorbar_x, y, colorbar_x + colorbar_w, y), fill=color)
    draw.rectangle(
        (
            colorbar_x,
            colorbar_y,
            colorbar_x + colorbar_w,
            colorbar_y + colorbar_h,
        ),
        outline=(17, 24, 39),
        width=1,
    )
    for idx in block_tick_indices():
        t = idx / max(len(blocks) - 1, 1)
        y = colorbar_y + t * colorbar_h
        draw.line(
            (colorbar_x + colorbar_w, y, colorbar_x + colorbar_w + 7, y),
            fill=(17, 24, 39),
            width=1,
        )
        draw.text(
            (colorbar_x + colorbar_w + 12, y - 7),
            str(blocks[idx]),
            font=font,
            fill=(17, 24, 39),
        )
    image.save(png_path)

    def svg_color(idx: int) -> str:
        red, green, blue = block_color(idx)
        return f"rgb({red},{green},{blue})"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.axis{stroke:#111827;stroke-width:1.4}.grid{stroke:#e5e7eb;stroke-width:1}.tick{font-size:13px}.label{font-size:16px;font-weight:600}.title{font-size:22px;font-weight:700}.legend{font-size:12px}</style>',
        f'<text class="title" x="{left}" y="32">{title}</text>',
    ]
    for idx in range(7):
        value = y_min + (y_max - y_min) * idx / 6
        yy = sy(value)
        svg.append(
            f'<line class="grid" x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}"/>'
        )
        svg.append(
            f'<text class="tick" x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end">{value:.3g}</text>'
        )
    for epoch in x_tick_values():
        xx = sx(epoch)
        svg.append(
            f'<line class="grid" x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top + plot_h}"/>'
        )
        svg.append(
            f'<text class="tick" x="{xx:.2f}" y="{top + plot_h + 24}" text-anchor="middle">{epoch}</text>'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<text class="label" x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle">Epoch</text>',
            f'<text class="label" transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Train loss</text>',
        ]
    )
    for block in blocks:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points_for(block))
        color = svg_color(block_index(block))
        if len(by_block[block]) > 1:
            svg.append(
                f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.0" opacity="0.82"/>'
            )
        for x, y in points_for(block):
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" opacity="0.9"/>'
            )
    colorbar_x = left + plot_w + 48
    colorbar_y = top + 40
    colorbar_w = 24
    colorbar_h = plot_h - 80
    gradient_id = "block-gradient"
    svg.append("<defs>")
    svg.append(
        f'<linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="0%" y2="100%">'
    )
    for idx in range(len(blocks)):
        offset = 100 * idx / max(len(blocks) - 1, 1)
        svg.append(
            f'<stop offset="{offset:.2f}%" stop-color="{svg_color(idx)}"/>'
        )
    svg.append("</linearGradient>")
    svg.append("</defs>")
    svg.append(
        f'<text class="label" x="{colorbar_x - 6}" y="{colorbar_y - 30}">Block</text>'
    )
    svg.append(
        f'<rect x="{colorbar_x}" y="{colorbar_y}" width="{colorbar_w}" height="{colorbar_h}" fill="url(#{gradient_id})" stroke="#111827" stroke-width="1"/>'
    )
    for idx in block_tick_indices():
        tick_y = colorbar_y + idx / max(len(blocks) - 1, 1) * colorbar_h
        svg.append(
            f'<line x1="{colorbar_x + colorbar_w}" y1="{tick_y:.2f}" x2="{colorbar_x + colorbar_w + 7}" y2="{tick_y:.2f}" stroke="#111827" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="legend" x="{colorbar_x + colorbar_w + 12}" y="{tick_y + 4:.2f}">{blocks[idx]}</text>'
        )
    svg.append("</svg>")
    with open(svg_path, "w") as f:
        f.write("\n".join(svg))
        f.write("\n")
    return True


def write_prediction_curve_eval_cycle_plot(
    history: list[dict],
    train_data: TensorDataset,
    test_data: TensorDataset,
    png_path: str,
    svg_path: str,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    rows = [
        row
        for item in history
        for row in item.get("prediction_curve_rows", [])
    ]
    if not rows:
        return False

    epochs = sorted({int(row["epoch"]) for row in rows})
    blocks = sorted({int(row["block_index"]) for row in rows})
    by_block_epoch = {
        (block, epoch): []
        for block in blocks
        for epoch in epochs
    }
    target_by_block = {block: [] for block in blocks}
    for row in rows:
        key = (int(row["block_index"]), int(row["epoch"]))
        by_block_epoch[key].append(
            (
                float(row["x"]),
                float(row["sequential_prediction"]),
            )
        )
        if int(row["epoch"]) == epochs[0]:
            target_by_block[int(row["block_index"])].append(
                (float(row["x"]), float(row["target_y"]))
            )

    train_x, train_y = train_data.tensors
    test_x, test_y = test_data.tensors
    train_points = [
        (float(x), float(y))
        for x, y in zip(train_x[:, 0].tolist(), train_y[:, 0].tolist())
    ]
    test_points = [
        (float(x), float(y))
        for x, y in zip(test_x[:, 0].tolist(), test_y[:, 0].tolist())
    ]

    x_values = [float(row["x"]) for row in rows] + [
        x for x, _ in train_points + test_points
    ]
    y_values = [float(row["target_y"]) for row in rows] + [
        float(row["sequential_prediction"]) for row in rows
    ] + [
        y for _, y in train_points + test_points
    ]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    y_pad = 0.08 * max(y_max - y_min, 1e-6)
    y_min -= y_pad
    y_max += y_pad

    columns = min(3, len(blocks))
    panel_rows = math.ceil(len(blocks) / columns)
    panel_w = 300
    panel_h = 220
    left = 72
    top = 72
    gap_x = 34
    gap_y = 46
    right = 150
    bottom = 64
    width = left + columns * panel_w + (columns - 1) * gap_x + right
    height = top + panel_rows * panel_h + (panel_rows - 1) * gap_y + bottom

    def epoch_index(epoch: int) -> int:
        return epochs.index(epoch)

    def epoch_tick_indices() -> list[int]:
        if len(epochs) == 1:
            return [0]
        return sorted({0, len(epochs) // 2, len(epochs) - 1})

    def sx(x: float, panel_left: float) -> float:
        return panel_left + (x - x_min) / (x_max - x_min) * panel_w

    def sy(y: float, panel_top: float) -> float:
        return panel_top + (y_max - y) / (y_max - y_min) * panel_h

    def panel_origin(block_position: int) -> tuple[int, int]:
        row = block_position // columns
        col = block_position % columns
        return (
            left + col * (panel_w + gap_x),
            top + row * (panel_h + gap_y),
        )

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 12
        )
        font_bold = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15
        )
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22
        )
    except OSError:
        font = font_bold = title_font = ImageFont.load_default()

    draw.text(
        (left, 26),
        "Per-block Sequential Predictions Across Eval Cycles",
        font=title_font,
        fill=(17, 24, 39),
    )

    for block_position, block in enumerate(blocks):
        px, py = panel_origin(block_position)
        draw.rectangle((px, py, px + panel_w, py + panel_h), outline=(17, 24, 39))
        draw.text(
            (px, py - 22),
            f"Block {block}",
            font=font_bold,
            fill=(17, 24, 39),
        )
        for tick in range(5):
            value = x_min + (x_max - x_min) * tick / 4
            xx = sx(value, px)
            draw.line((xx, py, xx, py + panel_h), fill=(229, 231, 235), width=1)
            if tick in [0, 2, 4]:
                draw.text(
                    (xx, py + panel_h + 14),
                    f"{value:.1f}",
                    font=font,
                    fill=(17, 24, 39),
                    anchor="ma",
                )
        for tick in range(5):
            value = y_min + (y_max - y_min) * tick / 4
            yy = sy(value, py)
            draw.line((px, yy, px + panel_w, yy), fill=(229, 231, 235), width=1)
            if tick in [0, 2, 4]:
                draw.text(
                    (px - 8, yy - 6),
                    f"{value:.1f}",
                    font=font,
                    fill=(17, 24, 39),
                    anchor="ra",
                )

        target_points = [
            (sx(x, px), sy(y, py))
            for x, y in sorted(target_by_block[block], key=lambda pair: pair[0])
        ]
        if len(target_points) > 1:
            draw.line(target_points, fill=(0, 0, 0), width=3)

        for epoch in epochs:
            color = epoch_color(epoch_index(epoch), len(epochs))
            prediction_points = [
                (sx(x, px), sy(y, py))
                for x, y in sorted(
                    by_block_epoch[(block, epoch)], key=lambda pair: pair[0]
                )
            ]
            if len(prediction_points) > 1:
                draw.line(prediction_points, fill=color, width=1)

        for x, y in train_points:
            xx = sx(x, px)
            yy = sy(y, py)
            draw.ellipse(
                (xx - 3, yy - 3, xx + 3, yy + 3),
                fill=(22, 163, 74),
                outline=(20, 83, 45),
            )
        for x, y in test_points:
            xx = sx(x, px)
            yy = sy(y, py)
            draw.ellipse(
                (xx - 3, yy - 3, xx + 3, yy + 3),
                fill=(250, 204, 21),
                outline=(113, 63, 18),
            )

    colorbar_x = width - right + 44
    colorbar_y = top + 30
    colorbar_w = 24
    colorbar_h = min(420, height - top - bottom - 40)
    draw.text(
        (colorbar_x - 6, colorbar_y - 30),
        "Epoch",
        font=font_bold,
        fill=(17, 24, 39),
    )
    for offset in range(colorbar_h):
        t = offset / max(colorbar_h - 1, 1)
        color = epoch_color(round(t * (len(epochs) - 1)), len(epochs))
        y = colorbar_y + offset
        draw.line((colorbar_x, y, colorbar_x + colorbar_w, y), fill=color)
    draw.rectangle(
        (
            colorbar_x,
            colorbar_y,
            colorbar_x + colorbar_w,
            colorbar_y + colorbar_h,
        ),
        outline=(17, 24, 39),
        width=1,
    )
    for idx in epoch_tick_indices():
        t = idx / max(len(epochs) - 1, 1)
        y = colorbar_y + t * colorbar_h
        draw.line(
            (colorbar_x + colorbar_w, y, colorbar_x + colorbar_w + 7, y),
            fill=(17, 24, 39),
            width=1,
        )
        draw.text(
            (colorbar_x + colorbar_w + 12, y - 7),
            str(epochs[idx]),
            font=font,
            fill=(17, 24, 39),
        )
    key_y = colorbar_y + colorbar_h + 34
    draw.ellipse(
        (colorbar_x, key_y - 4, colorbar_x + 8, key_y + 4),
        fill=(22, 163, 74),
        outline=(20, 83, 45),
    )
    draw.text(
        (colorbar_x + 14, key_y - 7),
        "train",
        font=font,
        fill=(17, 24, 39),
    )
    draw.ellipse(
        (colorbar_x, key_y + 18, colorbar_x + 8, key_y + 26),
        fill=(250, 204, 21),
        outline=(113, 63, 18),
    )
    draw.text(
        (colorbar_x + 14, key_y + 15),
        "test",
        font=font,
        fill=(17, 24, 39),
    )
    draw.line(
        (colorbar_x, key_y + 44, colorbar_x + 20, key_y + 44),
        fill=(0, 0, 0),
        width=3,
    )
    draw.text(
        (colorbar_x + 26, key_y + 37),
        "target",
        font=font,
        fill=(17, 24, 39),
    )
    image.save(png_path)

    def svg_color(idx: int) -> str:
        red, green, blue = epoch_color(idx, len(epochs))
        return f"rgb({red},{green},{blue})"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#111827;stroke-width:1}.tick{font-size:12px}.label{font-size:15px;font-weight:600}.title{font-size:22px;font-weight:700}.epoch{font-size:12px}</style>',
        f'<text class="title" x="{left}" y="42">Per-block Sequential Predictions Across Eval Cycles</text>',
    ]
    for block_position, block in enumerate(blocks):
        px, py = panel_origin(block_position)
        svg.append(
            f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#111827" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="label" x="{px}" y="{py - 8}">Block {block}</text>'
        )
        for tick in range(5):
            value = x_min + (x_max - x_min) * tick / 4
            xx = sx(value, px)
            svg.append(
                f'<line class="grid" x1="{xx:.2f}" y1="{py}" x2="{xx:.2f}" y2="{py + panel_h}"/>'
            )
            if tick in [0, 2, 4]:
                svg.append(
                    f'<text class="tick" x="{xx:.2f}" y="{py + panel_h + 20}" text-anchor="middle">{value:.1f}</text>'
                )
        for tick in range(5):
            value = y_min + (y_max - y_min) * tick / 4
            yy = sy(value, py)
            svg.append(
                f'<line class="grid" x1="{px}" y1="{yy:.2f}" x2="{px + panel_w}" y2="{yy:.2f}"/>'
            )
            if tick in [0, 2, 4]:
                svg.append(
                    f'<text class="tick" x="{px - 8}" y="{yy + 4:.2f}" text-anchor="end">{value:.1f}</text>'
                )
        target_points = " ".join(
            f"{sx(x, px):.2f},{sy(y, py):.2f}"
            for x, y in sorted(target_by_block[block], key=lambda pair: pair[0])
        )
        svg.append(
            f'<polyline points="{target_points}" fill="none" stroke="#000000" stroke-width="2.6"/>'
        )
        for epoch in epochs:
            prediction_points = " ".join(
                f"{sx(x, px):.2f},{sy(y, py):.2f}"
                for x, y in sorted(
                    by_block_epoch[(block, epoch)], key=lambda pair: pair[0]
                )
            )
            svg.append(
                f'<polyline points="{prediction_points}" fill="none" stroke="{svg_color(epoch_index(epoch))}" stroke-width="1.1" opacity="0.86"/>'
            )
        for x, y in train_points:
            svg.append(
                f'<circle cx="{sx(x, px):.2f}" cy="{sy(y, py):.2f}" r="3.2" fill="#16a34a" stroke="#14532d" stroke-width="0.8" opacity="0.92"/>'
            )
        for x, y in test_points:
            svg.append(
                f'<circle cx="{sx(x, px):.2f}" cy="{sy(y, py):.2f}" r="3.2" fill="#facc15" stroke="#713f12" stroke-width="0.8" opacity="0.92"/>'
            )

    gradient_id = "prediction-epoch-gradient"
    svg.append("<defs>")
    svg.append(
        f'<linearGradient id="{gradient_id}" x1="0%" y1="0%" x2="0%" y2="100%">'
    )
    for idx in range(len(epochs)):
        offset = 100 * idx / max(len(epochs) - 1, 1)
        svg.append(
            f'<stop offset="{offset:.2f}%" stop-color="{svg_color(idx)}"/>'
        )
    svg.append("</linearGradient>")
    svg.append("</defs>")
    svg.append(
        f'<text class="label" x="{colorbar_x - 6}" y="{colorbar_y - 30}">Epoch</text>'
    )
    svg.append(
        f'<rect x="{colorbar_x}" y="{colorbar_y}" width="{colorbar_w}" height="{colorbar_h}" fill="url(#{gradient_id})" stroke="#111827" stroke-width="1"/>'
    )
    for idx in epoch_tick_indices():
        tick_y = colorbar_y + idx / max(len(epochs) - 1, 1) * colorbar_h
        svg.append(
            f'<line x1="{colorbar_x + colorbar_w}" y1="{tick_y:.2f}" x2="{colorbar_x + colorbar_w + 7}" y2="{tick_y:.2f}" stroke="#111827" stroke-width="1"/>'
        )
        svg.append(
            f'<text class="epoch" x="{colorbar_x + colorbar_w + 12}" y="{tick_y + 4:.2f}">{epochs[idx]}</text>'
        )
    key_y = colorbar_y + colorbar_h + 34
    svg.append(
        f'<circle cx="{colorbar_x + 4}" cy="{key_y}" r="4" fill="#16a34a" stroke="#14532d" stroke-width="0.8"/>'
    )
    svg.append(
        f'<text class="epoch" x="{colorbar_x + 14}" y="{key_y + 4}">train</text>'
    )
    svg.append(
        f'<circle cx="{colorbar_x + 4}" cy="{key_y + 22}" r="4" fill="#facc15" stroke="#713f12" stroke-width="0.8"/>'
    )
    svg.append(
        f'<text class="epoch" x="{colorbar_x + 14}" y="{key_y + 26}">test</text>'
    )
    svg.append(
        f'<line x1="{colorbar_x}" y1="{key_y + 44}" x2="{colorbar_x + 20}" y2="{key_y + 44}" stroke="#000000" stroke-width="3"/>'
    )
    svg.append(
        f'<text class="epoch" x="{colorbar_x + 26}" y="{key_y + 48}">target</text>'
    )
    svg.append("</svg>")
    with open(svg_path, "w") as f:
        f.write("\n".join(svg))
        f.write("\n")
    return True


def write_prediction_plot(
    model: ToyDiffusionBlocks,
    config: ToyConfig,
    sigmas: torch.Tensor,
    train_data: TensorDataset,
    test_data: TensorDataset,
    path: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    model.eval()
    x = torch.linspace(-3.2, 3.2, 512, device=sigmas.device).view(-1, 1)
    with torch.no_grad():
        y_true = target_function(x, config.function)
        seq_preds, _ = model.sequential_predictions(x, sigmas)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    train_x, train_y = train_data.tensors
    test_x, test_y = test_data.tensors
    ax.plot(x.cpu(), y_true.cpu(), color="black", linewidth=2, label="target")
    for idx in range(config.num_blocks):
        alpha = 0.25 + 0.65 * (idx + 1) / config.num_blocks
        ax.plot(
            x.cpu(),
            seq_preds[idx].cpu(),
            linewidth=1.2,
            alpha=alpha,
            label=f"block {idx}",
        )
    ax.scatter(
        train_x.cpu(),
        train_y.cpu(),
        s=24,
        alpha=0.9,
        color="#16a34a",
        label="train points",
        edgecolors="#14532d",
        linewidths=0.4,
        zorder=5,
    )
    ax.scatter(
        test_x.cpu(),
        test_y.cpu(),
        s=24,
        alpha=0.9,
        color="#facc15",
        label="test points",
        edgecolors="#713f12",
        linewidths=0.4,
        zorder=5,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sequential Predictions by Block")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    return True


def train(config: ToyConfig):
    torch.manual_seed(config.seed)
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    train_data = make_dataset(config.num_train, config, seed_offset=0)
    test_data = make_dataset(config.num_test, config, seed_offset=10_000)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    train_eval_loader = DataLoader(train_data, batch_size=config.batch_size)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    model = ToyDiffusionBlocks(config).to(device)
    sigmas = sigma_schedule(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        train_example_count = 0
        per_block_loss_sum = torch.zeros(config.num_blocks)
        per_block_mse_loss_sum = torch.zeros(config.num_blocks)
        per_block_latent_mse_loss_sum = torch.zeros(config.num_blocks)
        per_block_encoder_prediction_loss_sum = torch.zeros(config.num_blocks)
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_parts = model.training_loss(x, y, sigmas)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batch_count += 1
            batch_size = x.shape[0]
            train_example_count += batch_size
            per_block_loss_sum += loss_parts["per_block_loss"].cpu() * batch_size
            per_block_mse_loss_sum += (
                loss_parts["per_block_mse_loss"].cpu() * batch_size
            )
            per_block_latent_mse_loss_sum += (
                loss_parts["per_block_latent_mse_loss"].cpu() * batch_size
            )
            per_block_encoder_prediction_loss_sum += (
                loss_parts["per_block_encoder_prediction_loss"].cpu() * batch_size
            )

        if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
            train_loss_rows = []
            train_loss_denominator = max(train_example_count, 1)
            per_block_loss = per_block_loss_sum / train_loss_denominator
            per_block_mse_loss = per_block_mse_loss_sum / train_loss_denominator
            per_block_latent_mse_loss = (
                per_block_latent_mse_loss_sum / train_loss_denominator
            )
            per_block_encoder_prediction_loss = (
                per_block_encoder_prediction_loss_sum / train_loss_denominator
            )
            for block_index in range(config.num_blocks):
                train_loss_rows.append(
                    {
                        "epoch": epoch,
                        "block_index": block_index,
                        "sigma": float(sigmas[block_index].detach().cpu()),
                        "train_loss": float(per_block_loss[block_index]),
                        "train_mse_loss": float(per_block_mse_loss[block_index]),
                        "train_latent_mse_loss": float(
                            per_block_latent_mse_loss[block_index]
                        ),
                        "train_encoder_prediction_loss": float(
                            per_block_encoder_prediction_loss[block_index]
                        ),
                    }
                )
            train_rows = evaluate_model(model, train_eval_loader, sigmas, "train")
            test_rows = evaluate_model(model, test_loader, sigmas, "test")
            prediction_curve_rows = evaluate_prediction_curves(
                model,
                config,
                sigmas,
                epoch,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / max(batch_count, 1),
                    "train_final_rmse": train_rows[-1]["sequential_rmse"],
                    "test_final_rmse": test_rows[-1]["sequential_rmse"],
                    "train_loss_rows": train_loss_rows,
                    "train_block_rows": train_rows,
                    "test_block_rows": test_rows,
                    "prediction_curve_rows": prediction_curve_rows,
                }
            )
            print(
                f"epoch={epoch:04d} "
                f"loss={history[-1]['train_loss']:.4f} "
                f"train_rmse={history[-1]['train_final_rmse']:.4f} "
                f"test_rmse={history[-1]['test_final_rmse']:.4f}"
            )

    timestamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    final_rows = history[-1]["train_block_rows"] + history[-1]["test_block_rows"]
    config_path = os.path.join(run_dir, "config.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    csv_path = os.path.join(run_dir, "blockwise_metrics.csv")
    plot_path = os.path.join(run_dir, "blockwise_rmse.png")
    latent_plot_path = os.path.join(run_dir, "blockwise_latent_distance.png")
    pred_plot_path = os.path.join(run_dir, "predictions_by_block.png")
    eval_cycle_csv_path = os.path.join(
        run_dir,
        "test_rmse_by_block_eval_cycles.csv",
    )
    eval_cycle_latent_csv_path = os.path.join(
        run_dir,
        "test_latent_distance_by_block_eval_cycles.csv",
    )
    eval_cycle_png_path = os.path.join(
        run_dir,
        "test_rmse_by_block_eval_cycles.png",
    )
    eval_cycle_svg_path = os.path.join(
        run_dir,
        "test_rmse_by_block_eval_cycles.svg",
    )
    eval_cycle_latent_png_path = os.path.join(
        run_dir,
        "test_latent_distance_by_block_eval_cycles.png",
    )
    eval_cycle_latent_svg_path = os.path.join(
        run_dir,
        "test_latent_distance_by_block_eval_cycles.svg",
    )
    train_loss_csv_path = os.path.join(
        run_dir,
        "train_loss_by_block_eval_cycles.csv",
    )
    train_loss_png_path = os.path.join(
        run_dir,
        "train_loss_by_block_eval_cycles.png",
    )
    train_loss_svg_path = os.path.join(
        run_dir,
        "train_loss_by_block_eval_cycles.svg",
    )
    prediction_curve_csv_path = os.path.join(
        run_dir,
        "prediction_curves_by_block_eval_cycles.csv",
    )
    prediction_curve_png_path = os.path.join(
        run_dir,
        "prediction_curves_by_block_eval_cycles.png",
    )
    prediction_curve_svg_path = os.path.join(
        run_dir,
        "prediction_curves_by_block_eval_cycles.svg",
    )

    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)
        f.write("\n")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "device": str(device),
                "history": history,
                "final_rows": final_rows,
            },
            f,
            indent=2,
        )
        f.write("\n")
    write_rows_csv(final_rows, csv_path)
    write_eval_cycle_rmse_csv(history, eval_cycle_csv_path)
    write_eval_cycle_latent_distance_csv(history, eval_cycle_latent_csv_path)
    write_eval_cycle_train_loss_csv(history, train_loss_csv_path)
    write_prediction_curve_csv(history, prediction_curve_csv_path)
    wrote_block_plot = False
    wrote_latent_plot = False
    wrote_pred_plot = False
    wrote_eval_cycle_plot = False
    wrote_eval_cycle_latent_plot = False
    wrote_train_loss_plot = False
    wrote_prediction_curve_plot = False
    if not config.no_plots:
        wrote_eval_cycle_plot = write_eval_cycle_rmse_plot(
            history,
            eval_cycle_png_path,
            eval_cycle_svg_path,
        )
        wrote_eval_cycle_latent_plot = write_eval_cycle_latent_distance_plot(
            history,
            eval_cycle_latent_png_path,
            eval_cycle_latent_svg_path,
        )
        wrote_train_loss_plot = write_eval_cycle_train_loss_plot(
            history,
            train_loss_png_path,
            train_loss_svg_path,
        )
        wrote_prediction_curve_plot = write_prediction_curve_eval_cycle_plot(
            history,
            train_data,
            test_data,
            prediction_curve_png_path,
            prediction_curve_svg_path,
        )
        wrote_block_plot = write_plot(final_rows, plot_path)
        wrote_latent_plot = write_latent_distance_plot(final_rows, latent_plot_path)
        wrote_pred_plot = write_prediction_plot(
            model,
            config,
            sigmas,
            train_data,
            test_data,
            pred_plot_path,
        )

    print(f"Wrote {config_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {eval_cycle_csv_path}")
    print(f"Wrote {eval_cycle_latent_csv_path}")
    print(f"Wrote {train_loss_csv_path}")
    print(f"Wrote {prediction_curve_csv_path}")
    if config.no_plots:
        print("Skipped plots because --no_plots was set")
    elif wrote_block_plot:
        print(f"Wrote {plot_path}")
    else:
        print("Skipped blockwise plot because matplotlib is not installed")
    if not config.no_plots:
        if wrote_latent_plot:
            print(f"Wrote {latent_plot_path}")
        else:
            print("Skipped latent-distance plot because matplotlib is not installed")
    if config.no_plots:
        return
    if wrote_eval_cycle_plot:
        print(f"Wrote {eval_cycle_png_path}")
        print(f"Wrote {eval_cycle_svg_path}")
    else:
        print("Skipped eval-cycle RMSE plot because Pillow is not installed")
    if wrote_eval_cycle_latent_plot:
        print(f"Wrote {eval_cycle_latent_png_path}")
        print(f"Wrote {eval_cycle_latent_svg_path}")
    else:
        print("Skipped eval-cycle latent-distance plot because matplotlib is not installed")
    if wrote_train_loss_plot:
        print(f"Wrote {train_loss_png_path}")
        print(f"Wrote {train_loss_svg_path}")
    else:
        print("Skipped eval-cycle train-loss plot because Pillow is not installed")
    if wrote_prediction_curve_plot:
        print(f"Wrote {prediction_curve_png_path}")
        print(f"Wrote {prediction_curve_svg_path}")
    else:
        print("Skipped eval-cycle prediction plot because Pillow is not installed")
    if wrote_pred_plot:
        print(f"Wrote {pred_plot_path}")
    else:
        print("Skipped prediction plot because matplotlib is not installed")


def parse_args() -> ToyConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=ToyConfig.num_train)
    parser.add_argument("--num_test", type=int, default=ToyConfig.num_test)
    parser.add_argument("--num_blocks", type=int, default=ToyConfig.num_blocks)
    parser.add_argument("--latent_dim", type=int, default=ToyConfig.latent_dim)
    parser.add_argument("--hidden_dim", type=int, default=ToyConfig.hidden_dim)
    parser.add_argument("--depth", type=int, default=ToyConfig.depth)
    parser.add_argument("--batch_size", type=int, default=ToyConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=ToyConfig.epochs)
    parser.add_argument("--lr", type=float, default=ToyConfig.lr)
    parser.add_argument("--weight_decay", type=float, default=ToyConfig.weight_decay)
    parser.add_argument("--sigma_max", type=float, default=ToyConfig.sigma_max)
    parser.add_argument("--sigma_min", type=float, default=ToyConfig.sigma_min)
    parser.add_argument(
        "--sigma_schedule",
        type=str,
        default=ToyConfig.sigma_schedule,
        choices=["log", "linear"],
    )
    parser.add_argument(
        "--prediction_loss_weight",
        type=float,
        default=ToyConfig.prediction_loss_weight,
    )
    parser.add_argument(
        "--latent_loss_weight",
        type=float,
        default=ToyConfig.latent_loss_weight,
    )
    parser.add_argument(
        "--training_objective",
        type=str,
        default=ToyConfig.training_objective,
        choices=["clean_latent", "residual_next_latent"],
        help=(
            "clean_latent keeps the original toy objective where each block "
            "predicts the clean latent; residual_next_latent makes each block "
            "predict delta_hat and applies z_next_hat = z + delta_hat toward "
            "the next scheduled latent state"
        ),
    )
    parser.add_argument(
        "--train_encoder_with_prediction_loss_only",
        action="store_true",
        help=(
            "detach clean latent targets in latent/residual losses and train "
            "the target encoder only through a final-block decoded prediction "
            "loss on decoder(target_encoder(y))"
        ),
    )
    parser.add_argument(
        "--observation_noise_std",
        "--target_noise_std",
        dest="observation_noise_std",
        type=float,
        default=ToyConfig.observation_noise_std,
        help=(
            "Gaussian observation noise std added to y for both train and test "
            "points; --target_noise_std is kept as a backward-compatible alias"
        ),
    )
    parser.add_argument(
        "--function",
        type=str,
        default=ToyConfig.function,
        choices=["sine_poly", "saw_sine", "cubic"],
    )
    parser.add_argument("--seed", type=int, default=ToyConfig.seed)
    parser.add_argument("--output_dir", type=str, default=ToyConfig.output_dir)
    parser.add_argument("--eval_every", type=int, default=ToyConfig.eval_every)
    parser.add_argument(
        "--prediction_grid_points",
        type=int,
        default=ToyConfig.prediction_grid_points,
    )
    parser.add_argument("--device", type=str, default=ToyConfig.device)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    config = ToyConfig(**vars(args))
    if config.num_blocks < 1:
        raise ValueError("--num_blocks must be at least 1")
    if config.sigma_min <= 0 or config.sigma_max <= 0:
        raise ValueError("--sigma_min and --sigma_max must be positive")
    if config.sigma_min >= config.sigma_max:
        raise ValueError("--sigma_min must be smaller than --sigma_max")
    if config.prediction_grid_points < 2:
        raise ValueError("--prediction_grid_points must be at least 2")
    if config.observation_noise_std < 0.0:
        raise ValueError("--observation_noise_std must be non-negative")
    if config.prediction_loss_weight < 0.0:
        raise ValueError("--prediction_loss_weight must be non-negative")
    if config.latent_loss_weight < 0.0:
        raise ValueError("--latent_loss_weight must be non-negative")
    return config


if __name__ == "__main__":
    train(parse_args())
