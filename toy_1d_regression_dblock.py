from __future__ import annotations

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
    if name == "sin_cos_bifurcation":
        return 0.5 * (torch.sin(x) + torch.cos(x))
    raise ValueError(f"Unknown target function: {name}")


def target_curves(x: torch.Tensor, name: str) -> list[tuple[str, torch.Tensor]]:
    if name == "sin_cos_bifurcation":
        return [
            ("sin(x)", torch.sin(x)),
            ("cos(x)", torch.cos(x)),
        ]
    return [("target", target_function(x, name))]


def target_curve_point_sets(
    x_values: list[float],
    name: str,
) -> list[dict]:
    sorted_x = sorted(set(x_values))
    x = torch.tensor(sorted_x, dtype=torch.float32).view(-1, 1)
    curves = []
    for index, (label, y) in enumerate(target_curves(x, name)):
        curves.append(
            {
                "label": label,
                "color": target_curve_color(index),
                "hex_color": target_curve_hex_color(index),
                "points": [
                    (float(x_value), float(y_value))
                    for x_value, y_value in zip(sorted_x, y[:, 0].tolist())
                ],
            }
        )
    return curves


def target_curve_color(index: int) -> tuple[int, int, int]:
    colors = [
        (0, 0, 0),
        (107, 114, 128),
        (124, 58, 237),
    ]
    return colors[index % len(colors)]


def target_curve_hex_color(index: int) -> str:
    red, green, blue = target_curve_color(index)
    return f"#{red:02x}{green:02x}{blue:02x}"


def sample_target_values(
    x: torch.Tensor,
    config: "ToyConfig",
    generator: torch.Generator,
) -> torch.Tensor:
    if config.function != "sin_cos_bifurcation":
        return target_function(x, config.function)

    branch = torch.randint(
        0,
        2,
        (x.shape[0], 1),
        generator=generator,
        device=x.device,
    )
    sin_y = torch.sin(x)
    cos_y = torch.cos(x)
    return torch.where(branch == 0, sin_y, cos_y)


def parse_shared_denoising_block_ranges(
    ranges: str | None,
    num_blocks: int,
) -> list[tuple[int, int, int]]:
    if ranges is None or ranges == "":
        return []

    parsed_ranges = []
    covered_blocks = set()
    for raw_range in ranges.split(","):
        raw_range = raw_range.strip()
        if raw_range == "":
            continue
        parts = raw_range.split(":")
        if len(parts) not in [2, 3]:
            raise ValueError(
                "--shared_denoising_block_ranges entries must be "
                "start:end or start:end:network_index"
            )
        try:
            start = int(parts[0])
            end = int(parts[1])
            network_index = int(parts[2]) if len(parts) == 3 else start
        except ValueError as exc:
            raise ValueError(
                "--shared_denoising_block_ranges entries must contain integer indices"
            ) from exc
        if start < 0 or end < 0 or network_index < 0:
            raise ValueError(
                "--shared_denoising_block_ranges indices must be non-negative"
            )
        if start >= end:
            raise ValueError(
                "--shared_denoising_block_ranges entries must have start < end"
            )
        if end > num_blocks:
            raise ValueError(
                "--shared_denoising_block_ranges range ends must be <= --num_blocks"
            )
        if network_index >= num_blocks:
            raise ValueError(
                "--shared_denoising_block_ranges network indices must be "
                "smaller than --num_blocks"
            )
        blocks = set(range(start, end))
        overlap = covered_blocks.intersection(blocks)
        if overlap:
            raise ValueError(
                "--shared_denoising_block_ranges entries must not overlap"
            )
        covered_blocks.update(blocks)
        parsed_ranges.append((start, end, network_index))
    return parsed_ranges


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
    block_objective_pattern: str = "global"
    denoising_target: str = "embedding"
    residual_next_latent_training_mode: str = "sequential"
    shared_denoising_block: bool = False
    shared_denoising_block_index: int = 0
    shared_denoising_block_ranges: str = ""
    initial_noise_mode: str = "per_example"
    initial_noise_std: float | None = None
    interblock_transition: str = "euler"
    noise_correction_mode: str = "none"
    train_encoder_with_prediction_loss_only: bool = False
    observation_noise_std: float = 0.0
    function: str = "sine_poly"
    seed: int = 0
    output_dir: str = "results/toy_1d_regression"
    eval_every: int = 100
    prediction_grid_points: int = 256
    prediction_uncertainty_samples: int = 50
    device: str = "auto"
    no_plots: bool = False


def make_dataset(
    num_examples: int,
    config: ToyConfig,
    seed_offset: int,
    include_all_bifurcation_branches: bool = False,
):
    generator = torch.Generator().manual_seed(config.seed + seed_offset)
    x = -3.0 + 6.0 * torch.rand(num_examples, 1, generator=generator)
    if config.function == "sin_cos_bifurcation" and include_all_bifurcation_branches:
        y = torch.cat([torch.sin(x), torch.cos(x)], dim=0)
        x = torch.cat([x, x], dim=0)
    else:
        y = sample_target_values(x, config, generator)
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
        self.state_dim = 1 if config.denoising_target == "value" else config.latent_dim
        self.target_encoder = nn.Sequential(
            nn.Linear(1, config.latent_dim),
            nn.Tanh(),
        )
        self.target_decoder = nn.Linear(config.latent_dim, 1)
        self.blocks = nn.ModuleList(
            [
                DenoisingBlock(self.state_dim, config.hidden_dim, config.depth)
                for _ in range(config.num_blocks)
            ]
        )
        self.shared_denoising_block_ranges = parse_shared_denoising_block_ranges(
            config.shared_denoising_block_ranges,
            config.num_blocks,
        )
        fixed_noise_generator = torch.Generator(device="cpu").manual_seed(
            config.seed + 75_000
        )
        self.register_buffer(
            "fixed_initial_noise",
            torch.randn(1, self.state_dim, generator=fixed_noise_generator),
            persistent=False,
        )

    def clean_latent(self, y: torch.Tensor) -> torch.Tensor:
        if self.config.denoising_target == "value":
            return y
        return self.target_encoder(y)

    def predict_from_latent(self, z_clean_pred: torch.Tensor) -> torch.Tensor:
        if self.config.denoising_target == "value":
            return z_clean_pred
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

    def corrected_noise_update(
        self,
        z_clean_pred: torch.Tensor,
        z_clean_target: torch.Tensor | None,
        next_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.noise_correction_mode == "none":
            raise ValueError("corrected_noise_update called with correction disabled")
        if self.config.noise_correction_mode != "batch_oracle":
            raise ValueError(
                f"Unknown noise_correction_mode: {self.config.noise_correction_mode}"
            )
        if z_clean_target is None:
            raise ValueError(
                "batch_oracle noise correction requires clean target latents"
            )

        clean_rmse_scalar = (z_clean_pred - z_clean_target).square().mean().sqrt()
        clean_rmse = clean_rmse_scalar.expand_as(next_sigma)
        remaining_variance = next_sigma.square() - clean_rmse_scalar.square()
        correction_sigma = remaining_variance.clamp_min(0.0).sqrt()
        noise = torch.randn_like(z_clean_pred)
        z_next = z_clean_pred + correction_sigma[:, None] * noise
        return z_next, clean_rmse.detach(), correction_sigma.detach()

    def interblock_update(
        self,
        z: torch.Tensor,
        z_clean_pred: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        *,
        base_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.interblock_transition == "euler":
            return self.euler_update(z, z_clean_pred, sigma, next_sigma)
        if self.config.interblock_transition == "direct_denoised":
            return z_clean_pred
        if self.config.interblock_transition == "direct_denoised_plus_noise":
            return z_clean_pred + next_sigma[:, None] * torch.randn_like(z_clean_pred)
        if self.config.interblock_transition == "direct_denoised_plus_rescaled_noise":
            if base_noise is None:
                raise ValueError(
                    "direct_denoised_plus_rescaled_noise requires base_noise"
                )
            return z_clean_pred + next_sigma[:, None] * base_noise
        raise ValueError(
            f"Unknown interblock_transition: {self.config.interblock_transition}"
        )

    def uses_residual_next_latent_objective(self) -> bool:
        return self.config.training_objective == "residual_next_latent"

    def uses_residual_to_clean_objective(self) -> bool:
        return self.config.training_objective == "residual_to_clean"

    def uses_residual_objective(self) -> bool:
        return self.config.training_objective in [
            "residual_next_latent",
            "residual_to_clean",
        ]

    def uses_patterned_block_objectives(self) -> bool:
        return self.config.block_objective_pattern != "global"

    def block_objective(self, block_index: int) -> str:
        pattern = self.config.block_objective_pattern
        residual_objective = (
            "residual_to_clean"
            if self.uses_residual_to_clean_objective()
            else "residual_next_latent"
        )
        if pattern == "global":
            if self.uses_residual_next_latent_objective():
                return "residual_next_latent"
            if self.uses_residual_to_clean_objective():
                return "residual_to_clean"
            return "prediction"
        if pattern == "all_prediction":
            return "prediction"
        if pattern == "all_residual":
            return residual_objective
        if pattern == "first_prediction_then_residual":
            if block_index == 0:
                return "prediction"
            return residual_objective
        if pattern == "alternating_prediction_residual":
            if block_index % 2 == 0:
                return "prediction"
            return residual_objective
        raise ValueError(f"Unknown block_objective_pattern: {pattern}")

    def block_uses_residual_next_latent(self, block_index: int) -> bool:
        return self.block_objective(block_index) == "residual_next_latent"

    def block_uses_residual_to_clean(self, block_index: int) -> bool:
        return self.block_objective(block_index) == "residual_to_clean"

    def block_uses_residual_update(self, block_index: int) -> bool:
        return self.block_objective(block_index) in [
            "residual_next_latent",
            "residual_to_clean",
        ]

    def residual_update_alpha(
        self,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
    ) -> torch.Tensor:
        return 1.0 - next_sigma / sigma

    def prediction_loss_weights(self, reference: torch.Tensor) -> torch.Tensor:
        weights = torch.full(
            (self.config.num_blocks,),
            self.config.prediction_loss_weight,
            device=reference.device,
            dtype=reference.dtype,
        )
        weights[-1] = torch.maximum(weights[-1], weights.new_tensor(1.0))
        return weights

    def network_block_index(self, block_index: int) -> int:
        if self.config.shared_denoising_block:
            return self.config.shared_denoising_block_index
        for start, end, network_index in self.shared_denoising_block_ranges:
            if start <= block_index < end:
                return network_index
        return block_index

    def denoising_block(self, block_index: int) -> DenoisingBlock:
        return self.blocks[self.network_block_index(block_index)]

    def initial_noise(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.config.initial_noise_mode == "per_example":
            return torch.randn(
                batch_size,
                self.state_dim,
                device=device,
                dtype=dtype,
            )
        if self.config.initial_noise_mode == "shared_per_batch":
            return torch.randn(
                1,
                self.state_dim,
                device=device,
                dtype=dtype,
            ).expand(batch_size, -1)
        if self.config.initial_noise_mode == "fixed_shared":
            return self.fixed_initial_noise.to(device=device, dtype=dtype).expand(
                batch_size,
                -1,
            )
        raise ValueError(f"Unknown initial_noise_mode: {self.config.initial_noise_mode}")

    def initial_noise_scale(self, sigmas: torch.Tensor) -> torch.Tensor:
        if self.config.initial_noise_std is not None:
            return sigmas.new_tensor(self.config.initial_noise_std)
        return torch.sqrt(1.0 + sigmas[0] ** 2)

    def latent_target_for_loss(self, z_clean: torch.Tensor) -> torch.Tensor:
        if self.config.train_encoder_with_prediction_loss_only:
            return z_clean.detach()
        return z_clean

    def encoder_prediction_loss(self, z_clean: torch.Tensor, y: torch.Tensor):
        if self.config.denoising_target == "value":
            return z_clean.new_zeros(())
        if not self.config.train_encoder_with_prediction_loss_only:
            return z_clean.new_zeros(())
        encoder_prediction = self.predict_from_latent(z_clean)
        return F.mse_loss(encoder_prediction, y)

    def sequential_predictions(
        self,
        x: torch.Tensor,
        sigmas: torch.Tensor,
        *,
        target_y: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        prediction_mode: str = "state",
        return_noise_correction_metrics: bool = False,
    ):
        if prediction_mode not in ["state", "clean_estimate"]:
            raise ValueError(f"Unknown prediction_mode: {prediction_mode}")
        batch_size = x.shape[0]
        z_clean_target = self.clean_latent(target_y) if target_y is not None else None
        if noise is None:
            base_noise = self.initial_noise(batch_size, x.device, x.dtype)
        else:
            base_noise = noise.to(device=x.device, dtype=x.dtype)
        z = base_noise * self.initial_noise_scale(sigmas)

        predictions = []
        latents = []
        correction_clean_rmse = []
        correction_sigma = []
        correction_saturated = []
        for block_index in range(self.config.num_blocks):
            block = self.denoising_block(block_index)
            sigma = sigmas[block_index].expand(batch_size)
            next_sigma = (
                sigmas[block_index + 1].expand(batch_size)
                if block_index < len(self.blocks) - 1
                else torch.zeros_like(sigma)
            )
            block_output = block(x, z, sigma)
            if self.block_uses_residual_next_latent(block_index):
                z_next_pred = z + block_output
                y_pred = self.predict_from_latent(z_next_pred)
                latent_prediction = z_next_pred
            elif self.block_uses_residual_to_clean(block_index):
                alpha = self.residual_update_alpha(sigma, next_sigma)
                z_next_pred = z + alpha[:, None] * block_output
                if prediction_mode == "clean_estimate":
                    y_pred = self.predict_from_latent(z + block_output)
                else:
                    y_pred = self.predict_from_latent(z_next_pred)
                latent_prediction = z_next_pred
            else:
                y_pred = self.predict_from_latent(block_output)
                latent_prediction = block_output
            predictions.append(y_pred)
            latents.append(latent_prediction)
            if block_index < len(self.blocks) - 1:
                if self.block_uses_residual_update(block_index):
                    z = z_next_pred.detach()
                    clean_rmse = sigma.new_full((batch_size,), float("nan"))
                    added_sigma = sigma.new_full((batch_size,), float("nan"))
                    saturated = sigma.new_full((batch_size,), float("nan"))
                elif self.config.noise_correction_mode != "none":
                    z, clean_rmse, added_sigma = self.corrected_noise_update(
                        block_output,
                        z_clean_target,
                        next_sigma,
                    )
                    z = z.detach()
                    saturated = (added_sigma <= 0.0).to(dtype=sigma.dtype)
                else:
                    z = self.interblock_update(
                        z,
                        block_output,
                        sigma,
                        next_sigma,
                        base_noise=base_noise,
                    ).detach()
                    if z_clean_target is None:
                        clean_rmse = sigma.new_full((batch_size,), float("nan"))
                    else:
                        clean_rmse = (
                            (latent_prediction - z_clean_target)
                            .square()
                            .mean(dim=1)
                            .sqrt()
                        )
                    if self.config.interblock_transition == "direct_denoised":
                        added_sigma = next_sigma.new_zeros(next_sigma.shape)
                    else:
                        added_sigma = next_sigma.detach()
                    saturated = sigma.new_zeros((batch_size,))
                correction_clean_rmse.append(clean_rmse.detach())
                correction_sigma.append(added_sigma.detach())
                correction_saturated.append(saturated.detach())
        outputs = (torch.stack(predictions), torch.stack(latents))
        if not return_noise_correction_metrics:
            return outputs
        if correction_clean_rmse:
            metrics = {
                "clean_rmse": torch.stack(correction_clean_rmse),
                "added_sigma": torch.stack(correction_sigma),
                "saturated": torch.stack(correction_saturated),
            }
        else:
            empty = x.new_empty((0, batch_size))
            metrics = {
                "clean_rmse": empty,
                "added_sigma": empty,
                "saturated": empty,
            }
        return (*outputs, metrics)

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
        for block_index in range(self.config.num_blocks):
            block = self.denoising_block(block_index)
            sigma = sigmas[block_index].expand(x.shape[0])
            if noise is None:
                epsilon = torch.randn_like(z_clean)
            else:
                epsilon = noise.to(device=x.device, dtype=x.dtype)
            z_noisy = z_clean + sigma[:, None] * epsilon
            next_sigma = (
                sigmas[block_index + 1].expand(x.shape[0])
                if block_index < len(self.blocks) - 1
                else torch.zeros_like(sigma)
            )
            block_output = block(x, z_noisy, sigma)
            if self.block_uses_residual_next_latent(block_index):
                z_next_pred = z_noisy + block_output
                y_pred = self.predict_from_latent(z_next_pred)
                latent_prediction = z_next_pred
            elif self.block_uses_residual_to_clean(block_index):
                alpha = self.residual_update_alpha(sigma, next_sigma)
                z_next_pred = z_noisy + alpha[:, None] * block_output
                y_pred = self.predict_from_latent(z_next_pred)
                latent_prediction = z_next_pred
            else:
                y_pred = self.predict_from_latent(block_output)
                latent_prediction = block_output
            predictions.append(y_pred)
            latents.append(latent_prediction)
        return torch.stack(predictions), torch.stack(latents)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor, sigmas: torch.Tensor):
        if self.uses_patterned_block_objectives():
            return self.patterned_block_objective_training_loss(x, y, sigmas)
        if self.uses_residual_next_latent_objective():
            return self.residual_next_latent_training_loss(x, y, sigmas)
        if self.uses_residual_to_clean_objective():
            return self.residual_to_clean_training_loss(x, y, sigmas)

        z_clean = self.clean_latent(y)
        z_clean_target = self.latent_target_for_loss(z_clean)
        predictions, latents = self.sequential_predictions(x, sigmas, target_y=y)
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

    def patterned_block_objective_training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
    ):
        batch_size = x.shape[0]
        z_clean = self.clean_latent(y)
        z_clean_target = self.latent_target_for_loss(z_clean)
        base_noise = self.initial_noise(batch_size, x.device, x.dtype)
        z = base_noise * self.initial_noise_scale(sigmas)

        predictions = []
        mse_losses = []
        residual_losses = []
        per_block_losses = []
        encoder_prediction_losses = []
        for block_index in range(self.config.num_blocks):
            block = self.denoising_block(block_index)
            sigma = sigmas[block_index].expand(batch_size)
            next_sigma = (
                sigmas[block_index + 1].expand(batch_size)
                if block_index < len(self.blocks) - 1
                else torch.zeros_like(sigma)
            )
            z_input = z.detach()
            block_output = block(x, z_input, sigma)
            if self.block_uses_residual_next_latent(block_index):
                z_next_pred = z_input + block_output
                z_next_target = z_clean_target + (next_sigma / sigma)[:, None] * (
                    z_input - z_clean_target
                )
                delta_target = z_next_target - z_input
                residual_loss = F.mse_loss(block_output, delta_target)
                y_pred = self.predict_from_latent(z_next_pred)
                prediction_loss = F.mse_loss(y_pred, y)
                per_block_loss = self.config.latent_loss_weight * residual_loss
                z = z_next_pred.detach()
            elif self.block_uses_residual_to_clean(block_index):
                alpha = self.residual_update_alpha(sigma, next_sigma)
                residual_target = z_clean_target - z_input
                z_next_pred = z_input + alpha[:, None] * block_output
                residual_loss = F.mse_loss(block_output, residual_target)
                y_pred = self.predict_from_latent(z_next_pred)
                prediction_loss = F.mse_loss(y_pred, y)
                per_block_loss = self.config.latent_loss_weight * residual_loss
                z = z_next_pred.detach()
            else:
                z_clean_pred = block_output
                y_pred = self.predict_from_latent(z_clean_pred)
                prediction_loss = F.mse_loss(y_pred, y)
                residual_loss = prediction_loss.new_zeros(())
                per_block_loss = prediction_loss
                if block_index < len(self.blocks) - 1:
                    if self.config.noise_correction_mode == "none":
                        z = self.interblock_update(
                            z_input,
                            z_clean_pred,
                            sigma,
                            next_sigma,
                            base_noise=base_noise,
                        ).detach()
                    else:
                        z, _, _ = self.corrected_noise_update(
                            z_clean_pred,
                            z_clean_target,
                            next_sigma,
                        )
                        z = z.detach()

            predictions.append(y_pred)
            mse_losses.append(prediction_loss)
            residual_losses.append(residual_loss)
            per_block_losses.append(per_block_loss)
            encoder_prediction_losses.append(prediction_loss.new_zeros(()))

        mse_losses = torch.stack(mse_losses)
        residual_losses = torch.stack(residual_losses)
        per_block_losses = torch.stack(per_block_losses)
        encoder_prediction_losses = torch.stack(encoder_prediction_losses)
        loss = per_block_losses.sum()
        return loss, {
            "mse_mean": mse_losses.mean().detach(),
            "latent_mse_mean": residual_losses.mean().detach(),
            "encoder_prediction_loss": encoder_prediction_losses.sum().detach(),
            "per_block_loss": per_block_losses.detach(),
            "per_block_mse_loss": mse_losses.detach(),
            "per_block_latent_mse_loss": residual_losses.detach(),
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
        if self.config.residual_next_latent_training_mode == "independent":
            return self.independent_residual_next_latent_training_loss(x, y, sigmas)
        if self.config.residual_next_latent_training_mode != "sequential":
            raise ValueError(
                "Unsupported residual_next_latent_training_mode: "
                f"{self.config.residual_next_latent_training_mode}"
            )

        batch_size = x.shape[0]
        z_clean = self.clean_latent(y)
        z = self.initial_noise(batch_size, x.device, x.dtype)
        z = z * self.initial_noise_scale(sigmas)

        predictions = []
        residual_losses = []
        for block_index in range(self.config.num_blocks):
            block = self.denoising_block(block_index)
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

    def independent_residual_next_latent_training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
    ):
        z_clean = self.clean_latent(y)
        z_clean_target = self.latent_target_for_loss(z_clean)

        predictions = []
        residual_losses = []
        for block_index in range(self.config.num_blocks):
            block = self.denoising_block(block_index)
            sigma = sigmas[block_index].expand(x.shape[0])
            next_sigma = (
                sigmas[block_index + 1].expand(x.shape[0])
                if block_index < len(self.blocks) - 1
                else torch.zeros_like(sigma)
            )
            epsilon = torch.randn_like(z_clean_target)
            z_input = (z_clean_target + sigma[:, None] * epsilon).detach()
            z_next_target = z_clean_target + next_sigma[:, None] * epsilon
            delta_target = z_next_target - z_input

            delta_pred = block(x, z_input, sigma)
            z_next_pred = z_input + delta_pred
            residual_losses.append(
                F.mse_loss(delta_pred, delta_target, reduction="none").mean(dim=(0, 1))
            )
            predictions.append(self.predict_from_latent(z_next_pred))

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

    def residual_to_clean_training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
    ):
        if self.config.residual_next_latent_training_mode == "independent":
            return self.independent_residual_to_clean_training_loss(x, y, sigmas)
        if self.config.residual_next_latent_training_mode != "sequential":
            raise ValueError(
                "Unsupported residual_next_latent_training_mode: "
                f"{self.config.residual_next_latent_training_mode}"
            )

        batch_size = x.shape[0]
        z_clean = self.clean_latent(y)
        z_clean_target = self.latent_target_for_loss(z_clean)
        z = self.initial_noise(batch_size, x.device, x.dtype)
        z = z * self.initial_noise_scale(sigmas)

        predictions = []
        residual_losses = []
        for block_index in range(self.config.num_blocks):
            block = self.denoising_block(block_index)
            sigma = sigmas[block_index].expand(batch_size)
            next_sigma = (
                sigmas[block_index + 1].expand(batch_size)
                if block_index < len(self.blocks) - 1
                else torch.zeros_like(sigma)
            )
            z_input = z.detach()
            residual_pred = block(x, z_input, sigma)
            residual_target = z_clean_target - z_input
            alpha = self.residual_update_alpha(sigma, next_sigma)
            z_next_pred = z_input + alpha[:, None] * residual_pred
            residual_losses.append(
                F.mse_loss(residual_pred, residual_target, reduction="none").mean(
                    dim=(0, 1)
                )
            )
            predictions.append(self.predict_from_latent(z_next_pred))
            z = z_next_pred.detach()

        return self.aggregate_residual_training_loss(
            predictions,
            residual_losses,
            z_clean,
            y,
        )

    def independent_residual_to_clean_training_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sigmas: torch.Tensor,
    ):
        z_clean = self.clean_latent(y)
        z_clean_target = self.latent_target_for_loss(z_clean)

        predictions = []
        residual_losses = []
        for block_index in range(self.config.num_blocks):
            block = self.denoising_block(block_index)
            sigma = sigmas[block_index].expand(x.shape[0])
            next_sigma = (
                sigmas[block_index + 1].expand(x.shape[0])
                if block_index < len(self.blocks) - 1
                else torch.zeros_like(sigma)
            )
            epsilon = torch.randn_like(z_clean_target)
            z_input = (z_clean_target + sigma[:, None] * epsilon).detach()
            residual_target = z_clean_target - z_input

            residual_pred = block(x, z_input, sigma)
            alpha = self.residual_update_alpha(sigma, next_sigma)
            z_next_pred = z_input + alpha[:, None] * residual_pred
            residual_losses.append(
                F.mse_loss(residual_pred, residual_target, reduction="none").mean(
                    dim=(0, 1)
                )
            )
            predictions.append(self.predict_from_latent(z_next_pred))

        return self.aggregate_residual_training_loss(
            predictions,
            residual_losses,
            z_clean,
            y,
        )

    def aggregate_residual_training_loss(
        self,
        predictions: list[torch.Tensor],
        residual_losses: list[torch.Tensor],
        z_clean: torch.Tensor,
        y: torch.Tensor,
    ):
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
    correction_clean_rmse_sum = torch.zeros(
        model.config.num_blocks,
        device=sigmas.device,
    )
    correction_added_sigma_sum = torch.zeros(
        model.config.num_blocks,
        device=sigmas.device,
    )
    correction_saturated_sum = torch.zeros(
        model.config.num_blocks,
        device=sigmas.device,
    )
    correction_count = torch.zeros(model.config.num_blocks, device=sigmas.device)

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(sigmas.device)
            y = y.to(sigmas.device)
            z_clean = model.clean_latent(y)
            seq_preds, seq_latents, correction_metrics = model.sequential_predictions(
                x,
                sigmas,
                target_y=y,
                return_noise_correction_metrics=True,
            )
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
            if correction_metrics["clean_rmse"].numel() > 0:
                clean_rmse = correction_metrics["clean_rmse"]
                added_sigma = correction_metrics["added_sigma"]
                saturated = correction_metrics["saturated"]
                finite = torch.isfinite(clean_rmse)
                transition_count = finite.sum(dim=1).to(dtype=sigmas.dtype)
                correction_clean_rmse_sum[: clean_rmse.shape[0]] += (
                    clean_rmse.nan_to_num(nan=0.0) * finite
                ).sum(dim=1)
                correction_added_sigma_sum[: added_sigma.shape[0]] += (
                    added_sigma.nan_to_num(nan=0.0) * finite
                ).sum(dim=1)
                correction_saturated_sum[: saturated.shape[0]] += (
                    saturated.nan_to_num(nan=0.0) * finite
                ).sum(dim=1)
                correction_count[: clean_rmse.shape[0]] += transition_count
            count += y.numel()
            latent_count += z_clean.numel()

    rows = []
    for block_index in range(model.config.num_blocks):
        if correction_count[block_index] > 0:
            correction_clean_rmse = (
                correction_clean_rmse_sum[block_index] / correction_count[block_index]
            )
            correction_added_sigma = (
                correction_added_sigma_sum[block_index] / correction_count[block_index]
            )
            correction_saturated_fraction = (
                correction_saturated_sum[block_index] / correction_count[block_index]
            )
        else:
            correction_clean_rmse = sigmas.new_tensor(float("nan"))
            correction_added_sigma = sigmas.new_tensor(float("nan"))
            correction_saturated_fraction = sigmas.new_tensor(float("nan"))
        rows.append(
            {
                "split": split,
                "block_index": block_index,
                "network_block_index": model.network_block_index(block_index),
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
                "noise_correction_clean_rmse": float(
                    correction_clean_rmse.detach().cpu()
                ),
                "noise_correction_added_sigma": float(
                    correction_added_sigma.detach().cpu()
                ),
                "noise_correction_saturated_fraction": float(
                    correction_saturated_fraction.detach().cpu()
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
    if config.initial_noise_mode == "per_example":
        noise = torch.randn(
            config.prediction_grid_points,
            model.state_dim,
            generator=generator,
            device="cpu",
        ).to(sigmas.device)
    elif config.initial_noise_mode == "shared_per_batch":
        noise = torch.randn(
            1,
            model.state_dim,
            generator=generator,
            device="cpu",
        ).to(sigmas.device).expand(config.prediction_grid_points, -1)
    elif config.initial_noise_mode == "fixed_shared":
        noise = model.fixed_initial_noise.to(sigmas.device).expand(
            config.prediction_grid_points,
            -1,
        )
    else:
        raise ValueError(f"Unknown initial_noise_mode: {config.initial_noise_mode}")
    with torch.no_grad():
        target = target_function(x, config.function)
        predictions, _ = model.sequential_predictions(
            x,
            sigmas,
            target_y=target,
            noise=noise,
            prediction_mode="clean_estimate",
        )

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


def evaluate_prediction_uncertainty(
    model: ToyDiffusionBlocks,
    config: ToyConfig,
    sigmas: torch.Tensor,
) -> list[dict]:
    if config.prediction_uncertainty_samples <= 0:
        return []

    model.eval()
    x = torch.linspace(
        -3.2,
        3.2,
        config.prediction_grid_points,
        device=sigmas.device,
    ).view(-1, 1)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 90_000)
    sample_predictions = []
    with torch.no_grad():
        target = target_function(x, config.function)
        for _ in range(config.prediction_uncertainty_samples):
            noise = torch.randn(
                config.prediction_grid_points,
                model.state_dim,
                generator=generator,
                device="cpu",
            ).to(sigmas.device)
            predictions, _ = model.sequential_predictions(
                x,
                sigmas,
                target_y=target,
                noise=noise,
                prediction_mode="clean_estimate",
            )
            sample_predictions.append(predictions[:, :, 0].detach().cpu())

    samples = torch.stack(sample_predictions)
    means = samples.mean(dim=0)
    stds = samples.std(dim=0, unbiased=False)
    cpu_x = x[:, 0].detach().cpu().tolist()
    cpu_target = target[:, 0].detach().cpu().tolist()
    rows = []
    for block_index in range(config.num_blocks):
        for point_index, x_value in enumerate(cpu_x):
            mean = float(means[block_index, point_index])
            std = float(stds[block_index, point_index])
            rows.append(
                {
                    "block_index": block_index,
                    "point_index": point_index,
                    "x": x_value,
                    "target_y": cpu_target[point_index],
                    "prediction_mean": mean,
                    "prediction_std": std,
                    "prediction_lower_1sigma": mean - std,
                    "prediction_upper_1sigma": mean + std,
                    "prediction_lower_2sigma": mean - 2.0 * std,
                    "prediction_upper_2sigma": mean + 2.0 * std,
                }
            )
    return rows


def evaluate_prediction_uncertainty_samples(
    model: ToyDiffusionBlocks,
    config: ToyConfig,
    sigmas: torch.Tensor,
) -> list[dict]:
    if config.prediction_uncertainty_samples <= 0:
        return []

    model.eval()
    x = torch.linspace(
        -3.2,
        3.2,
        config.prediction_grid_points,
        device=sigmas.device,
    ).view(-1, 1)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 90_000)
    rows = []
    cpu_x = x[:, 0].detach().cpu().tolist()
    with torch.no_grad():
        for sample_index in range(config.prediction_uncertainty_samples):
            noise = torch.randn(
                config.prediction_grid_points,
                model.state_dim,
                generator=generator,
                device="cpu",
            ).to(sigmas.device)
            predictions, _ = model.sequential_predictions(
                x,
                sigmas,
                target_y=target_function(x, config.function),
                noise=noise,
                prediction_mode="clean_estimate",
            )
            cpu_predictions = predictions[:, :, 0].detach().cpu().tolist()
            for block_index, block_predictions in enumerate(cpu_predictions):
                for point_index, (x_value, prediction_value) in enumerate(
                    zip(cpu_x, block_predictions)
                ):
                    rows.append(
                        {
                            "sample_index": sample_index,
                            "block_index": block_index,
                            "point_index": point_index,
                            "x": x_value,
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


def write_eval_cycle_latent_distance_csv(
    history: list[dict],
    path: str,
    split: str,
) -> None:
    rows = []
    block_rows_key = f"{split}_block_rows"
    for item in history:
        epoch = item["epoch"]
        for row in sorted(item[block_rows_key], key=lambda r: r["block_index"]):
            rows.append(
                {
                    "epoch": epoch,
                    "block_index": row["block_index"],
                    f"{split}_sequential_latent_rmse": row[
                        "sequential_latent_rmse"
                    ],
                    f"{split}_oracle_latent_rmse": row["oracle_latent_rmse"],
                }
            )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "block_index",
                f"{split}_sequential_latent_rmse",
                f"{split}_oracle_latent_rmse",
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
                "network_block_index",
                "block_objective",
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


def write_prediction_uncertainty_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "block_index",
                "point_index",
                "x",
                "target_y",
                "prediction_mean",
                "prediction_std",
                "prediction_lower_1sigma",
                "prediction_upper_1sigma",
                "prediction_lower_2sigma",
                "prediction_upper_2sigma",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_uncertainty_plot(
    rows: list[dict],
    config: ToyConfig,
    sample_rows: list[dict],
    train_data: TensorDataset,
    test_data: TensorDataset,
    png_path: str,
    svg_path: str,
) -> bool:
    if not rows:
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    blocks = sorted({int(row["block_index"]) for row in rows})
    by_block = {block: [] for block in blocks}
    for row in rows:
        by_block[int(row["block_index"])].append(row)
    for block_rows in by_block.values():
        block_rows.sort(key=lambda row: int(row["point_index"]))
    sample_by_block = {block: {} for block in blocks}
    for row in sample_rows:
        block = int(row["block_index"])
        if block not in sample_by_block:
            continue
        sample_index = int(row["sample_index"])
        sample_by_block[block].setdefault(sample_index, []).append(
            (float(row["x"]), float(row["sequential_prediction"]))
        )
    for block_samples in sample_by_block.values():
        for points in block_samples.values():
            points.sort(key=lambda point: point[0])
    show_sample_predictions = (
        config.function == "sin_cos_bifurcation"
        and any(sample_by_block[block] for block in blocks)
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
    target_curve_sets = target_curve_point_sets(
        [float(row["x"]) for row in rows],
        config.function,
    )
    y_values = (
        [y for curve in target_curve_sets for _, y in curve["points"]]
        + [y for _, y in train_points + test_points]
    )
    if show_sample_predictions:
        y_values += [
            y
            for block_samples in sample_by_block.values()
            for points in block_samples.values()
            for _, y in points
        ]
    else:
        y_values += (
            [float(row["prediction_lower_2sigma"]) for row in rows]
            + [float(row["prediction_upper_2sigma"]) for row in rows]
        )
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
    top = 74
    gap_x = 34
    gap_y = 48
    right = 160
    bottom = 64
    width = left + columns * panel_w + (columns - 1) * gap_x + right
    height = top + panel_rows * panel_h + (panel_rows - 1) * gap_y + bottom

    def sx(x: float, panel_left: float) -> float:
        if x_max == x_min:
            return panel_left + panel_w / 2
        return panel_left + (x - x_min) / (x_max - x_min) * panel_w

    def sy(y: float, panel_top: float) -> float:
        if y_max == y_min:
            return panel_top + panel_h / 2
        return panel_top + (y_max - y) / (y_max - y_min) * panel_h

    def line_points(block: int, field: str, panel_left: float, panel_top: float):
        return [
            (sx(float(row["x"]), panel_left), sy(float(row[field]), panel_top))
            for row in by_block[block]
        ]

    def band_polygon(
        block: int,
        lower_field: str,
        upper_field: str,
        panel_left: float,
        panel_top: float,
    ):
        upper = line_points(block, upper_field, panel_left, panel_top)
        lower = line_points(block, lower_field, panel_left, panel_top)
        return upper + list(reversed(lower))

    image = Image.new("RGB", (width, height), "white")
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    overlay_draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 12
        )
        small_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 11
        )
        font_bold = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14
        )
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22
        )
    except OSError:
        font = small_font = font_bold = title_font = ImageFont.load_default()

    draw.text(
        (left, 26),
        "Prediction Distribution by Initial Noise",
        font=title_font,
        fill=(17, 24, 39),
    )

    for panel_index, block in enumerate(blocks):
        row_index = panel_index // columns
        col_index = panel_index % columns
        panel_left = left + col_index * (panel_w + gap_x)
        panel_top = top + row_index * (panel_h + gap_y)
        panel_right = panel_left + panel_w
        panel_bottom = panel_top + panel_h

        for tick in range(4):
            yy = panel_top + tick * panel_h / 3
            draw.line(
                (panel_left, yy, panel_right, yy),
                fill=(229, 231, 235),
                width=1,
            )
        for tick in range(4):
            xx = panel_left + tick * panel_w / 3
            draw.line(
                (xx, panel_top, xx, panel_bottom),
                fill=(229, 231, 235),
                width=1,
            )
        draw.rectangle(
            (panel_left, panel_top, panel_right, panel_bottom),
            outline=(17, 24, 39),
            width=1,
        )
        draw.text(
            (panel_left + 6, panel_top + 6),
            f"block {block}",
            font=font_bold,
            fill=(17, 24, 39),
        )
        if row_index == panel_rows - 1:
            draw.text(
                (panel_left, panel_bottom + 8),
                f"{x_min:.1f}",
                font=small_font,
                fill=(17, 24, 39),
            )
            draw.text(
                (panel_right, panel_bottom + 8),
                f"{x_max:.1f}",
                font=small_font,
                fill=(17, 24, 39),
                anchor="ra",
            )
        if col_index == 0:
            draw.text(
                (panel_left - 8, panel_top - 5),
                f"{y_max:.2g}",
                font=small_font,
                fill=(17, 24, 39),
                anchor="ra",
            )
            draw.text(
                (panel_left - 8, panel_bottom - 8),
                f"{y_min:.2g}",
                font=small_font,
                fill=(17, 24, 39),
                anchor="ra",
            )

        if show_sample_predictions:
            for sample_points in sample_by_block[block].values():
                prediction_points = [
                    (sx(x, panel_left), sy(y, panel_top))
                    for x, y in sample_points
                ]
                if len(prediction_points) > 1:
                    overlay_draw.line(
                        prediction_points,
                        fill=(37, 99, 235, 42),
                        width=1,
                        joint="curve",
                    )
        else:
            overlay_draw.polygon(
                band_polygon(
                    block,
                    "prediction_lower_2sigma",
                    "prediction_upper_2sigma",
                    panel_left,
                    panel_top,
                ),
                fill=(191, 219, 254, 115),
            )
            overlay_draw.polygon(
                band_polygon(
                    block,
                    "prediction_lower_1sigma",
                    "prediction_upper_1sigma",
                    panel_left,
                    panel_top,
                ),
                fill=(96, 165, 250, 135),
            )
        mean_points = line_points(block, "prediction_mean", panel_left, panel_top)
        for curve in target_curve_sets:
            target_points = [
                (sx(x, panel_left), sy(y, panel_top))
                for x, y in curve["points"]
            ]
            draw.line(target_points, fill=curve["color"], width=3, joint="curve")
        draw.line(mean_points, fill=(37, 99, 235), width=2, joint="curve")
        for x, y in train_points:
            px = sx(x, panel_left)
            py = sy(y, panel_top)
            draw.ellipse(
                (px - 3, py - 3, px + 3, py + 3),
                fill=(22, 163, 74),
                outline=(20, 83, 45),
                width=1,
            )
        for x, y in test_points:
            px = sx(x, panel_left)
            py = sy(y, panel_top)
            draw.ellipse(
                (px - 3, py - 3, px + 3, py + 3),
                fill=(250, 204, 21),
                outline=(113, 63, 18),
                width=1,
            )

    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)
    for panel_index, block in enumerate(blocks):
        row_index = panel_index // columns
        col_index = panel_index % columns
        panel_left = left + col_index * (panel_w + gap_x)
        panel_top = top + row_index * (panel_h + gap_y)
        mean_points = line_points(block, "prediction_mean", panel_left, panel_top)
        for curve in target_curve_sets:
            target_points = [
                (sx(x, panel_left), sy(y, panel_top))
                for x, y in curve["points"]
            ]
            draw.line(target_points, fill=curve["color"], width=3, joint="curve")
        draw.line(mean_points, fill=(37, 99, 235), width=2, joint="curve")
        for x, y in train_points:
            px = sx(x, panel_left)
            py = sy(y, panel_top)
            draw.ellipse(
                (px - 3, py - 3, px + 3, py + 3),
                fill=(22, 163, 74),
                outline=(20, 83, 45),
                width=1,
            )
        for x, y in test_points:
            px = sx(x, panel_left)
            py = sy(y, panel_top)
            draw.ellipse(
                (px - 3, py - 3, px + 3, py + 3),
                fill=(250, 204, 21),
                outline=(113, 63, 18),
                width=1,
            )

    key_x = width - right + 34
    key_y = top + 4
    if show_sample_predictions:
        draw.line(
            (key_x, key_y + 8, key_x + 24, key_y + 8),
            fill=(37, 99, 235),
            width=1,
        )
        draw.text((key_x + 32, key_y), "samples", font=font, fill=(17, 24, 39))
        draw.line(
            (key_x, key_y + 32, key_x + 24, key_y + 32),
            fill=(37, 99, 235),
            width=2,
        )
        draw.text((key_x + 32, key_y + 25), "mean", font=font, fill=(17, 24, 39))
        legend_y = key_y + 56
    else:
        draw.rectangle((key_x, key_y, key_x + 24, key_y + 12), fill=(191, 219, 254))
        draw.text((key_x + 32, key_y - 2), "2 sigma", font=font, fill=(17, 24, 39))
        draw.rectangle((key_x, key_y + 24, key_x + 24, key_y + 36), fill=(96, 165, 250))
        draw.text((key_x + 32, key_y + 22), "1 sigma", font=font, fill=(17, 24, 39))
        draw.line((key_x, key_y + 56, key_x + 24, key_y + 56), fill=(37, 99, 235), width=2)
        draw.text((key_x + 32, key_y + 49), "mean", font=font, fill=(17, 24, 39))
        legend_y = key_y + 80
    for curve in target_curve_sets:
        draw.line(
            (key_x, legend_y, key_x + 24, legend_y),
            fill=curve["color"],
            width=3,
        )
        draw.text(
            (key_x + 32, legend_y - 7),
            curve["label"],
            font=font,
            fill=(17, 24, 39),
        )
        legend_y += 24
    draw.ellipse((key_x, legend_y - 4, key_x + 8, legend_y + 4), fill=(22, 163, 74))
    draw.text((key_x + 16, legend_y - 8), "train", font=font, fill=(17, 24, 39))
    legend_y += 22
    draw.ellipse((key_x, legend_y - 4, key_x + 8, legend_y + 4), fill=(250, 204, 21))
    draw.text((key_x + 16, legend_y - 8), "test", font=font, fill=(17, 24, 39))
    image.convert("RGB").save(png_path)

    def svg_points(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    def svg_band(
        block: int,
        lower_field: str,
        upper_field: str,
        panel_left: float,
        panel_top: float,
    ) -> str:
        return svg_points(band_polygon(block, lower_field, upper_field, panel_left, panel_top))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:22px;font-weight:700}.label{font-size:14px;font-weight:700}.tick{font-size:11px}.legend{font-size:12px}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#111827;stroke-width:1}</style>',
        f'<text class="title" x="{left}" y="34">Prediction Distribution by Initial Noise</text>',
    ]
    for panel_index, block in enumerate(blocks):
        row_index = panel_index // columns
        col_index = panel_index % columns
        panel_left = left + col_index * (panel_w + gap_x)
        panel_top = top + row_index * (panel_h + gap_y)
        panel_right = panel_left + panel_w
        panel_bottom = panel_top + panel_h
        for tick in range(4):
            yy = panel_top + tick * panel_h / 3
            svg.append(
                f'<line class="grid" x1="{panel_left}" y1="{yy:.2f}" x2="{panel_right}" y2="{yy:.2f}"/>'
            )
            xx = panel_left + tick * panel_w / 3
            svg.append(
                f'<line class="grid" x1="{xx:.2f}" y1="{panel_top}" x2="{xx:.2f}" y2="{panel_bottom}"/>'
            )
        svg.append(
            f'<rect class="axis" x="{panel_left}" y="{panel_top}" width="{panel_w}" height="{panel_h}" fill="none"/>'
        )
        svg.append(
            f'<text class="label" x="{panel_left + 6}" y="{panel_top + 18}">block {block}</text>'
        )
        if show_sample_predictions:
            for sample_points in sample_by_block[block].values():
                prediction_points = [
                    (sx(x, panel_left), sy(y, panel_top))
                    for x, y in sample_points
                ]
                if len(prediction_points) > 1:
                    svg.append(
                        f'<polyline points="{svg_points(prediction_points)}" fill="none" stroke="#2563eb" stroke-width="1" opacity="0.18"/>'
                    )
        else:
            svg.append(
                f'<polygon points="{svg_band(block, "prediction_lower_2sigma", "prediction_upper_2sigma", panel_left, panel_top)}" fill="#bfdbfe" opacity="0.45"/>'
            )
            svg.append(
                f'<polygon points="{svg_band(block, "prediction_lower_1sigma", "prediction_upper_1sigma", panel_left, panel_top)}" fill="#60a5fa" opacity="0.55"/>'
            )
        for curve in target_curve_sets:
            target_points = [
                (sx(x, panel_left), sy(y, panel_top))
                for x, y in curve["points"]
            ]
            svg.append(
                f'<polyline points="{svg_points(target_points)}" fill="none" stroke="{curve["hex_color"]}" stroke-width="3"/>'
            )
        svg.append(
            f'<polyline points="{svg_points(line_points(block, "prediction_mean", panel_left, panel_top))}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        )
        for x, y in train_points:
            svg.append(
                f'<circle cx="{sx(x, panel_left):.2f}" cy="{sy(y, panel_top):.2f}" r="3" fill="#16a34a" stroke="#14532d" stroke-width="0.8"/>'
            )
        for x, y in test_points:
            svg.append(
                f'<circle cx="{sx(x, panel_left):.2f}" cy="{sy(y, panel_top):.2f}" r="3" fill="#facc15" stroke="#713f12" stroke-width="0.8"/>'
            )
    key_x = width - right + 34
    key_y = top + 4
    if show_sample_predictions:
        svg.extend(
            [
                f'<line x1="{key_x}" y1="{key_y + 8}" x2="{key_x + 24}" y2="{key_y + 8}" stroke="#2563eb" stroke-width="1" opacity="0.35"/>',
                f'<text class="legend" x="{key_x + 32}" y="{key_y + 12}">samples</text>',
                f'<line x1="{key_x}" y1="{key_y + 32}" x2="{key_x + 24}" y2="{key_y + 32}" stroke="#2563eb" stroke-width="2"/>',
                f'<text class="legend" x="{key_x + 32}" y="{key_y + 36}">mean</text>',
            ]
        )
        legend_y = key_y + 56
    else:
        svg.extend(
            [
                f'<rect x="{key_x}" y="{key_y}" width="24" height="12" fill="#bfdbfe" opacity="0.75"/>',
                f'<text class="legend" x="{key_x + 32}" y="{key_y + 10}">2 sigma</text>',
                f'<rect x="{key_x}" y="{key_y + 24}" width="24" height="12" fill="#60a5fa" opacity="0.85"/>',
                f'<text class="legend" x="{key_x + 32}" y="{key_y + 34}">1 sigma</text>',
                f'<line x1="{key_x}" y1="{key_y + 56}" x2="{key_x + 24}" y2="{key_y + 56}" stroke="#2563eb" stroke-width="2"/>',
                f'<text class="legend" x="{key_x + 32}" y="{key_y + 60}">mean</text>',
            ]
        )
        legend_y = key_y + 80
    for curve in target_curve_sets:
        svg.append(
            f'<line x1="{key_x}" y1="{legend_y}" x2="{key_x + 24}" y2="{legend_y}" stroke="{curve["hex_color"]}" stroke-width="3"/>'
        )
        svg.append(
            f'<text class="legend" x="{key_x + 32}" y="{legend_y + 4}">{curve["label"]}</text>'
        )
        legend_y += 24
    svg.append(
        f'<circle cx="{key_x + 4}" cy="{legend_y}" r="4" fill="#16a34a" stroke="#14532d" stroke-width="0.8"/>'
    )
    svg.append(
        f'<text class="legend" x="{key_x + 16}" y="{legend_y + 4}">train</text>'
    )
    legend_y += 22
    svg.append(
        f'<circle cx="{key_x + 4}" cy="{legend_y}" r="4" fill="#facc15" stroke="#713f12" stroke-width="0.8"/>'
    )
    svg.append(
        f'<text class="legend" x="{key_x + 16}" y="{legend_y + 4}">test</text>'
    )
    svg.append("</svg>")
    with open(svg_path, "w") as f:
        f.write("\n".join(svg))
        f.write("\n")
    return True


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
    split: str,
) -> bool:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rows = []
    block_rows_key = f"{split}_block_rows"
    for item in history:
        for row in item.get(block_rows_key, []):
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

    split_title = split.capitalize()
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
    ax.set_ylabel(f"{split_title} sequential latent RMSE to clean")
    ax.set_title(f"{split_title} Latent Distance by Block Across Eval Cycles")
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
    config: ToyConfig,
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
    for row in rows:
        key = (int(row["block_index"]), int(row["epoch"]))
        by_block_epoch[key].append(
            (
                float(row["x"]),
                float(row["sequential_prediction"]),
            )
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
    target_curve_sets = target_curve_point_sets(
        [float(row["x"]) for row in rows],
        config.function,
    )
    y_values = [y for curve in target_curve_sets for _, y in curve["points"]] + [
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

        for curve in target_curve_sets:
            target_points = [
                (sx(x, px), sy(y, py))
                for x, y in curve["points"]
            ]
            if len(target_points) > 1:
                draw.line(target_points, fill=curve["color"], width=3)

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
    colorbar_h = min(420, max(90, height - colorbar_y - 160))
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
    target_key_y = key_y + 44
    for curve in target_curve_sets:
        draw.line(
            (colorbar_x, target_key_y, colorbar_x + 20, target_key_y),
            fill=curve["color"],
            width=3,
        )
        draw.text(
            (colorbar_x + 26, target_key_y - 7),
            curve["label"],
            font=font,
            fill=(17, 24, 39),
        )
        target_key_y += 22
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
        for curve in target_curve_sets:
            target_points = " ".join(
                f"{sx(x, px):.2f},{sy(y, py):.2f}"
                for x, y in curve["points"]
            )
            svg.append(
                f'<polyline points="{target_points}" fill="none" stroke="{curve["hex_color"]}" stroke-width="2.6"/>'
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
    target_key_y = key_y + 44
    for curve in target_curve_sets:
        svg.append(
            f'<line x1="{colorbar_x}" y1="{target_key_y}" x2="{colorbar_x + 20}" y2="{target_key_y}" stroke="{curve["hex_color"]}" stroke-width="3"/>'
        )
        svg.append(
            f'<text class="epoch" x="{colorbar_x + 26}" y="{target_key_y + 4}">{curve["label"]}</text>'
        )
        target_key_y += 22
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
        target = target_function(x, config.function)
        seq_preds, _ = model.sequential_predictions(x, sigmas, target_y=target)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    train_x, train_y = train_data.tensors
    test_x, test_y = test_data.tensors
    for index, (label, y_true) in enumerate(target_curves(x, config.function)):
        ax.plot(
            x.cpu(),
            y_true.cpu(),
            color=target_curve_hex_color(index),
            linewidth=2,
            label=label,
        )
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

    train_data = make_dataset(
        config.num_train,
        config,
        seed_offset=0,
        include_all_bifurcation_branches=True,
    )
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
                        "network_block_index": model.network_block_index(block_index),
                        "block_objective": model.block_objective(block_index),
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

    prediction_uncertainty_rows = evaluate_prediction_uncertainty(
        model,
        config,
        sigmas,
    )
    prediction_uncertainty_sample_rows = evaluate_prediction_uncertainty_samples(
        model,
        config,
        sigmas,
    )
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
    train_eval_cycle_latent_csv_path = os.path.join(
        run_dir,
        "train_latent_distance_by_block_eval_cycles.csv",
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
    train_eval_cycle_latent_png_path = os.path.join(
        run_dir,
        "train_latent_distance_by_block_eval_cycles.png",
    )
    train_eval_cycle_latent_svg_path = os.path.join(
        run_dir,
        "train_latent_distance_by_block_eval_cycles.svg",
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
    prediction_uncertainty_csv_path = os.path.join(
        run_dir,
        "prediction_uncertainty_by_block.csv",
    )
    prediction_uncertainty_png_path = os.path.join(
        run_dir,
        "prediction_uncertainty_by_block.png",
    )
    prediction_uncertainty_svg_path = os.path.join(
        run_dir,
        "prediction_uncertainty_by_block.svg",
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
    write_eval_cycle_latent_distance_csv(
        history,
        eval_cycle_latent_csv_path,
        "test",
    )
    write_eval_cycle_latent_distance_csv(
        history,
        train_eval_cycle_latent_csv_path,
        "train",
    )
    write_eval_cycle_train_loss_csv(history, train_loss_csv_path)
    write_prediction_curve_csv(history, prediction_curve_csv_path)
    write_prediction_uncertainty_csv(
        prediction_uncertainty_rows,
        prediction_uncertainty_csv_path,
    )
    wrote_block_plot = False
    wrote_latent_plot = False
    wrote_pred_plot = False
    wrote_eval_cycle_plot = False
    wrote_eval_cycle_latent_plot = False
    wrote_train_eval_cycle_latent_plot = False
    wrote_train_loss_plot = False
    wrote_prediction_curve_plot = False
    wrote_prediction_uncertainty_plot = False
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
            "test",
        )
        wrote_train_eval_cycle_latent_plot = write_eval_cycle_latent_distance_plot(
            history,
            train_eval_cycle_latent_png_path,
            train_eval_cycle_latent_svg_path,
            "train",
        )
        wrote_train_loss_plot = write_eval_cycle_train_loss_plot(
            history,
            train_loss_png_path,
            train_loss_svg_path,
        )
        wrote_prediction_curve_plot = write_prediction_curve_eval_cycle_plot(
            history,
            config,
            train_data,
            test_data,
            prediction_curve_png_path,
            prediction_curve_svg_path,
        )
        wrote_prediction_uncertainty_plot = write_prediction_uncertainty_plot(
            prediction_uncertainty_rows,
            config,
            prediction_uncertainty_sample_rows,
            train_data,
            test_data,
            prediction_uncertainty_png_path,
            prediction_uncertainty_svg_path,
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
    print(f"Wrote {train_eval_cycle_latent_csv_path}")
    print(f"Wrote {train_loss_csv_path}")
    print(f"Wrote {prediction_curve_csv_path}")
    if prediction_uncertainty_rows:
        print(f"Wrote {prediction_uncertainty_csv_path}")
    else:
        print("Skipped prediction uncertainty CSV because no samples were requested")
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
        print("Skipped test eval-cycle latent-distance plot because matplotlib is not installed")
    if wrote_train_eval_cycle_latent_plot:
        print(f"Wrote {train_eval_cycle_latent_png_path}")
        print(f"Wrote {train_eval_cycle_latent_svg_path}")
    else:
        print("Skipped train eval-cycle latent-distance plot because matplotlib is not installed")
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
    if wrote_prediction_uncertainty_plot:
        print(f"Wrote {prediction_uncertainty_png_path}")
        print(f"Wrote {prediction_uncertainty_svg_path}")
    else:
        print("Skipped prediction uncertainty plot because Pillow is not installed")
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
        choices=["clean_latent", "residual_next_latent", "residual_to_clean"],
        help=(
            "clean_latent keeps the original toy objective where each block "
            "predicts the clean latent; residual_next_latent makes each block "
            "predict delta_hat and applies z_next_hat = z + delta_hat toward "
            "the next scheduled latent state; residual_to_clean makes each "
            "block predict r_hat = z_clean - z and updates with "
            "z_next = z + alpha * r_hat"
        ),
    )
    parser.add_argument(
        "--block_objective_pattern",
        type=str,
        default=ToyConfig.block_objective_pattern,
        choices=[
            "global",
            "all_prediction",
            "all_residual",
            "first_prediction_then_residual",
            "alternating_prediction_residual",
        ],
        help=(
            "per-block training objective schedule; global preserves "
            "--training_objective, all_prediction trains every block with "
            "decoded prediction MSE, all_residual trains every block with "
            "residual-next-latent loss, first_prediction_then_residual trains "
            "block 0 with prediction loss and later blocks with residual loss, "
            "and alternating_prediction_residual uses prediction loss on even "
            "blocks and residual loss on odd blocks"
        ),
    )
    parser.add_argument(
        "--denoising_target",
        type=str,
        default=ToyConfig.denoising_target,
        choices=["embedding", "value"],
        help=(
            "space to denoise; embedding uses target_encoder/target_decoder, "
            "value denoises the scalar function value directly"
        ),
    )
    parser.add_argument(
        "--residual_next_latent_training_mode",
        type=str,
        default=ToyConfig.residual_next_latent_training_mode,
        choices=["sequential", "independent"],
        help=(
            "training mode for residual objectives; sequential rolls the "
            "denoising chain forward, independent trains each block from "
            "z_clean + sigma * epsilon"
        ),
    )
    parser.add_argument(
        "--shared_denoising_block",
        action="store_true",
        help=(
            "route every denoising/noise step through the same block network "
            "instead of using one network per sigma level"
        ),
    )
    parser.add_argument(
        "--shared_denoising_block_index",
        type=int,
        default=ToyConfig.shared_denoising_block_index,
        help=(
            "which block network to reuse when --shared_denoising_block is set"
        ),
    )
    parser.add_argument(
        "--shared_denoising_block_ranges",
        type=str,
        default=ToyConfig.shared_denoising_block_ranges,
        help=(
            "comma-separated half-open block sharing ranges. Each entry is "
            "start:end or start:end:network_index; start:end reuses the "
            "network at start for denoising block indices start through end-1, "
            "for example 0:4,4:8,8:12"
        ),
    )
    parser.add_argument(
        "--initial_noise_mode",
        type=str,
        default=ToyConfig.initial_noise_mode,
        choices=["per_example", "shared_per_batch", "fixed_shared"],
        help=(
            "initial sequential denoising noise mode; per_example samples one "
            "noise vector per input, shared_per_batch samples one vector per "
            "batch and shares it across inputs, fixed_shared reuses one seeded "
            "vector for all inputs and calls"
        ),
    )
    parser.add_argument(
        "--initial_noise_std",
        type=float,
        default=ToyConfig.initial_noise_std,
        help=(
            "std for the initial sequential denoising state; defaults to "
            "sqrt(1 + sigma[0]^2) when unset"
        ),
    )
    parser.add_argument(
        "--interblock_transition",
        type=str,
        default=ToyConfig.interblock_transition,
        choices=[
            "euler",
            "direct_denoised",
            "direct_denoised_plus_noise",
            "direct_denoised_plus_rescaled_noise",
        ],
        help=(
            "state transition for prediction-objective toy DBlock steps. "
            "euler keeps the scheduled diffusion update; direct_denoised feeds "
            "the predicted clean latent directly to the next block; "
            "direct_denoised_plus_noise re-noises it with fresh noise at the "
            "next sigma; direct_denoised_plus_rescaled_noise reuses the same "
            "initial noise direction and rescales it by the next sigma"
        ),
    )
    parser.add_argument(
        "--noise_correction_mode",
        type=str,
        default=ToyConfig.noise_correction_mode,
        choices=["none", "batch_oracle"],
        help=(
            "inter-block noise correction mode. none uses the scheduled Euler "
            "transition. batch_oracle estimates each block's clean-latent RMSE "
            "against the current batch targets and re-noises the clean estimate "
            "with sqrt(max(next_sigma^2 - rmse^2, 0))"
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
        choices=["sine_poly", "saw_sine", "cubic", "sin_cos_bifurcation"],
    )
    parser.add_argument("--seed", type=int, default=ToyConfig.seed)
    parser.add_argument("--output_dir", type=str, default=ToyConfig.output_dir)
    parser.add_argument("--eval_every", type=int, default=ToyConfig.eval_every)
    parser.add_argument(
        "--prediction_grid_points",
        type=int,
        default=ToyConfig.prediction_grid_points,
    )
    parser.add_argument(
        "--prediction_uncertainty_samples",
        type=int,
        default=ToyConfig.prediction_uncertainty_samples,
        help=(
            "number of random initial-noise draws for the final prediction "
            "uncertainty plot; set to 0 to skip"
        ),
    )
    parser.add_argument("--device", type=str, default=ToyConfig.device)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()
    config = ToyConfig(**vars(args))
    if config.num_blocks < 1:
        raise ValueError("--num_blocks must be at least 1")
    if config.shared_denoising_block and config.shared_denoising_block_ranges:
        raise ValueError(
            "--shared_denoising_block and --shared_denoising_block_ranges are "
            "mutually exclusive"
        )
    if config.shared_denoising_block_index < 0:
        raise ValueError("--shared_denoising_block_index must be non-negative")
    if config.shared_denoising_block_index >= config.num_blocks:
        raise ValueError(
            "--shared_denoising_block_index must be smaller than --num_blocks"
        )
    parse_shared_denoising_block_ranges(
        config.shared_denoising_block_ranges,
        config.num_blocks,
    )
    if config.sigma_min <= 0 or config.sigma_max <= 0:
        raise ValueError("--sigma_min and --sigma_max must be positive")
    if config.sigma_min >= config.sigma_max:
        raise ValueError("--sigma_min must be smaller than --sigma_max")
    if config.prediction_grid_points < 2:
        raise ValueError("--prediction_grid_points must be at least 2")
    if config.prediction_uncertainty_samples < 0:
        raise ValueError("--prediction_uncertainty_samples must be non-negative")
    if config.observation_noise_std < 0.0:
        raise ValueError("--observation_noise_std must be non-negative")
    if config.initial_noise_std is not None and config.initial_noise_std < 0.0:
        raise ValueError("--initial_noise_std must be non-negative when set")
    if (
        config.interblock_transition != "euler"
        and config.noise_correction_mode != "none"
    ):
        raise ValueError(
            "--interblock_transition direct modes require "
            "--noise_correction_mode none"
        )
    if config.prediction_loss_weight < 0.0:
        raise ValueError("--prediction_loss_weight must be non-negative")
    if config.latent_loss_weight < 0.0:
        raise ValueError("--latent_loss_weight must be non-negative")
    if (
        config.training_objective
        not in ["residual_next_latent", "residual_to_clean"]
        and config.residual_next_latent_training_mode != "sequential"
    ):
        raise ValueError(
            "--residual_next_latent_training_mode independent requires a "
            "residual --training_objective"
        )
    if (
        config.block_objective_pattern != "global"
        and config.residual_next_latent_training_mode != "sequential"
    ):
        raise ValueError(
            "--block_objective_pattern values other than global currently "
            "require --residual_next_latent_training_mode sequential"
        )
    if (
        config.denoising_target == "value"
        and config.train_encoder_with_prediction_loss_only
    ):
        raise ValueError(
            "--train_encoder_with_prediction_loss_only requires "
            "--denoising_target embedding"
        )
    return config


if __name__ == "__main__":
    train(parse_args())
