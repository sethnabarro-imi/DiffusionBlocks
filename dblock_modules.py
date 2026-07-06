from __future__ import annotations

import numpy as np
from scipy.stats import norm
import torch


def alpha_bar_from_sigma(sigma: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.square(sigma))


def sigma_from_alpha_bar(alpha_bar: np.ndarray | float) -> np.ndarray | float:
    return np.sqrt((1.0 - alpha_bar) / alpha_bar)


def cosine_alpha_bar_from_progress(
    progress: np.ndarray | float,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> np.ndarray | float:
    alpha_start = alpha_bar_from_sigma(sigma_min)
    alpha_end = alpha_bar_from_sigma(sigma_max)
    signal_weight = np.cos(progress * np.pi / 2.0) ** 2
    return alpha_end + (alpha_start - alpha_end) * signal_weight


def cosine_progress_from_alpha_bar(
    alpha_bar: np.ndarray | float,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> np.ndarray | float:
    alpha_start = alpha_bar_from_sigma(sigma_min)
    alpha_end = alpha_bar_from_sigma(sigma_max)
    signal_weight = (alpha_bar - alpha_end) / (alpha_start - alpha_end)
    signal_weight = np.clip(signal_weight, 0.0, 1.0)
    return (2.0 / np.pi) * np.arccos(np.sqrt(signal_weight))


def get_alpha_bar_schedule_sigmas(
    num_points: int,
    schedule: str,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
) -> np.ndarray:
    if num_points < 1:
        raise ValueError("num_points must be at least 1")
    if schedule not in ["linear", "cosine"]:
        raise ValueError("schedule must be one of linear or cosine")
    if sigma_min <= 0.0:
        raise ValueError("sigma_min must be positive")
    if sigma_max <= sigma_min:
        raise ValueError("sigma_max must be greater than sigma_min")

    progress = np.linspace(0.0, 1.0, num_points)
    alpha_start = alpha_bar_from_sigma(sigma_min)
    alpha_end = alpha_bar_from_sigma(sigma_max)
    if schedule == "linear":
        alpha_bar = alpha_start + progress * (alpha_end - alpha_start)
    else:
        # NoProp-DT uses a fixed cosine noise-level schedule. We keep the same
        # cosine signal-power shape while matching the configured sigma bounds.
        alpha_bar = cosine_alpha_bar_from_progress(
            progress,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    return sigma_from_alpha_bar(alpha_bar)


def get_block_sigmas(
    num_layers,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    schedule: str = "edm",
) -> list[float]:
    if schedule in ["linear", "cosine"]:
        return get_alpha_bar_schedule_sigmas(
            num_layers + 1,
            schedule=schedule,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        ).tolist()
    if schedule != "edm":
        raise ValueError("schedule must be one of edm, linear, or cosine")
    cdf_min = norm.cdf((np.log(sigma_min) - p_mean) / p_std)
    cdf_max = norm.cdf((np.log(sigma_max) - p_mean) / p_std)
    block_sigmas = []
    for i in range(num_layers + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / num_layers)
        sigma = np.exp(p_mean + p_std * norm.ppf(p))
        block_sigmas.append(sigma)
    return block_sigmas


def get_discrete_sigmas(
    num_steps,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    p_mean=-1.2,
    p_std=1.2,
    dblock=False,
    schedule="edm",
):
    if schedule in ["linear", "cosine"]:
        sigmas = get_alpha_bar_schedule_sigmas(
            num_steps,
            schedule=schedule,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        sigmas = torch.tensor(sigmas, dtype=torch.float32)
        return torch.flip(sigmas, dims=[0])
    if schedule != "edm":
        raise ValueError("schedule must be one of edm, linear, or cosine")
    if not dblock:
        ramp = torch.linspace(0, 1, num_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas
    else:
        log_sigma_min = np.log(sigma_min)
        log_sigma_max = np.log(sigma_max)
        cdf_min = norm.cdf((log_sigma_min - p_mean) / p_std)
        cdf_max = norm.cdf((log_sigma_max - p_mean) / p_std)
        cdf_points = np.linspace(cdf_min, cdf_max, num_steps)
        sigmas = np.exp(p_mean + p_std * norm.ppf(cdf_points))
        sigmas = torch.tensor(sigmas, dtype=torch.float32)
        sigmas = torch.flip(sigmas, dims=[0])
        return sigmas
