import random
import numpy as np
from scipy.stats import norm
import torch
import torch.nn.functional as F
import lightning as L
import torchmetrics
from transformers import get_scheduler

from vit import load_vit
from dblock_modules import get_block_sigmas, get_discrete_sigmas


class ExpectedCalibrationError(torchmetrics.Metric):
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_bins: int = 15):
        super().__init__()
        self.num_bins = num_bins
        self.add_state(
            "confidence_sum",
            default=torch.zeros(num_bins),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "correct_sum",
            default=torch.zeros(num_bins),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.zeros(num_bins),
            dist_reduce_fx="sum",
        )

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        probs = F.softmax(logits.float(), dim=-1)
        confidences, predictions = probs.max(dim=-1)
        labels = labels.view(-1)
        confidences = confidences.view(-1)
        predictions = predictions.view(-1)
        correct = predictions.eq(labels).float()

        bin_indices = torch.clamp(
            (confidences * self.num_bins).long(),
            min=0,
            max=self.num_bins - 1,
        )
        self.confidence_sum.scatter_add_(0, bin_indices, confidences)
        self.correct_sum.scatter_add_(0, bin_indices, correct)
        self.count.scatter_add_(0, bin_indices, torch.ones_like(confidences))

    def compute(self) -> torch.Tensor:
        non_empty = self.count > 0
        safe_count = self.count.clamp_min(1)
        accuracy = self.correct_sum / safe_count
        confidence = self.confidence_sum / safe_count
        weights = self.count / self.count.sum().clamp_min(1)
        return (
            weights[non_empty]
            * (accuracy[non_empty] - confidence[non_empty]).abs()
        ).sum()

    def calibration_stats(self) -> dict[str, torch.Tensor]:
        safe_count = self.count.clamp_min(1)
        accuracy = self.correct_sum / safe_count
        confidence = self.confidence_sum / safe_count
        total = self.count.sum().clamp_min(1)
        edges = torch.linspace(
            0.0,
            1.0,
            self.num_bins + 1,
            device=self.count.device,
            dtype=self.count.dtype,
        )
        return {
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
            "bin_center": (edges[:-1] + edges[1:]) / 2,
            "count": self.count,
            "frequency": self.count / total,
            "accuracy": accuracy,
            "confidence": confidence,
            "gap": (accuracy - confidence).abs(),
        }


class MulticlassLogLikelihood(torchmetrics.Metric):
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "log_likelihood_sum",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        labels = labels.view(-1)
        log_likelihoods = log_probs.gather(1, labels[:, None]).squeeze(1)
        self.log_likelihood_sum += log_likelihoods.sum()
        self.count += labels.numel()

    def compute(self) -> torch.Tensor:
        return self.log_likelihood_sum / self.count.clamp_min(1)


def load_model(args):
    if args.model_type == "vit":
        return ViTModel(args)
    elif args.model_type == "dblock":
        return ViTDBlockModel(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")


class ViTModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = args.image_size
        self.num_labels = args.num_labels
        self.valid_metrics = self.build_eval_metrics("val/")
        self.train_eval_metrics = self.build_eval_metrics("train_eval/")
        self.test_metrics = self.build_eval_metrics("test/")
        self.eval_split = "test"
        self.eval_results_by_split = {}
        self.save_hyperparameters(args)

    def build_eval_metrics(self, prefix: str):
        return torchmetrics.MetricCollection(
            {
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=self.num_labels
                ),
                "f1": torchmetrics.F1Score(
                    task="multiclass", num_classes=self.num_labels
                ),
                "ece": ExpectedCalibrationError(num_bins=self.args.ece_num_bins),
                "log_likelihood": MulticlassLogLikelihood(),
            },
            prefix=prefix,
        )

    def configure_model(self):
        self.model = load_vit(image_size=self.image_size, num_labels=self.num_labels)
        print(self.model)
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = get_scheduler(
            name=self.args.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            scheduler_specific_kwargs=self.args.scheduler_specific_kwargs,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, **kwargs):
        return self.model(**kwargs).logits

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        logits = self(pixel_values=pixel_values, **kwargs)
        if return_metrics:
            logits = logits.view(-1, self.num_labels)
            labels = labels.view(-1)
            if step == "val":
                return self.valid_metrics(logits, labels)
            elif step == "train_eval":
                return self.train_eval_metrics(logits, labels)
            elif step == "test":
                return self.test_metrics(logits, labels)
            else:
                raise NotImplementedError(f"Step {step} is not supported")

        loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        loss_dict = {f"{step}/loss": loss}
        return loss, loss_dict

    def get_model_kwargs(self, batch):
        return {}

    def training_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        loss, loss_dict = self.shared_step(batch, step="train", **model_kwargs)
        self.log_dict(loss_dict, batch_size=batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(batch, step="val", return_metrics=True, **model_kwargs)
        self.log_dict(res, batch_size=batch_size, prog_bar=True)

    def test_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        model_kwargs = self.get_model_kwargs(batch)
        res = self.shared_step(
            batch, step=self.eval_split, return_metrics=True, **model_kwargs
        )
        self.log_dict(res, batch_size=batch_size, prog_bar=True)

    def on_test_epoch_start(self):
        self.get_eval_metrics(self.eval_split).reset()

    def on_test_epoch_end(self):
        metrics = self.get_eval_metrics(self.eval_split)
        split_results = {}
        for metric_name, metric in metrics.items():
            with metric.sync_context():
                value = metric.compute().detach().float().cpu().item()
            split_results[self.strip_metric_prefix(metric_name)] = value

        calibration_metric = self.get_calibration_metric(self.eval_split)
        with calibration_metric.sync_context():
            stats = calibration_metric.calibration_stats()
            split_results["ece_bins"] = self.calibration_rows(stats)

        if self.trainer.is_global_zero:
            self.eval_results_by_split[self.eval_split] = split_results

    def get_eval_metrics(self, split: str):
        if split == "train_eval":
            return self.train_eval_metrics
        if split == "test":
            return self.test_metrics
        if split == "val":
            return self.valid_metrics
        raise NotImplementedError(f"Step {split} is not supported")

    def get_calibration_metric(self, split: str) -> ExpectedCalibrationError:
        for metric in self.get_eval_metrics(split).values():
            if isinstance(metric, ExpectedCalibrationError):
                return metric
        raise RuntimeError(f"No calibration metric configured for {split}")

    def strip_metric_prefix(self, metric_name: str) -> str:
        prefix = f"{self.eval_split}/"
        if metric_name.startswith(prefix):
            return metric_name[len(prefix) :]
        return metric_name

    def calibration_rows(self, stats: dict[str, torch.Tensor]) -> list[dict]:
        cpu_stats = {
            key: value.detach().float().cpu().tolist()
            for key, value in stats.items()
        }
        rows = []
        for idx in range(len(cpu_stats["count"])):
            rows.append(
                {
                    "bin_index": idx,
                    "bin_lower": cpu_stats["bin_lower"][idx],
                    "bin_upper": cpu_stats["bin_upper"][idx],
                    "bin_center": cpu_stats["bin_center"][idx],
                    "count": int(cpu_stats["count"][idx]),
                    "frequency": cpu_stats["frequency"][idx],
                    "accuracy_rate": cpu_stats["accuracy"][idx],
                    "confidence_rate": cpu_stats["confidence"][idx],
                    "gap": cpu_stats["gap"][idx],
                }
            )
        return rows


class ViTDBlockModel(ViTModel):
    def __init__(self, args):
        super().__init__(args)
        self.gamma = args.gamma
        self.sigma_data = 0.5
        self.cfg_scale = args.cfg_scale
        self.class_dropout_prob = (
            args.class_dropout_prob if self.cfg_scale > 0.0 else 0.0
        )
        if self.args.num_prediction_samples < 1:
            raise ValueError("--num_prediction_samples must be at least 1")
        self.num_prediction_samples = self.args.num_prediction_samples
        self.prediction_average = self.args.prediction_average
        self.num_inference_steps = self.args.num_inference_steps or self.args.num_blocks
        self.epsilon_seed = self.args.epsilon_seed
        if self.epsilon_seed is not None and self.epsilon_seed < 0:
            raise ValueError("--epsilon_seed must be non-negative")
        self.block_sigmas = get_block_sigmas(num_layers=self.args.num_blocks)
        self.layer_assignment = None
        self.register_buffer(
            "sigmas",
            get_discrete_sigmas(num_steps=self.num_inference_steps, dblock=True).to(
                self.device
            ),
        )
        self.save_hyperparameters(
            {
                "gamma": self.gamma,
                "num_inference_steps": self.num_inference_steps,
                "num_prediction_samples": self.num_prediction_samples,
                "prediction_average": self.prediction_average,
                "cfg_scale": self.cfg_scale,
                "class_dropout_prob": self.class_dropout_prob,
                "epsilon_seed": self.epsilon_seed,
            },
        )

    def configure_model(self):
        self.model = load_vit(
            image_size=self.image_size, num_labels=self.num_labels, is_dblock=True
        )
        print(self.model)

    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def get_embeds(
        self, input_ids: torch.Tensor, is_input: bool = True
    ) -> torch.Tensor:
        if is_input:
            embeds = self.model.get_input_embeddings()(input_ids)
        else:
            embeds = F.embedding(
                input_ids, weight=self.model.get_output_embeddings().weight
            )
        return self.normalize_embeddings(embeds)

    def get_sigmas(self, n_samples: int, p_mean: float = -1.2, p_std: float = 1.2):
        block_idx = random.choices(range(self.args.num_blocks), k=1)[0]
        sigma_min_block = self.block_sigmas[block_idx]
        sigma_max_block = self.block_sigmas[block_idx + 1]
        # extend the range
        if self.gamma > 0.0:
            log_sigma_min = np.log(sigma_min_block)
            log_sigma_max = np.log(sigma_max_block)
            log_range = log_sigma_max - log_sigma_min
            sigma_min_block = np.exp(log_sigma_min - self.gamma * log_range)
            sigma_max_block = np.exp(log_sigma_max + self.gamma * log_range)
            sigma_min_block = max(sigma_min_block, self.block_sigmas[0])
            sigma_max_block = min(sigma_max_block, self.block_sigmas[-1])

        cdf_min_block = norm.cdf((np.log(sigma_min_block) - p_mean) / p_std)
        cdf_max_block = norm.cdf((np.log(sigma_max_block) - p_mean) / p_std)

        rand = np.random.uniform(cdf_min_block, cdf_max_block, n_samples)
        sigma = np.exp(p_mean + p_std * norm.ppf(rand))
        sigma = torch.from_numpy(sigma)
        return sigma

    def get_weights(self, sigmas):
        return (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data) ** 2

    def sample_epsilon_like(self, reference: torch.Tensor) -> torch.Tensor:
        if self.epsilon_seed is None:
            return torch.randn_like(reference)
        generator = torch.Generator(device="cpu").manual_seed(self.epsilon_seed)
        epsilon = torch.randn(
            reference.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        return epsilon.to(device=reference.device, dtype=reference.dtype)

    def sample_epsilon(
        self,
        shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self.epsilon_seed is None:
            return torch.randn(shape, device=device, dtype=dtype)
        generator = torch.Generator(device="cpu").manual_seed(self.epsilon_seed)
        epsilon = torch.randn(
            shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        return epsilon.to(device=device, dtype=dtype)

    def estimate_target_layer(self, sigma: torch.Tensor) -> int:
        block_sigmas = torch.tensor(self.block_sigmas, device=sigma.device)
        block_idx = torch.bucketize(sigma, block_sigmas, right=True) - 1
        block_idx = (self.args.num_blocks - 1) - block_idx
        block_idx = torch.clamp(block_idx, 0, self.args.num_blocks - 1).long()
        values, counts = block_idx.unique(return_counts=True)
        return values[counts.argmax()].item()

    def denoise(self, x, zt, sigma, block_idx=None):
        if block_idx is None:
            block_idx = self.estimate_target_layer(sigma)
        if self.class_dropout_prob > 0.0 and self.training:
            drop_x = torch.rand(x.shape[0], device=x.device) < self.class_dropout_prob
            uncond_x = torch.zeros_like(x)
            x = torch.where(drop_x[:, None, None, None], uncond_x, x)
        elif not self.training and self.cfg_scale > 0.0:
            uncond_x = torch.zeros_like(x)
            x = torch.cat([uncond_x, x])
            zt = torch.cat([zt] * 2)
            sigma = torch.cat([sigma] * 2)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()

        if self.layer_assignment is None:
            split_size = self.model.config.num_hidden_layers // self.args.num_blocks
            self.layer_assignment = [
                list(range(i * split_size, (i + 1) * split_size))
                for i in range(self.args.num_blocks)
            ]
        outputs = self.model.forward_block(
            layer_indices=self.layer_assignment[block_idx],
            pixel_values=x,
            noisy_embeds=zt * c_in[:, None],
            timesteps=c_noise,
        )
        hidden_states = outputs.last_hidden_state
        conditioning = outputs.conditioning
        model_out = hidden_states * c_out[:, None] + zt * c_skip[:, None]
        logits = self.model.forward_output_embeddings(
            model_out.unsqueeze(1), conditioning
        )
        if not self.training and self.cfg_scale > 0.0:
            logits_uncond, logits_cond = logits.chunk(2)
            logits = logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
        return logits

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        if return_metrics:
            logits = self.diffusion_step(pixel_values)
            logits = logits.view(-1, self.num_labels)
            labels = labels.view(-1)
            if step == "val":
                return self.valid_metrics(logits, labels)
            elif step == "train_eval":
                return self.train_eval_metrics(logits, labels)
            elif step == "test":
                return self.test_metrics(logits, labels)
            else:
                raise NotImplementedError(f"Step {step} is not supported")

        z = self.get_embeds(labels, is_input=True)
        sigmas = self.get_sigmas(z.shape[0])
        block_idx = self.estimate_target_layer(sigmas)
        sigmas = sigmas.to(z)
        zt = z + sigmas[:, None] * self.sample_epsilon_like(z)
        logits = self.denoise(pixel_values, zt, sigmas, block_idx)
        loss = F.cross_entropy(
            logits.view(-1, self.num_labels), labels.view(-1), reduction="none"
        )
        ce_loss = loss.mean()
        w = self.get_weights(sigmas)[:, None]
        loss = (loss * w).mean()

        loss_dict = {
            f"{step}/loss": loss,
            f"{step}/loss_{block_idx}": loss,
            f"{step}/ce_loss": ce_loss,
            f"{step}/ce_loss_{block_idx}": ce_loss,
        }
        return loss, loss_dict

    def diffusion_step(self, x):
        outputs = None
        for _ in range(self.num_prediction_samples):
            logits = self.diffusion_sample(x)
            if self.prediction_average == "probability":
                sample_output = F.softmax(logits.float(), dim=1)
            elif self.prediction_average == "logit":
                sample_output = logits.float()
            else:
                raise ValueError(
                    f"Unsupported prediction_average: {self.prediction_average}"
                )
            outputs = sample_output if outputs is None else outputs + sample_output
        outputs = outputs / self.num_prediction_samples
        if self.prediction_average == "probability":
            # Downstream metrics expect logits; log averaged probabilities preserves
            # the requested probability average because softmax(log p) = p.
            return outputs.clamp_min(torch.finfo(outputs.dtype).tiny).log()
        return outputs

    def diffusion_sample(self, x):
        bsz = x.shape[0]
        hidden_size = self.model.config.hidden_size
        z = self.sample_epsilon((bsz, hidden_size), device=x.device)
        z *= torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x.new_ones([x.shape[0]])
        for i in range(self.sigmas.shape[0] - 1):
            sigma = self.sigmas[i] * s_in
            next_sigma = self.sigmas[i + 1] * s_in
            # denoise
            logits = self.denoise(x, z, sigma)
            probs = F.softmax(logits, dim=1)
            denoised = F.linear(probs, self.model.get_input_embeddings().weight.t())
            # to d
            d = (z - denoised) / sigma[:, None]
            dt = next_sigma - sigma
            # euler step
            euler_step = z + dt[:, None] * d
            z = euler_step
        min_sigma = self.sigmas[-1].item()
        sigmas = torch.full((x.shape[0],), min_sigma, device=x.device)
        logits = self.denoise(x, z, sigmas)
        return logits
