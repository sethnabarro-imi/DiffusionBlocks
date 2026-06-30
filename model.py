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


class RegressionExplainedVariance(torchmetrics.Metric):
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("sse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("target_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "target_sq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.float().view(targets.shape)
        targets = targets.float()
        residual = preds - targets
        self.sse += residual.square().sum()
        self.target_sum += targets.sum()
        self.target_sq_sum += targets.square().sum()
        self.count += targets.numel()

    def compute(self) -> torch.Tensor:
        variance_sum = self.target_sq_sum - self.target_sum.square() / self.count.clamp_min(1)
        return 1.0 - self.sse / variance_sum.clamp_min(torch.finfo(self.sse.dtype).tiny)


class DBlockRegressionPredictionMetric(torchmetrics.Metric):
    higher_is_better = None
    full_state_update = False

    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.add_state("count", default=torch.zeros(num_steps), dist_reduce_fx="sum")
        self.add_state("mse_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum")
        self.add_state("mae_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum")
        self.add_state(
            "final_mse_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum"
        )
        self.add_state(
            "changed_from_prev_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "block_index_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum"
        )
        self.add_state("sigma_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum")
        self.add_state(
            "metadata_count", default=torch.zeros(num_steps), dist_reduce_fx="sum"
        )
        self.add_state(
            "denoise_step_index_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "layer_index_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum"
        )
        self.add_state(
            "layer_position_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum"
        )
        self.add_state(
            "layers_in_block_sum", default=torch.zeros(num_steps), dist_reduce_fx="sum"
        )

    def update(
        self,
        step_predictions: torch.Tensor,
        targets: torch.Tensor,
        block_indices: torch.Tensor,
        sigmas: torch.Tensor,
        denoise_step_indices: torch.Tensor | None = None,
        layer_indices: torch.Tensor | None = None,
        layer_positions: torch.Tensor | None = None,
        layers_in_block: torch.Tensor | None = None,
    ) -> None:
        if step_predictions.ndim != 3:
            raise ValueError(
                "step_predictions must have shape [num_steps, batch_size, target_dim]"
            )
        if step_predictions.shape[0] != self.num_steps:
            raise ValueError(
                f"Expected {self.num_steps} traced steps, got {step_predictions.shape[0]}"
            )
        targets = targets.float()
        predictions = step_predictions.float()
        if targets.ndim == 1:
            targets = targets[:, None]
        if predictions.shape[1:] != targets.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape[1:]} does not match target shape {targets.shape}"
            )

        squared_error = (predictions - targets[None, :, :]).square().mean(dim=-1)
        absolute_error = (predictions - targets[None, :, :]).abs().mean(dim=-1)
        final_squared_error = (predictions - predictions[-1][None, :, :]).square().mean(
            dim=-1
        )
        changed_from_prev = torch.zeros_like(squared_error)
        if self.num_steps > 1:
            changed_from_prev[1:] = (
                predictions[1:] - predictions[:-1]
            ).square().mean(dim=-1)

        batch_size = targets.shape[0]
        self.count += torch.full_like(self.count, batch_size)
        self.mse_sum += squared_error.sum(dim=1)
        self.mae_sum += absolute_error.sum(dim=1)
        self.final_mse_sum += final_squared_error.sum(dim=1)
        self.changed_from_prev_sum += changed_from_prev.sum(dim=1)

        block_indices = block_indices.to(self.block_index_sum.device).float()
        sigmas = sigmas.to(self.sigma_sum.device).float()
        if denoise_step_indices is None:
            denoise_step_indices = torch.arange(
                self.num_steps,
                device=self.denoise_step_index_sum.device,
                dtype=torch.float32,
            )
        else:
            denoise_step_indices = denoise_step_indices.to(
                self.denoise_step_index_sum.device
            ).float()
        if layer_indices is None:
            layer_indices = torch.full_like(self.layer_index_sum, -1.0)
        else:
            layer_indices = layer_indices.to(self.layer_index_sum.device).float()
        if layer_positions is None:
            layer_positions = torch.full_like(self.layer_position_sum, -1.0)
        else:
            layer_positions = layer_positions.to(
                self.layer_position_sum.device
            ).float()
        if layers_in_block is None:
            layers_in_block = torch.zeros_like(self.layers_in_block_sum)
        else:
            layers_in_block = layers_in_block.to(
                self.layers_in_block_sum.device
            ).float()

        self.block_index_sum += block_indices
        self.sigma_sum += sigmas
        self.denoise_step_index_sum += denoise_step_indices
        self.layer_index_sum += layer_indices
        self.layer_position_sum += layer_positions
        self.layers_in_block_sum += layers_in_block
        self.metadata_count += torch.ones_like(self.metadata_count)

    def compute(self) -> dict[str, torch.Tensor]:
        safe_count = self.count.clamp_min(1)
        safe_metadata_count = self.metadata_count.clamp_min(1)
        mse = self.mse_sum / safe_count
        return {
            "count": self.count,
            "denoise_step_index": self.denoise_step_index_sum / safe_metadata_count,
            "block_index": self.block_index_sum / safe_metadata_count,
            "layer_index": self.layer_index_sum / safe_metadata_count,
            "layer_position": self.layer_position_sum / safe_metadata_count,
            "layers_in_block": self.layers_in_block_sum / safe_metadata_count,
            "sigma": self.sigma_sum / safe_metadata_count,
            "mse": mse,
            "rmse": mse.clamp_min(0).sqrt(),
            "mae": self.mae_sum / safe_count,
            "final_mse": self.final_mse_sum / safe_count,
            "changed_from_previous_mse": self.changed_from_prev_sum / safe_count,
        }


class DBlockIntermediatePredictionMetric(torchmetrics.Metric):
    higher_is_better = None
    full_state_update = False

    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.add_state(
            "count",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "correct_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "confidence_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "true_prob_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "log_likelihood_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "entropy_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "final_agreement_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "changed_from_prev_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "incorrect_to_correct_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "correct_to_incorrect_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "block_index_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sigma_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "metadata_count",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "denoise_step_index_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "layer_index_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "layer_position_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "layers_in_block_sum",
            default=torch.zeros(num_steps),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        step_logits: torch.Tensor,
        labels: torch.Tensor,
        block_indices: torch.Tensor,
        sigmas: torch.Tensor,
        denoise_step_indices: torch.Tensor | None = None,
        layer_indices: torch.Tensor | None = None,
        layer_positions: torch.Tensor | None = None,
        layers_in_block: torch.Tensor | None = None,
    ) -> None:
        if step_logits.ndim != 3:
            raise ValueError(
                "step_logits must have shape [num_steps, batch_size, num_labels]"
            )
        if step_logits.shape[0] != self.num_steps:
            raise ValueError(
                f"Expected {self.num_steps} traced steps, got {step_logits.shape[0]}"
            )

        labels = labels.view(-1)
        probs = F.softmax(step_logits.float(), dim=-1)
        confidences, predictions = probs.max(dim=-1)
        correct = predictions.eq(labels[None, :])
        true_probs = probs.gather(
            dim=-1,
            index=labels[None, :, None].expand(self.num_steps, -1, 1),
        ).squeeze(-1)
        log_probs = F.log_softmax(step_logits.float(), dim=-1)
        log_likelihoods = log_probs.gather(
            dim=-1,
            index=labels[None, :, None].expand(self.num_steps, -1, 1),
        ).squeeze(-1)
        entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(
            dim=-1
        )

        batch_size = labels.numel()
        self.count += torch.full_like(self.count, batch_size)
        self.correct_sum += correct.float().sum(dim=1)
        self.confidence_sum += confidences.sum(dim=1)
        self.true_prob_sum += true_probs.sum(dim=1)
        self.log_likelihood_sum += log_likelihoods.sum(dim=1)
        self.entropy_sum += entropy.sum(dim=1)

        final_predictions = predictions[-1]
        self.final_agreement_sum += (
            predictions.eq(final_predictions[None, :]).float().sum(dim=1)
        )

        changed_from_prev = torch.zeros_like(correct, dtype=torch.float32)
        incorrect_to_correct = torch.zeros_like(correct, dtype=torch.float32)
        correct_to_incorrect = torch.zeros_like(correct, dtype=torch.float32)
        if self.num_steps > 1:
            changed_from_prev[1:] = predictions[1:].ne(predictions[:-1]).float()
            incorrect_to_correct[1:] = (correct[1:] & ~correct[:-1]).float()
            correct_to_incorrect[1:] = (~correct[1:] & correct[:-1]).float()
        self.changed_from_prev_sum += changed_from_prev.sum(dim=1)
        self.incorrect_to_correct_sum += incorrect_to_correct.sum(dim=1)
        self.correct_to_incorrect_sum += correct_to_incorrect.sum(dim=1)

        block_indices = block_indices.to(self.block_index_sum.device).float()
        sigmas = sigmas.to(self.sigma_sum.device).float()
        if denoise_step_indices is None:
            denoise_step_indices = torch.arange(
                self.num_steps,
                device=self.denoise_step_index_sum.device,
                dtype=torch.float32,
            )
        else:
            denoise_step_indices = denoise_step_indices.to(
                self.denoise_step_index_sum.device
            ).float()
        if layer_indices is None:
            layer_indices = torch.full_like(self.layer_index_sum, -1.0)
        else:
            layer_indices = layer_indices.to(self.layer_index_sum.device).float()
        if layer_positions is None:
            layer_positions = torch.full_like(self.layer_position_sum, -1.0)
        else:
            layer_positions = layer_positions.to(
                self.layer_position_sum.device
            ).float()
        if layers_in_block is None:
            layers_in_block = torch.zeros_like(self.layers_in_block_sum)
        else:
            layers_in_block = layers_in_block.to(
                self.layers_in_block_sum.device
            ).float()
        self.block_index_sum += block_indices
        self.sigma_sum += sigmas
        self.denoise_step_index_sum += denoise_step_indices
        self.layer_index_sum += layer_indices
        self.layer_position_sum += layer_positions
        self.layers_in_block_sum += layers_in_block
        self.metadata_count += torch.ones_like(self.metadata_count)

    def compute(self) -> dict[str, torch.Tensor]:
        safe_count = self.count.clamp_min(1)
        safe_metadata_count = self.metadata_count.clamp_min(1)
        return {
            "count": self.count,
            "denoise_step_index": self.denoise_step_index_sum / safe_metadata_count,
            "block_index": self.block_index_sum / safe_metadata_count,
            "layer_index": self.layer_index_sum / safe_metadata_count,
            "layer_position": self.layer_position_sum / safe_metadata_count,
            "layers_in_block": self.layers_in_block_sum / safe_metadata_count,
            "sigma": self.sigma_sum / safe_metadata_count,
            "accuracy": self.correct_sum / safe_count,
            "confidence": self.confidence_sum / safe_count,
            "true_probability": self.true_prob_sum / safe_count,
            "log_likelihood": self.log_likelihood_sum / safe_count,
            "entropy": self.entropy_sum / safe_count,
            "final_agreement": self.final_agreement_sum / safe_count,
            "changed_from_previous": self.changed_from_prev_sum / safe_count,
            "incorrect_to_correct": self.incorrect_to_correct_sum / safe_count,
            "correct_to_incorrect": self.correct_to_incorrect_sum / safe_count,
        }


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
        self.task_type = getattr(args, "task_type", "classification")
        self.classification_loss_type = getattr(
            args, "classification_loss_type", "cross_entropy"
        )
        self.label_smoothing = getattr(args, "label_smoothing", 0.0)
        self.multiclass_hinge_margin = getattr(
            args, "multiclass_hinge_margin", 1.0
        )
        self.classifier_head_type = getattr(args, "classifier_head_type", "linear")
        self.cosine_classifier_scale = getattr(args, "cosine_classifier_scale", 16.0)
        if self.label_smoothing < 0.0 or self.label_smoothing > 1.0:
            raise ValueError("--label_smoothing must be between 0 and 1")
        if (
            self.label_smoothing > 0.0
            and self.classification_loss_type != "cross_entropy"
        ):
            raise ValueError(
                "--label_smoothing requires --classification_loss_type cross_entropy"
            )
        if self.multiclass_hinge_margin <= 0.0:
            raise ValueError("--multiclass_hinge_margin must be positive")
        if self.cosine_classifier_scale <= 0.0:
            raise ValueError("--cosine_classifier_scale must be positive")
        if (
            self.classifier_head_type == "cosine"
            and self.task_type != "classification"
        ):
            raise ValueError("--classifier_head_type cosine requires classification")
        self.one_hot_mse_top_k = getattr(args, "one_hot_mse_top_k", None)
        if self.one_hot_mse_top_k is not None:
            if self.one_hot_mse_top_k < 0:
                raise ValueError("--one_hot_mse_top_k must be non-negative")
            if self.one_hot_mse_top_k > self.num_labels:
                raise ValueError("--one_hot_mse_top_k cannot exceed num_labels")
        self.valid_metrics = self.build_eval_metrics("val/")
        self.train_eval_metrics = self.build_eval_metrics("train_eval/")
        self.test_metrics = self.build_eval_metrics("test/")
        self.eval_split = "test"
        self.eval_results_by_split = {}
        self.save_hyperparameters(args)

    def architecture_kwargs(self) -> dict:
        return {
            "num_hidden_layers": self.args.num_hidden_layers,
            "attention_probs_dropout_prob": self.args.attention_probs_dropout_prob,
            "hidden_dropout_prob": self.args.hidden_dropout_prob,
            "classifier_head_type": self.classifier_head_type,
            "cosine_classifier_scale": self.cosine_classifier_scale,
        }

    def build_eval_metrics(self, prefix: str):
        if self.task_type == "regression":
            return torchmetrics.MetricCollection(
                {
                    "mse": torchmetrics.MeanSquaredError(),
                    "mae": torchmetrics.MeanAbsoluteError(),
                    "explained_variance": RegressionExplainedVariance(),
                },
                prefix=prefix,
            )
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
        self.model = load_vit(
            image_size=self.image_size,
            num_labels=self.num_labels,
            **self.architecture_kwargs(),
        )
        print(self.model)
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
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

    def per_example_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        loss_type: str | None = None,
        label_smoothing: float | None = None,
    ) -> torch.Tensor:
        loss_type = loss_type or self.classification_loss_type
        logits = logits.view(-1, self.num_labels)
        labels = labels.view(-1)
        if loss_type == "cross_entropy":
            smoothing = (
                self.label_smoothing
                if label_smoothing is None
                else label_smoothing
            )
            return F.cross_entropy(
                logits,
                labels,
                reduction="none",
                label_smoothing=smoothing,
            )
        if loss_type == "one_hot_mse":
            targets = F.one_hot(labels, num_classes=self.num_labels).to(logits)
            per_class_loss = F.mse_loss(logits, targets, reduction="none")
            if self.one_hot_mse_top_k is None:
                return per_class_loss.mean(dim=-1)

            mask = torch.zeros_like(per_class_loss, dtype=torch.bool)
            if self.one_hot_mse_top_k > 0:
                top_indices = logits.detach().topk(
                    self.one_hot_mse_top_k,
                    dim=-1,
                ).indices
                mask.scatter_(dim=-1, index=top_indices, value=True)
            mask.scatter_(dim=-1, index=labels[:, None], value=True)
            selected_loss = per_class_loss.masked_fill(~mask, 0.0).sum(dim=-1)
            selected_count = mask.sum(dim=-1).clamp_min(1)
            return selected_loss / selected_count
        if loss_type == "brier_score":
            probs = F.softmax(logits.float(), dim=-1)
            targets = F.one_hot(labels, num_classes=self.num_labels).to(probs)
            return (probs - targets).square().sum(dim=-1)
        if loss_type in ["multiclass_hinge", "squared_multiclass_hinge"]:
            true_logits = logits.gather(dim=-1, index=labels[:, None]).squeeze(-1)
            competitor_logits = logits.masked_fill(
                F.one_hot(labels, num_classes=self.num_labels).bool(),
                -torch.inf,
            ).max(dim=-1).values
            margins = (
                self.multiclass_hinge_margin
                + competitor_logits
                - true_logits
            ).clamp_min(0.0)
            if loss_type == "squared_multiclass_hinge":
                return margins.square()
            return margins
        raise ValueError(f"Unsupported classification loss type: {loss_type}")

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        return self.per_example_classification_loss(logits, labels).mean()

    def per_example_regression_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        predictions = predictions.view(-1, self.num_labels)
        targets = targets.float().view(-1, self.num_labels)
        return F.mse_loss(predictions, targets, reduction="none").mean(dim=-1)

    def regression_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        return self.per_example_regression_loss(predictions, targets).mean()

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        predictions = self(pixel_values=pixel_values, **kwargs)
        if return_metrics:
            predictions = predictions.view(-1, self.num_labels)
            if self.task_type == "regression":
                labels = labels.float().view(-1, self.num_labels)
            else:
                labels = labels.view(-1)
            if step == "val":
                return self.valid_metrics(predictions, labels)
            elif step == "train_eval":
                return self.train_eval_metrics(predictions, labels)
            elif step == "test":
                return self.test_metrics(predictions, labels)
            else:
                raise NotImplementedError(f"Step {step} is not supported")

        if self.task_type == "regression":
            loss = self.regression_loss(predictions, labels)
            return loss, {f"{step}/loss": loss, f"{step}/mse": loss}

        loss = self.classification_loss(predictions, labels)
        loss_dict = {f"{step}/loss": loss}
        if self.classification_loss_type != "cross_entropy":
            ce_loss = self.per_example_classification_loss(
                predictions, labels, loss_type="cross_entropy"
            ).mean()
            loss_dict[f"{step}/ce_loss"] = ce_loss
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
        if calibration_metric is not None:
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
        return None

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
        self.sequential_denoising_training = getattr(
            self.args, "sequential_denoising_training", False
        )
        self.hybrid_block0_independent_training = getattr(
            self.args, "hybrid_block0_independent_training", False
        )
        self.dblock_training_objective = getattr(
            self.args, "dblock_training_objective", "classification"
        )
        if self.dblock_training_objective not in [
            "classification",
            "residual_next_latent",
        ]:
            raise ValueError(
                f"Unsupported DBlock training objective: {self.dblock_training_objective}"
            )
        if (
            self.dblock_training_objective == "residual_next_latent"
            and self.task_type != "classification"
        ):
            raise ValueError(
                "--dblock_training_objective residual_next_latent currently "
                "requires classification targets"
            )
        self.trace_block_layers = getattr(self.args, "trace_block_layers", False)
        self.trace_intermediate_predictions = (
            getattr(self.args, "trace_intermediate_predictions", False)
            or self.trace_block_layers
        )
        self.trace_oracle_noise_predictions = getattr(
            self.args, "trace_oracle_noise_predictions", False
        )
        self.trace_prediction_examples = getattr(
            self.args, "trace_prediction_examples", 16
        )
        self.epsilon_seed = self.args.epsilon_seed
        if self.epsilon_seed is not None and self.epsilon_seed < 0:
            raise ValueError("--epsilon_seed must be non-negative")
        if self.trace_prediction_examples < 0:
            raise ValueError("--trace_prediction_examples must be non-negative")
        self.block_sigmas = get_block_sigmas(num_layers=self.args.num_blocks)
        self.layer_assignment = None
        intermediate_metric_cls = (
            DBlockRegressionPredictionMetric
            if self.task_type == "regression"
            else DBlockIntermediatePredictionMetric
        )
        self.intermediate_prediction_metrics = torch.nn.ModuleDict(
            {
                split: intermediate_metric_cls(self.num_inference_steps)
                for split in ["val", "train_eval", "test"]
            }
        )
        self.oracle_noise_prediction_metrics = torch.nn.ModuleDict(
            {
                split: intermediate_metric_cls(self.num_inference_steps)
                for split in ["val", "train_eval", "test"]
            }
        )
        self.layer_prediction_metrics = None
        self.oracle_noise_layer_prediction_metrics = None
        self.intermediate_prediction_examples_by_split = {
            "val": [],
            "train_eval": [],
            "test": [],
        }
        self.oracle_noise_prediction_examples_by_split = {
            "val": [],
            "train_eval": [],
            "test": [],
        }
        self.layer_prediction_examples_by_split = {
            "val": [],
            "train_eval": [],
            "test": [],
        }
        self.oracle_noise_layer_prediction_examples_by_split = {
            "val": [],
            "train_eval": [],
            "test": [],
        }
        self.register_buffer(
            "sigmas",
            get_discrete_sigmas(num_steps=self.num_inference_steps, dblock=True).to(
                self.device
            ),
            persistent=False,
        )
        self.save_hyperparameters(
            {
                "gamma": self.gamma,
                "num_inference_steps": self.num_inference_steps,
                "num_prediction_samples": self.num_prediction_samples,
                "prediction_average": self.prediction_average,
                "classification_loss_type": self.classification_loss_type,
                "label_smoothing": self.label_smoothing,
                "multiclass_hinge_margin": self.multiclass_hinge_margin,
                "classifier_head_type": self.classifier_head_type,
                "cosine_classifier_scale": self.cosine_classifier_scale,
                "one_hot_mse_top_k": self.one_hot_mse_top_k,
                "dblock_training_objective": self.dblock_training_objective,
                "sequential_denoising_training": (
                    self.sequential_denoising_training
                ),
                "hybrid_block0_independent_training": (
                    self.hybrid_block0_independent_training
                ),
                "cfg_scale": self.cfg_scale,
                "class_dropout_prob": self.class_dropout_prob,
                "trace_intermediate_predictions": self.trace_intermediate_predictions,
                "trace_oracle_noise_predictions": (
                    self.trace_oracle_noise_predictions
                ),
                "trace_block_layers": self.trace_block_layers,
                "trace_prediction_examples": self.trace_prediction_examples,
                "epsilon_seed": self.epsilon_seed,
            },
        )

    def configure_model(self):
        architecture_kwargs = self.architecture_kwargs()
        if self.dblock_training_objective == "residual_next_latent":
            architecture_kwargs["latent_prediction_head"] = True
        self.model = load_vit(
            image_size=self.image_size,
            num_labels=self.num_labels,
            is_dblock=True,
            **architecture_kwargs,
        )
        if self.task_type == "regression":
            self.regression_target_encoder = torch.nn.Linear(
                self.num_labels,
                self.model.config.hidden_size,
            )
        self.build_layer_prediction_metrics()
        print(self.model)

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            return
        for key in list(state_dict):
            if key == "sigmas" or key.startswith(
                (
                    "intermediate_prediction_metrics.",
                    "oracle_noise_prediction_metrics.",
                    "layer_prediction_metrics.",
                    "oracle_noise_layer_prediction_metrics.",
                )
            ):
                state_dict.pop(key)

    def build_layer_prediction_metrics(self):
        layer_points = self.num_inference_steps * self.layers_per_block()
        metric_cls = (
            DBlockRegressionPredictionMetric
            if self.task_type == "regression"
            else DBlockIntermediatePredictionMetric
        )
        self.layer_prediction_metrics = torch.nn.ModuleDict(
            {
                split: metric_cls(layer_points)
                for split in ["val", "train_eval", "test"]
            }
        )
        self.oracle_noise_layer_prediction_metrics = torch.nn.ModuleDict(
            {
                split: metric_cls(layer_points)
                for split in ["val", "train_eval", "test"]
            }
        )

    def layers_per_block(self) -> int:
        return self.model.config.num_hidden_layers // self.args.num_blocks

    def get_layer_assignment(self) -> list[list[int]]:
        if self.layer_assignment is None:
            split_size = self.layers_per_block()
            if split_size < 1:
                raise ValueError(
                    "--num_blocks must be no larger than the ViT hidden layer count"
                )
            self.layer_assignment = [
                list(range(i * split_size, (i + 1) * split_size))
                for i in range(self.args.num_blocks)
            ]
        return self.layer_assignment

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

    def get_target_latents(self, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return self.get_embeds(targets, is_input=True)
        targets = targets.float().view(-1, self.num_labels)
        return self.normalize_embeddings(self.regression_target_encoder(targets))

    def get_sigmas(self, n_samples: int, p_mean: float = -1.2, p_std: float = 1.2):
        block_idx = random.choices(range(self.args.num_blocks), k=1)[0]
        sigma_min_block = self.block_sigmas[block_idx]
        sigma_max_block = self.block_sigmas[block_idx + 1]
        return self.sample_sigmas_from_interval(
            n_samples,
            sigma_min_block,
            sigma_max_block,
            p_mean=p_mean,
            p_std=p_std,
        )

    def get_sigmas_for_target_block(
        self,
        n_samples: int,
        target_block_idx: int,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
        if target_block_idx < 0 or target_block_idx >= self.args.num_blocks:
            raise ValueError(
                f"target_block_idx must be in [0, {self.args.num_blocks - 1}]"
            )
        sigma_interval_idx = (self.args.num_blocks - 1) - target_block_idx
        sigma_min_block = self.block_sigmas[sigma_interval_idx]
        sigma_max_block = self.block_sigmas[sigma_interval_idx + 1]
        return self.sample_sigmas_from_interval(
            n_samples,
            sigma_min_block,
            sigma_max_block,
            p_mean=p_mean,
            p_std=p_std,
        )

    def sample_sigmas_from_interval(
        self,
        n_samples: int,
        sigma_min_block: float,
        sigma_max_block: float,
        p_mean: float = -1.2,
        p_std: float = 1.2,
    ):
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

    def apply_classifier_free_guidance(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.training and self.cfg_scale > 0.0:
            logits_uncond, logits_cond = logits.chunk(2, dim=-2)
            return logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
        return logits

    def uses_residual_next_latent_objective(self) -> bool:
        return self.dblock_training_objective == "residual_next_latent"

    def logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        label_embeddings = self.normalize_embeddings(
            self.model.get_input_embeddings().weight
        )
        latent = self.normalize_embeddings(latent)
        logits = F.linear(latent, label_embeddings)
        if self.classifier_head_type == "cosine":
            logits = logits * self.cosine_classifier_scale
        return logits

    def pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.vit.layernorm(hidden_states)
        if self.model.config.pooling_type == "cls":
            return hidden_states[:, 0, :]
        if self.model.config.pooling_type == "mean":
            return hidden_states[:, 1:, :].mean(dim=1)
        raise ValueError(f"Invalid pooling type: {self.model.config.pooling_type}")

    def denoise(self, x, zt, sigma, block_idx=None, return_layer_logits=False):
        if block_idx is None:
            block_idx = self.estimate_target_layer(sigma)
        zt_for_update = zt
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

        layer_indices = self.get_layer_assignment()[block_idx]
        outputs = self.model.forward_block(
            layer_indices=layer_indices,
            pixel_values=x,
            noisy_embeds=zt * c_in[:, None],
            timesteps=c_noise,
            output_hidden_states=return_layer_logits,
        )
        hidden_states = outputs.last_hidden_state
        conditioning = outputs.conditioning
        model_out = hidden_states * c_out[:, None] + zt * c_skip[:, None]
        latent_delta = None
        if self.uses_residual_next_latent_objective():
            latent_delta = self.model.forward_latent_delta(
                hidden_states.unsqueeze(1), conditioning
            )
            latent_delta = self.apply_classifier_free_guidance(latent_delta)
            model_out = zt_for_update + latent_delta
            logits = self.logits_from_latent(model_out)
        else:
            logits = self.model.forward_output_embeddings(
                model_out.unsqueeze(1), conditioning
            )
            logits = self.apply_classifier_free_guidance(logits)
        if self.task_type == "regression":
            model_out = self.apply_classifier_free_guidance(model_out)
        if not return_layer_logits:
            if self.task_type == "regression" or self.uses_residual_next_latent_objective():
                if latent_delta is not None:
                    return {
                        "logits": logits,
                        "latent": model_out,
                        "latent_delta": latent_delta,
                    }
                return {"logits": logits, "latent": model_out}
            return logits

        layer_logits = []
        layer_latents = []
        layer_latent_deltas = []
        for layer_hidden_states in outputs.hidden_states[1:]:
            layer_hidden_states = self.pool_hidden_states(layer_hidden_states)
            if self.uses_residual_next_latent_objective():
                layer_delta = self.model.forward_latent_delta(
                    layer_hidden_states.unsqueeze(1), conditioning
                )
                layer_delta = self.apply_classifier_free_guidance(layer_delta)
                layer_model_out = zt_for_update + layer_delta
                layer_logit = self.logits_from_latent(layer_model_out)
                layer_latent_deltas.append(layer_delta)
            else:
                layer_model_out = (
                    layer_hidden_states * c_out[:, None] + zt * c_skip[:, None]
                )
                layer_logit = self.model.forward_output_embeddings(
                    layer_model_out.unsqueeze(1), conditioning
                )
                layer_logit = self.apply_classifier_free_guidance(layer_logit)
            layer_logits.append(layer_logit)
            if self.task_type == "regression":
                layer_latents.append(self.apply_classifier_free_guidance(layer_model_out))
            elif self.uses_residual_next_latent_objective():
                layer_latents.append(layer_model_out)
        result = {
            "logits": logits,
            "layer_logits": torch.stack(layer_logits),
            "layer_indices": torch.tensor(
                layer_indices, device=logits.device, dtype=torch.long
            ),
        }
        if self.task_type == "regression":
            result["latent"] = model_out
            result["layer_latents"] = torch.stack(layer_latents)
        if self.uses_residual_next_latent_objective():
            result["latent"] = model_out
            result["latent_delta"] = latent_delta
            result["layer_latents"] = torch.stack(layer_latents)
            result["layer_latent_deltas"] = torch.stack(layer_latent_deltas)
        return result

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        if self.trace_intermediate_predictions:
            self.get_intermediate_prediction_metric(self.eval_split).reset()
            self.intermediate_prediction_examples_by_split[self.eval_split] = []
            if self.trace_block_layers:
                self.get_layer_prediction_metric(self.eval_split).reset()
                self.layer_prediction_examples_by_split[self.eval_split] = []
        if self.trace_oracle_noise_predictions:
            self.get_oracle_noise_prediction_metric(self.eval_split).reset()
            self.oracle_noise_prediction_examples_by_split[self.eval_split] = []
            if self.trace_block_layers:
                self.get_oracle_noise_layer_prediction_metric(
                    self.eval_split
                ).reset()
                self.oracle_noise_layer_prediction_examples_by_split[
                    self.eval_split
                ] = []

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        if self.trainer.is_global_zero:
            if self.trace_intermediate_predictions:
                self.eval_results_by_split[self.eval_split][
                    "intermediate_predictions"
                ] = self.intermediate_prediction_results(self.eval_split)
            if self.trace_oracle_noise_predictions:
                self.eval_results_by_split[self.eval_split][
                    "oracle_noise_predictions"
                ] = self.oracle_noise_prediction_results(self.eval_split)

    def get_intermediate_prediction_metric(self, split: str):
        if split not in self.intermediate_prediction_metrics:
            raise NotImplementedError(f"Step {split} is not supported")
        return self.intermediate_prediction_metrics[split]

    def get_layer_prediction_metric(self, split: str):
        if self.layer_prediction_metrics is None:
            self.build_layer_prediction_metrics()
        if split not in self.layer_prediction_metrics:
            raise NotImplementedError(f"Step {split} is not supported")
        return self.layer_prediction_metrics[split]

    def get_oracle_noise_prediction_metric(self, split: str):
        if split not in self.oracle_noise_prediction_metrics:
            raise NotImplementedError(f"Step {split} is not supported")
        return self.oracle_noise_prediction_metrics[split]

    def get_oracle_noise_layer_prediction_metric(self, split: str):
        if self.oracle_noise_layer_prediction_metrics is None:
            self.build_layer_prediction_metrics()
        if split not in self.oracle_noise_layer_prediction_metrics:
            raise NotImplementedError(f"Step {split} is not supported")
        return self.oracle_noise_layer_prediction_metrics[split]

    def intermediate_prediction_results(self, split: str) -> dict:
        metric = self.get_intermediate_prediction_metric(split)
        with metric.sync_context():
            stats = metric.compute()
        rows = self.intermediate_prediction_rows(stats)
        results = {
            "prediction_average": self.prediction_average,
            "num_prediction_samples": self.num_prediction_samples,
            "num_inference_steps": self.num_inference_steps,
            "rows": rows,
            "examples": self.intermediate_prediction_examples_by_split[split],
        }
        if self.trace_block_layers:
            layer_metric = self.get_layer_prediction_metric(split)
            with layer_metric.sync_context():
                layer_stats = layer_metric.compute()
            results["layer_rows"] = self.layer_prediction_rows(layer_stats)
            results["layer_examples"] = self.layer_prediction_examples_by_split[split]
        return results

    def oracle_noise_prediction_results(self, split: str) -> dict:
        metric = self.get_oracle_noise_prediction_metric(split)
        with metric.sync_context():
            stats = metric.compute()
        rows = self.intermediate_prediction_rows(stats)
        results = {
            "prediction_average": self.prediction_average,
            "num_prediction_samples": self.num_prediction_samples,
            "num_inference_steps": self.num_inference_steps,
            "input_type": "oracle_label_embedding_plus_noise",
            "rows": rows,
            "examples": self.oracle_noise_prediction_examples_by_split[split],
        }
        if self.trace_block_layers:
            layer_metric = self.get_oracle_noise_layer_prediction_metric(split)
            with layer_metric.sync_context():
                layer_stats = layer_metric.compute()
            results["layer_rows"] = self.layer_prediction_rows(layer_stats)
            results["layer_examples"] = (
                self.oracle_noise_layer_prediction_examples_by_split[split]
            )
        return results

    def intermediate_prediction_rows(
        self, stats: dict[str, torch.Tensor]
    ) -> list[dict]:
        cpu_stats = {
            key: value.detach().float().cpu().tolist()
            for key, value in stats.items()
        }
        rows = []
        for step_index in range(len(cpu_stats["count"])):
            block_index = round(cpu_stats["block_index"][step_index])
            row = {
                "step_index": step_index,
                "block_index": int(block_index),
                "sigma": cpu_stats["sigma"][step_index],
                "count": int(cpu_stats["count"][step_index]),
            }
            if self.task_type == "regression":
                row.update(
                    {
                        "mse": cpu_stats["mse"][step_index],
                        "rmse": cpu_stats["rmse"][step_index],
                        "mae": cpu_stats["mae"][step_index],
                        "final_mse": cpu_stats["final_mse"][step_index],
                        "changed_from_previous_mse": cpu_stats[
                            "changed_from_previous_mse"
                        ][step_index],
                    }
                )
            else:
                row.update(
                    {
                        "accuracy_rate": cpu_stats["accuracy"][step_index],
                        "confidence_rate": cpu_stats["confidence"][step_index],
                        "true_probability": cpu_stats["true_probability"][step_index],
                        "log_likelihood": cpu_stats["log_likelihood"][step_index],
                        "entropy": cpu_stats["entropy"][step_index],
                        "final_agreement_rate": cpu_stats["final_agreement"][
                            step_index
                        ],
                        "changed_from_previous_rate": cpu_stats[
                            "changed_from_previous"
                        ][step_index],
                        "incorrect_to_correct_rate": cpu_stats[
                            "incorrect_to_correct"
                        ][step_index],
                        "correct_to_incorrect_rate": cpu_stats[
                            "correct_to_incorrect"
                        ][step_index],
                    }
                )
            rows.append(row)
        return rows

    def layer_prediction_rows(self, stats: dict[str, torch.Tensor]) -> list[dict]:
        cpu_stats = {
            key: value.detach().float().cpu().tolist()
            for key, value in stats.items()
        }
        rows = []
        for event_index in range(len(cpu_stats["count"])):
            denoise_step_index = round(cpu_stats["denoise_step_index"][event_index])
            block_index = round(cpu_stats["block_index"][event_index])
            layer_index = round(cpu_stats["layer_index"][event_index])
            layer_position = round(cpu_stats["layer_position"][event_index])
            layers_in_block = round(cpu_stats["layers_in_block"][event_index])
            row = {
                "event_index": event_index,
                "denoise_step_index": int(denoise_step_index),
                "block_index": int(block_index),
                "layer_index": int(layer_index),
                "layer_position": int(layer_position),
                "layers_in_block": int(layers_in_block),
                "sigma": cpu_stats["sigma"][event_index],
                "count": int(cpu_stats["count"][event_index]),
            }
            if self.task_type == "regression":
                row.update(
                    {
                        "mse": cpu_stats["mse"][event_index],
                        "rmse": cpu_stats["rmse"][event_index],
                        "mae": cpu_stats["mae"][event_index],
                        "final_mse": cpu_stats["final_mse"][event_index],
                        "changed_from_previous_mse": cpu_stats[
                            "changed_from_previous_mse"
                        ][event_index],
                    }
                )
            else:
                row.update(
                    {
                        "accuracy_rate": cpu_stats["accuracy"][event_index],
                        "confidence_rate": cpu_stats["confidence"][event_index],
                        "true_probability": cpu_stats["true_probability"][
                            event_index
                        ],
                        "log_likelihood": cpu_stats["log_likelihood"][event_index],
                        "entropy": cpu_stats["entropy"][event_index],
                        "final_agreement_rate": cpu_stats["final_agreement"][
                            event_index
                        ],
                        "changed_from_previous_rate": cpu_stats[
                            "changed_from_previous"
                        ][event_index],
                        "incorrect_to_correct_rate": cpu_stats[
                            "incorrect_to_correct"
                        ][event_index],
                        "correct_to_incorrect_rate": cpu_stats[
                            "correct_to_incorrect"
                        ][event_index],
                    }
                )
            rows.append(row)
        return rows

    def record_intermediate_predictions(
        self,
        trace: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        if self.task_type == "regression":
            labels = labels.float().view(-1, self.num_labels)
        else:
            labels = labels.view(-1)
        self.get_intermediate_prediction_metric(self.eval_split).update(
            trace["logits"],
            labels,
            trace["block_indices"],
            trace["sigmas"],
        )
        if self.trace_block_layers and "layer_logits" in trace:
            self.get_layer_prediction_metric(self.eval_split).update(
                trace["layer_logits"],
                labels,
                trace["layer_block_indices"],
                trace["layer_sigmas"],
                denoise_step_indices=trace["layer_denoise_step_indices"],
                layer_indices=trace["layer_indices"],
                layer_positions=trace["layer_positions"],
                layers_in_block=trace["layers_in_block"],
            )
        if self.trace_prediction_examples == 0:
            return
        if not self.trainer.is_global_zero:
            return

        if (
            self.trace_block_layers
            and self.task_type == "classification"
            and "layer_logits" in trace
        ):
            self.record_layer_prediction_examples(trace, labels)

        examples = self.intermediate_prediction_examples_by_split[self.eval_split]
        remaining = self.trace_prediction_examples - len(examples)
        if remaining <= 0:
            return

        if self.task_type == "regression":
            predictions = trace["logits"].float()
            targets = labels.float()
            block_indices = trace["block_indices"].detach().cpu().tolist()
            sigmas = trace["sigmas"].detach().float().cpu().tolist()
            squared_error = (predictions - targets[None, :, :]).square().mean(dim=-1)
            for batch_index in range(min(remaining, labels.shape[0])):
                steps = []
                for step_index in range(predictions.shape[0]):
                    steps.append(
                        {
                            "step_index": step_index,
                            "block_index": int(block_indices[step_index]),
                            "sigma": sigmas[step_index],
                            "prediction": predictions[
                                step_index, batch_index
                            ].detach().cpu().tolist(),
                            "mse": float(
                                squared_error[
                                    step_index, batch_index
                                ].detach().cpu()
                            ),
                        }
                    )
                examples.append(
                    {
                        "example_index": len(examples),
                        "target": targets[batch_index].detach().cpu().tolist(),
                        "steps": steps,
                    }
                )
            return

        probs = F.softmax(trace["logits"].float(), dim=-1)
        confidences, predictions = probs.max(dim=-1)
        true_probs = probs.gather(
            dim=-1,
            index=labels[None, :, None].expand(probs.shape[0], -1, 1),
        ).squeeze(-1)
        log_probs = F.log_softmax(trace["logits"].float(), dim=-1)
        log_likelihoods = log_probs.gather(
            dim=-1,
            index=labels[None, :, None].expand(log_probs.shape[0], -1, 1),
        ).squeeze(-1)
        block_indices = trace["block_indices"].detach().cpu().tolist()
        sigmas = trace["sigmas"].detach().float().cpu().tolist()
        for batch_index in range(min(remaining, labels.numel())):
            steps = []
            for step_index in range(probs.shape[0]):
                prediction = int(predictions[step_index, batch_index].detach().cpu())
                steps.append(
                    {
                        "step_index": step_index,
                        "block_index": int(block_indices[step_index]),
                        "sigma": sigmas[step_index],
                        "prediction": prediction,
                        "is_correct": prediction == int(labels[batch_index].cpu()),
                        "confidence": float(
                            confidences[step_index, batch_index].detach().cpu()
                        ),
                        "true_probability": float(
                            true_probs[step_index, batch_index].detach().cpu()
                        ),
                        "log_likelihood": float(
                            log_likelihoods[step_index, batch_index].detach().cpu()
                        ),
                    }
                )
            examples.append(
                {
                    "example_index": len(examples),
                    "label": int(labels[batch_index].detach().cpu()),
                    "steps": steps,
                }
            )

    def record_oracle_noise_predictions(
        self,
        trace: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        if self.task_type == "regression":
            labels = labels.float().view(-1, self.num_labels)
        else:
            labels = labels.view(-1)
        self.get_oracle_noise_prediction_metric(self.eval_split).update(
            trace["logits"],
            labels,
            trace["block_indices"],
            trace["sigmas"],
        )
        if self.trace_block_layers and "layer_logits" in trace:
            self.get_oracle_noise_layer_prediction_metric(self.eval_split).update(
                trace["layer_logits"],
                labels,
                trace["layer_block_indices"],
                trace["layer_sigmas"],
                denoise_step_indices=trace["layer_denoise_step_indices"],
                layer_indices=trace["layer_indices"],
                layer_positions=trace["layer_positions"],
                layers_in_block=trace["layers_in_block"],
            )

    def record_layer_prediction_examples(
        self,
        trace: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        examples = self.layer_prediction_examples_by_split[self.eval_split]
        remaining = self.trace_prediction_examples - len(examples)
        if remaining <= 0:
            return

        probs = F.softmax(trace["layer_logits"].float(), dim=-1)
        confidences, predictions = probs.max(dim=-1)
        true_probs = probs.gather(
            dim=-1,
            index=labels[None, :, None].expand(probs.shape[0], -1, 1),
        ).squeeze(-1)
        log_probs = F.log_softmax(trace["layer_logits"].float(), dim=-1)
        log_likelihoods = log_probs.gather(
            dim=-1,
            index=labels[None, :, None].expand(log_probs.shape[0], -1, 1),
        ).squeeze(-1)
        denoise_step_indices = (
            trace["layer_denoise_step_indices"].detach().cpu().tolist()
        )
        block_indices = trace["layer_block_indices"].detach().cpu().tolist()
        layer_indices = trace["layer_indices"].detach().cpu().tolist()
        layer_positions = trace["layer_positions"].detach().cpu().tolist()
        layers_in_block = trace["layers_in_block"].detach().cpu().tolist()
        sigmas = trace["layer_sigmas"].detach().float().cpu().tolist()
        for batch_index in range(min(remaining, labels.numel())):
            events = []
            for event_index in range(probs.shape[0]):
                prediction = int(predictions[event_index, batch_index].detach().cpu())
                events.append(
                    {
                        "event_index": event_index,
                        "denoise_step_index": int(
                            denoise_step_indices[event_index]
                        ),
                        "block_index": int(block_indices[event_index]),
                        "layer_index": int(layer_indices[event_index]),
                        "layer_position": int(layer_positions[event_index]),
                        "layers_in_block": int(layers_in_block[event_index]),
                        "sigma": sigmas[event_index],
                        "prediction": prediction,
                        "is_correct": prediction == int(labels[batch_index].cpu()),
                        "confidence": float(
                            confidences[event_index, batch_index].detach().cpu()
                        ),
                        "true_probability": float(
                            true_probs[event_index, batch_index].detach().cpu()
                        ),
                        "log_likelihood": float(
                            log_likelihoods[event_index, batch_index].detach().cpu()
                        ),
                    }
                )
            examples.append(
                {
                    "example_index": len(examples),
                    "label": int(labels[batch_index].detach().cpu()),
                    "events": events,
                }
            )

    def oracle_noise_sample(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        return_layer_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.task_type == "classification":
            labels = labels.view(-1)
        else:
            labels = labels.float().view(-1, self.num_labels)
        z = self.get_target_latents(labels)
        oracle_logits = []
        block_indices = []
        sigmas_trace = []
        layer_logits = []
        layer_denoise_step_indices = []
        layer_block_indices = []
        layer_indices_trace = []
        layer_positions = []
        layers_in_block_trace = []
        layer_sigmas = []
        s_in = x.new_ones([x.shape[0]])
        for step_index, sigma_value in enumerate(self.sigmas):
            sigma = sigma_value * s_in
            block_idx = self.estimate_target_layer(sigma)
            zt = z + sigma[:, None] * self.sample_epsilon_like(z)
            denoise_output = self.denoise(
                x,
                zt,
                sigma,
                block_idx=block_idx,
                return_layer_logits=return_layer_logits,
            )
            if isinstance(denoise_output, dict):
                logits = denoise_output["logits"]
                if "layer_logits" in denoise_output:
                    self.append_layer_trace(
                        denoise_output,
                        denoise_step_index=step_index,
                        block_idx=block_idx,
                        sigma=sigma_value,
                        layer_logits=layer_logits,
                        layer_denoise_step_indices=layer_denoise_step_indices,
                        layer_block_indices=layer_block_indices,
                        layer_indices_trace=layer_indices_trace,
                        layer_positions=layer_positions,
                        layers_in_block_trace=layers_in_block_trace,
                        layer_sigmas=layer_sigmas,
                    )
            else:
                logits = denoise_output
            oracle_logits.append(logits)
            block_indices.append(block_idx)
            sigmas_trace.append(sigma_value)

        sample = {
            "logits": torch.stack(oracle_logits),
            "block_indices": torch.tensor(
                block_indices, device=x.device, dtype=torch.long
            ),
            "sigmas": torch.stack(sigmas_trace).to(x.device),
        }
        if layer_logits:
            sample.update(
                {
                    "layer_logits": torch.cat(layer_logits),
                    "layer_denoise_step_indices": torch.tensor(
                        layer_denoise_step_indices,
                        device=x.device,
                        dtype=torch.long,
                    ),
                    "layer_block_indices": torch.tensor(
                        layer_block_indices,
                        device=x.device,
                        dtype=torch.long,
                    ),
                    "layer_indices": torch.tensor(
                        layer_indices_trace,
                        device=x.device,
                        dtype=torch.long,
                    ),
                    "layer_positions": torch.tensor(
                        layer_positions,
                        device=x.device,
                        dtype=torch.long,
                    ),
                    "layers_in_block": torch.tensor(
                        layers_in_block_trace,
                        device=x.device,
                        dtype=torch.long,
                    ),
                    "layer_sigmas": torch.stack(layer_sigmas).to(x.device),
                }
            )
        return sample

    def oracle_noise_step(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        return_layer_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        trace_outputs = None
        layer_trace_outputs = None
        trace_metadata = None
        layer_trace_metadata = None
        for _ in range(self.num_prediction_samples):
            sample = self.oracle_noise_sample(
                x,
                labels,
                return_layer_logits=return_layer_logits,
            )
            trace_logits = sample["logits"]
            trace_metadata = {
                "block_indices": sample["block_indices"],
                "sigmas": sample["sigmas"],
            }
            if "layer_logits" in sample:
                layer_trace_logits = sample["layer_logits"]
                layer_trace_metadata = {
                    "layer_denoise_step_indices": sample[
                        "layer_denoise_step_indices"
                    ],
                    "layer_block_indices": sample["layer_block_indices"],
                    "layer_indices": sample["layer_indices"],
                    "layer_positions": sample["layer_positions"],
                    "layers_in_block": sample["layers_in_block"],
                    "layer_sigmas": sample["layer_sigmas"],
                }

            if self.prediction_average == "probability":
                sample_trace_output = F.softmax(trace_logits.float(), dim=-1)
                if "layer_logits" in sample:
                    sample_layer_trace_output = F.softmax(
                        layer_trace_logits.float(), dim=-1
                    )
            elif self.prediction_average == "logit":
                sample_trace_output = trace_logits.float()
                if "layer_logits" in sample:
                    sample_layer_trace_output = layer_trace_logits.float()
            else:
                raise ValueError(
                    f"Unsupported prediction_average: {self.prediction_average}"
                )

            trace_outputs = (
                sample_trace_output
                if trace_outputs is None
                else trace_outputs + sample_trace_output
            )
            if "layer_logits" in sample:
                layer_trace_outputs = (
                    sample_layer_trace_output
                    if layer_trace_outputs is None
                    else layer_trace_outputs + sample_layer_trace_output
                )

        trace_outputs = trace_outputs / self.num_prediction_samples
        if layer_trace_outputs is not None:
            layer_trace_outputs = layer_trace_outputs / self.num_prediction_samples
        if self.prediction_average == "probability":
            trace_outputs = trace_outputs.clamp_min(
                torch.finfo(trace_outputs.dtype).tiny
            ).log()
            if layer_trace_outputs is not None:
                layer_trace_outputs = layer_trace_outputs.clamp_min(
                    torch.finfo(layer_trace_outputs.dtype).tiny
                ).log()

        trace = {
            "logits": trace_outputs,
            "block_indices": trace_metadata["block_indices"],
            "sigmas": trace_metadata["sigmas"],
        }
        if layer_trace_outputs is not None:
            trace.update(
                {
                    "layer_logits": layer_trace_outputs,
                    **layer_trace_metadata,
                }
            )
        return trace

    def shared_step(self, batch, step="train", return_metrics=False, **kwargs):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        if return_metrics:
            should_trace = self.trace_intermediate_predictions and step in [
                "train_eval",
                "test",
            ]
            should_trace_oracle_noise = (
                self.trace_oracle_noise_predictions
                and step in [
                    "train_eval",
                    "test",
                ]
            )
            if should_trace:
                logits, trace = self.diffusion_step(
                    pixel_values, return_intermediates=True
                )
                self.record_intermediate_predictions(trace, labels)
            else:
                logits = self.diffusion_step(pixel_values)
            if should_trace_oracle_noise:
                oracle_noise_trace = self.oracle_noise_step(
                    pixel_values,
                    labels,
                    return_layer_logits=self.trace_block_layers,
                )
                self.record_oracle_noise_predictions(oracle_noise_trace, labels)
            logits = logits.view(-1, self.num_labels)
            if self.task_type == "regression":
                labels = labels.float().view(-1, self.num_labels)
            else:
                labels = labels.view(-1)
            if step == "val":
                return self.valid_metrics(logits, labels)
            elif step == "train_eval":
                return self.train_eval_metrics(logits, labels)
            elif step == "test":
                return self.test_metrics(logits, labels)
            else:
                raise NotImplementedError(f"Step {step} is not supported")

        if self.hybrid_block0_independent_training:
            return self.hybrid_block0_independent_training_step(
                pixel_values,
                labels,
                step=step,
            )

        if self.sequential_denoising_training:
            return self.sequential_denoising_training_step(
                pixel_values,
                labels,
                step=step,
            )

        z = self.get_target_latents(labels)
        sigmas = self.get_sigmas(z.shape[0])
        block_idx = self.estimate_target_layer(sigmas)
        sigmas = sigmas.to(z)
        zt = z + sigmas[:, None] * self.sample_epsilon_like(z)
        denoise_output = self.denoise(pixel_values, zt, sigmas, block_idx)
        logits = self.prediction_from_denoise_output(denoise_output)
        if self.task_type == "regression":
            per_example_loss = self.per_example_regression_loss(logits, labels)
            raw_loss = per_example_loss.mean()
            w = self.get_weights(sigmas)
            loss = (per_example_loss * w).mean()
            return loss, {
                f"{step}/loss": loss,
                f"{step}/loss_{block_idx}": loss,
                f"{step}/mse": raw_loss,
                f"{step}/mse_{block_idx}": raw_loss,
            }

        per_example_loss = self.per_example_classification_loss(logits, labels)
        ce_per_example_loss = self.per_example_classification_loss(
            logits, labels, loss_type="cross_entropy"
        )
        ce_loss = ce_per_example_loss.mean()
        w = self.get_weights(sigmas)
        if self.classification_loss_type == "cross_entropy":
            w = w[:, None]
        loss = (per_example_loss * w).mean()

        loss_dict = {
            f"{step}/loss": loss,
            f"{step}/loss_{block_idx}": loss,
            f"{step}/ce_loss": ce_loss,
            f"{step}/ce_loss_{block_idx}": ce_loss,
        }
        if self.classification_loss_type != "cross_entropy":
            raw_loss = per_example_loss.mean()
            loss_dict[f"{step}/{self.classification_loss_type}"] = raw_loss
            loss_dict[f"{step}/{self.classification_loss_type}_{block_idx}"] = (
                raw_loss
            )
        return loss, loss_dict

    def denoised_embedding_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.classification_loss_type == "one_hot_mse":
            weights = logits
        else:
            weights = F.softmax(logits, dim=1)
        return F.linear(weights, self.model.get_input_embeddings().weight.t())

    def prediction_from_denoise_output(
        self, denoise_output: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(denoise_output, dict):
            return denoise_output["logits"]
        return denoise_output

    def latent_from_denoise_output(
        self, denoise_output: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.task_type == "regression":
            if not isinstance(denoise_output, dict):
                raise ValueError("Regression denoise output must include a latent")
            return denoise_output["latent"]
        if self.uses_residual_next_latent_objective():
            if not isinstance(denoise_output, dict):
                raise ValueError("Residual denoise output must include a latent")
            return denoise_output["latent"]
        return self.denoised_embedding_from_logits(
            self.prediction_from_denoise_output(denoise_output)
        )

    def latent_delta_from_denoise_output(
        self, denoise_output: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if not isinstance(denoise_output, dict) or "latent_delta" not in denoise_output:
            raise ValueError("Residual denoise output must include latent_delta")
        return denoise_output["latent_delta"]

    def append_denoising_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sigmas: torch.Tensor,
        losses: list[torch.Tensor],
        ce_losses: list[torch.Tensor],
        loss_dict: dict[str, torch.Tensor],
        *,
        log_prefix: str,
        step_index: int,
        block_idx: int,
        broadcast_weights: bool = False,
    ) -> None:
        if self.task_type == "regression":
            per_example_loss = self.per_example_regression_loss(logits, labels)
            raw_loss = per_example_loss.mean()
            weights = self.get_weights(sigmas)
            weighted_loss = (per_example_loss * weights).mean()
            losses.append(weighted_loss)
            ce_losses.append(raw_loss)
            loss_dict[f"{log_prefix}/loss_step_{step_index}"] = weighted_loss
            loss_dict[f"{log_prefix}/mse_step_{step_index}"] = raw_loss
            loss_dict[f"{log_prefix}/loss_block_{block_idx}"] = weighted_loss
            loss_dict[f"{log_prefix}/mse_block_{block_idx}"] = raw_loss
            loss_dict[f"{log_prefix}/loss_{block_idx}"] = weighted_loss
            loss_dict[f"{log_prefix}/mse_{block_idx}"] = raw_loss
            return

        per_example_loss = self.per_example_classification_loss(logits, labels)
        ce_per_example_loss = self.per_example_classification_loss(
            logits, labels, loss_type="cross_entropy"
        )
        ce_loss = ce_per_example_loss.mean()
        raw_loss = per_example_loss.mean()
        weights = self.get_weights(sigmas)
        if broadcast_weights and self.classification_loss_type == "cross_entropy":
            weights = weights[:, None]
        weighted_loss = (per_example_loss * weights).mean()
        losses.append(weighted_loss)
        ce_losses.append(ce_loss)
        loss_dict[f"{log_prefix}/loss_step_{step_index}"] = weighted_loss
        loss_dict[f"{log_prefix}/ce_loss_step_{step_index}"] = ce_loss
        loss_dict[f"{log_prefix}/loss_block_{block_idx}"] = weighted_loss
        loss_dict[f"{log_prefix}/ce_loss_block_{block_idx}"] = ce_loss
        loss_dict[f"{log_prefix}/loss_{block_idx}"] = weighted_loss
        loss_dict[f"{log_prefix}/ce_loss_{block_idx}"] = ce_loss
        if self.classification_loss_type != "cross_entropy":
            loss_name = self.classification_loss_type
            loss_dict[f"{log_prefix}/{loss_name}_step_{step_index}"] = raw_loss
            loss_dict[f"{log_prefix}/{loss_name}_block_{block_idx}"] = raw_loss
            loss_dict[f"{log_prefix}/{loss_name}_{block_idx}"] = raw_loss

    def euler_update(
        self,
        z: torch.Tensor,
        denoise_output: torch.Tensor | dict[str, torch.Tensor],
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
    ) -> torch.Tensor:
        denoised = self.latent_from_denoise_output(denoise_output)
        d = (z - denoised) / sigma[:, None]
        z = z + (next_sigma - sigma)[:, None] * d
        return z.detach()

    def aggregate_denoising_losses(
        self,
        losses: list[torch.Tensor],
        ce_losses: list[torch.Tensor],
        loss_dict: dict[str, torch.Tensor],
        step: str,
    ):
        loss = torch.stack(losses).sum()
        ce_loss = torch.stack(ce_losses).sum()
        loss_dict[f"{step}/loss"] = loss
        if self.task_type == "regression":
            loss_dict[f"{step}/mse"] = ce_loss
        else:
            loss_dict[f"{step}/ce_loss"] = ce_loss
        loss_dict[f"{step}/loss_mean"] = loss / len(losses)
        if self.task_type == "regression":
            loss_dict[f"{step}/mse_mean"] = ce_loss / len(ce_losses)
        else:
            loss_dict[f"{step}/ce_loss_mean"] = ce_loss / len(ce_losses)
        return loss, loss_dict

    def sequential_denoising_training_step(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
        step: str = "train",
    ):
        if self.uses_residual_next_latent_objective():
            return self.residual_next_latent_training_step(
                pixel_values,
                labels,
                step=step,
            )

        batch_size = pixel_values.shape[0]
        hidden_size = self.model.config.hidden_size
        if self.task_type == "classification":
            labels = labels.view(-1)
        else:
            labels = labels.float().view(-1, self.num_labels)
        z = self.sample_epsilon(
            (batch_size, hidden_size),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        z = z * torch.sqrt(1.0 + self.sigmas[0].to(z) ** 2.0)
        s_in = pixel_values.new_ones([batch_size])

        losses = []
        ce_losses = []
        loss_dict = {}

        for step_index in range(self.sigmas.shape[0]):
            sigma = self.sigmas[step_index] * s_in
            block_idx = self.estimate_target_layer(sigma)
            denoise_output = self.denoise(
                pixel_values, z, sigma, block_idx=block_idx
            )
            logits = self.prediction_from_denoise_output(denoise_output)
            self.append_denoising_loss(
                logits,
                labels,
                sigma,
                losses,
                ce_losses,
                loss_dict,
                log_prefix=step,
                step_index=step_index,
                block_idx=block_idx,
            )

            if step_index == self.sigmas.shape[0] - 1:
                continue

            next_sigma = self.sigmas[step_index + 1] * s_in
            z = self.euler_update(z, denoise_output, sigma, next_sigma)

        return self.aggregate_denoising_losses(losses, ce_losses, loss_dict, step)

    def residual_next_latent_training_step(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
        step: str = "train",
    ):
        batch_size = pixel_values.shape[0]
        hidden_size = self.model.config.hidden_size
        labels = labels.view(-1)
        z_clean = self.get_target_latents(labels)
        z = self.sample_epsilon(
            (batch_size, hidden_size),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
        z = z * torch.sqrt(1.0 + self.sigmas[0].to(z) ** 2.0)
        s_in = pixel_values.new_ones([batch_size])

        losses = []
        ce_losses = []
        loss_dict = {}

        for step_index in range(self.sigmas.shape[0]):
            sigma = self.sigmas[step_index] * s_in
            next_sigma = (
                self.sigmas[step_index + 1] * s_in
                if step_index < self.sigmas.shape[0] - 1
                else torch.zeros_like(sigma)
            )
            block_idx = self.estimate_target_layer(sigma)
            z_input = z.detach()
            denoise_output = self.denoise(
                pixel_values,
                z_input,
                sigma,
                block_idx=block_idx,
            )
            latent_delta = self.latent_delta_from_denoise_output(denoise_output)
            z_next_target = z_clean + (next_sigma / sigma)[:, None] * (
                z_input - z_clean
            )
            delta_target = z_next_target - z_input
            per_example_loss = F.mse_loss(
                latent_delta,
                delta_target,
                reduction="none",
            ).mean(dim=-1)
            residual_loss = per_example_loss.mean()
            logits = self.prediction_from_denoise_output(denoise_output)
            ce_loss = self.per_example_classification_loss(
                logits,
                labels,
                loss_type="cross_entropy",
            ).mean()

            losses.append(residual_loss)
            ce_losses.append(ce_loss)
            loss_dict[f"{step}/loss_step_{step_index}"] = residual_loss
            loss_dict[f"{step}/residual_next_latent_mse_step_{step_index}"] = (
                residual_loss
            )
            loss_dict[f"{step}/ce_loss_step_{step_index}"] = ce_loss
            loss_dict[f"{step}/loss_block_{block_idx}"] = residual_loss
            loss_dict[f"{step}/residual_next_latent_mse_block_{block_idx}"] = (
                residual_loss
            )
            loss_dict[f"{step}/ce_loss_block_{block_idx}"] = ce_loss
            loss_dict[f"{step}/loss_{block_idx}"] = residual_loss
            loss_dict[f"{step}/residual_next_latent_mse_{block_idx}"] = residual_loss
            loss_dict[f"{step}/ce_loss_{block_idx}"] = ce_loss

            z = z_input + latent_delta.detach()

        return self.aggregate_denoising_losses(losses, ce_losses, loss_dict, step)

    def hybrid_block0_independent_training_step(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
        step: str = "train",
    ):
        batch_size = pixel_values.shape[0]
        if self.task_type == "classification":
            labels = labels.view(-1)
        else:
            labels = labels.float().view(-1, self.num_labels)
        z_clean = self.get_target_latents(labels)
        s_in = pixel_values.new_ones([batch_size])

        sigma = self.get_sigmas_for_target_block(batch_size, target_block_idx=0).to(
            z_clean
        )
        z = z_clean + sigma[:, None] * self.sample_epsilon_like(z_clean)

        losses = []
        ce_losses = []
        loss_dict = {}

        block_idx = 0
        denoise_output = self.denoise(pixel_values, z, sigma, block_idx=block_idx)
        logits = self.prediction_from_denoise_output(denoise_output)
        self.append_denoising_loss(
            logits,
            labels,
            sigma,
            losses,
            ce_losses,
            loss_dict,
            log_prefix=step,
            step_index=0,
            block_idx=block_idx,
            broadcast_weights=True,
        )
        loss_dict[f"{step}/hybrid_block0_sigma_mean"] = sigma.mean()

        if self.sigmas.shape[0] > 1:
            next_sigma = self.sigmas[1] * s_in
            z = self.euler_update(z, denoise_output, sigma, next_sigma)

        for step_index in range(1, self.sigmas.shape[0]):
            sigma = self.sigmas[step_index] * s_in
            block_idx = self.estimate_target_layer(sigma)
            denoise_output = self.denoise(
                pixel_values, z, sigma, block_idx=block_idx
            )
            logits = self.prediction_from_denoise_output(denoise_output)
            self.append_denoising_loss(
                logits,
                labels,
                sigma,
                losses,
                ce_losses,
                loss_dict,
                log_prefix=step,
                step_index=step_index,
                block_idx=block_idx,
            )

            if step_index == self.sigmas.shape[0] - 1:
                continue

            next_sigma = self.sigmas[step_index + 1] * s_in
            z = self.euler_update(z, denoise_output, sigma, next_sigma)

        return self.aggregate_denoising_losses(losses, ce_losses, loss_dict, step)

    def diffusion_step(self, x, return_intermediates: bool = False):
        outputs = None
        trace_outputs = None
        layer_trace_outputs = None
        trace_metadata = None
        layer_trace_metadata = None
        for _ in range(self.num_prediction_samples):
            sample = self.diffusion_sample(
                x, return_intermediates=return_intermediates
            )
            if return_intermediates:
                logits = sample["logits"]
                trace_logits = sample["intermediate_logits"]
                trace_metadata = {
                    "block_indices": sample["block_indices"],
                    "sigmas": sample["sigmas"],
                }
                if "layer_logits" in sample:
                    layer_trace_logits = sample["layer_logits"]
                    layer_trace_metadata = {
                        "layer_denoise_step_indices": sample[
                            "layer_denoise_step_indices"
                        ],
                        "layer_block_indices": sample["layer_block_indices"],
                        "layer_indices": sample["layer_indices"],
                        "layer_positions": sample["layer_positions"],
                        "layers_in_block": sample["layers_in_block"],
                        "layer_sigmas": sample["layer_sigmas"],
                    }
            else:
                logits = sample
            if self.prediction_average == "probability":
                sample_output = F.softmax(logits.float(), dim=1)
                if return_intermediates:
                    sample_trace_output = F.softmax(trace_logits.float(), dim=-1)
                    if "layer_logits" in sample:
                        sample_layer_trace_output = F.softmax(
                            layer_trace_logits.float(), dim=-1
                        )
            elif self.prediction_average == "logit":
                sample_output = logits.float()
                if return_intermediates:
                    sample_trace_output = trace_logits.float()
                    if "layer_logits" in sample:
                        sample_layer_trace_output = layer_trace_logits.float()
            else:
                raise ValueError(
                    f"Unsupported prediction_average: {self.prediction_average}"
                )
            outputs = sample_output if outputs is None else outputs + sample_output
            if return_intermediates:
                trace_outputs = (
                    sample_trace_output
                    if trace_outputs is None
                    else trace_outputs + sample_trace_output
                )
                if "layer_logits" in sample:
                    layer_trace_outputs = (
                        sample_layer_trace_output
                        if layer_trace_outputs is None
                        else layer_trace_outputs + sample_layer_trace_output
                    )
        outputs = outputs / self.num_prediction_samples
        if return_intermediates:
            trace_outputs = trace_outputs / self.num_prediction_samples
            if layer_trace_outputs is not None:
                layer_trace_outputs = layer_trace_outputs / self.num_prediction_samples
        if self.prediction_average == "probability":
            # Downstream metrics expect logits; log averaged probabilities preserves
            # the requested probability average because softmax(log p) = p.
            outputs = outputs.clamp_min(torch.finfo(outputs.dtype).tiny).log()
            if return_intermediates:
                trace_outputs = trace_outputs.clamp_min(
                    torch.finfo(trace_outputs.dtype).tiny
                ).log()
                if layer_trace_outputs is not None:
                    layer_trace_outputs = layer_trace_outputs.clamp_min(
                        torch.finfo(layer_trace_outputs.dtype).tiny
                    ).log()
        if return_intermediates:
            trace = {
                "logits": trace_outputs,
                "block_indices": trace_metadata["block_indices"],
                "sigmas": trace_metadata["sigmas"],
            }
            if layer_trace_outputs is not None:
                trace.update(
                    {
                        "layer_logits": layer_trace_outputs,
                        **layer_trace_metadata,
                    }
                )
            return outputs, trace
        return outputs

    def append_layer_trace(
        self,
        denoise_output: dict[str, torch.Tensor],
        denoise_step_index: int,
        block_idx: int,
        sigma: torch.Tensor,
        layer_logits: list[torch.Tensor],
        layer_denoise_step_indices: list[int],
        layer_block_indices: list[int],
        layer_indices_trace: list[int],
        layer_positions: list[int],
        layers_in_block_trace: list[int],
        layer_sigmas: list[torch.Tensor],
    ) -> None:
        current_layer_logits = denoise_output["layer_logits"]
        current_layer_indices = denoise_output["layer_indices"].detach().tolist()
        layers_in_block = len(current_layer_indices)
        layer_logits.append(current_layer_logits)
        for layer_position, layer_index in enumerate(current_layer_indices):
            layer_denoise_step_indices.append(denoise_step_index)
            layer_block_indices.append(block_idx)
            layer_indices_trace.append(layer_index)
            layer_positions.append(layer_position)
            layers_in_block_trace.append(layers_in_block)
            layer_sigmas.append(sigma)

    def diffusion_sample(self, x, return_intermediates: bool = False):
        bsz = x.shape[0]
        hidden_size = self.model.config.hidden_size
        z = self.sample_epsilon((bsz, hidden_size), device=x.device)
        z *= torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = x.new_ones([x.shape[0]])
        intermediate_logits = []
        block_indices = []
        sigmas_trace = []
        layer_logits = []
        layer_denoise_step_indices = []
        layer_block_indices = []
        layer_indices_trace = []
        layer_positions = []
        layers_in_block_trace = []
        layer_sigmas = []
        for i in range(self.sigmas.shape[0] - 1):
            sigma = self.sigmas[i] * s_in
            next_sigma = self.sigmas[i + 1] * s_in
            # denoise
            block_idx = self.estimate_target_layer(sigma)
            denoise_output = self.denoise(
                x,
                z,
                sigma,
                block_idx=block_idx,
                return_layer_logits=return_intermediates
                and self.trace_block_layers,
            )
            if isinstance(denoise_output, dict):
                logits = denoise_output["logits"]
            else:
                logits = denoise_output
            if return_intermediates:
                intermediate_logits.append(logits)
                block_indices.append(block_idx)
                sigmas_trace.append(self.sigmas[i])
                if isinstance(denoise_output, dict) and "layer_logits" in denoise_output:
                    self.append_layer_trace(
                        denoise_output,
                        denoise_step_index=i,
                        block_idx=block_idx,
                        sigma=self.sigmas[i],
                        layer_logits=layer_logits,
                        layer_denoise_step_indices=layer_denoise_step_indices,
                        layer_block_indices=layer_block_indices,
                        layer_indices_trace=layer_indices_trace,
                        layer_positions=layer_positions,
                        layers_in_block_trace=layers_in_block_trace,
                        layer_sigmas=layer_sigmas,
                    )
            if self.uses_residual_next_latent_objective():
                z = self.latent_from_denoise_output(denoise_output)
            else:
                denoised = self.latent_from_denoise_output(denoise_output)
                # to d
                d = (z - denoised) / sigma[:, None]
                dt = next_sigma - sigma
                # euler step
                euler_step = z + dt[:, None] * d
                z = euler_step
        min_sigma = self.sigmas[-1].item()
        sigmas = torch.full((x.shape[0],), min_sigma, device=x.device)
        block_idx = self.estimate_target_layer(sigmas)
        denoise_output = self.denoise(
            x,
            z,
            sigmas,
            block_idx=block_idx,
            return_layer_logits=return_intermediates and self.trace_block_layers,
        )
        if isinstance(denoise_output, dict):
            logits = denoise_output["logits"]
        else:
            logits = denoise_output
        if return_intermediates:
            intermediate_logits.append(logits)
            block_indices.append(block_idx)
            sigmas_trace.append(self.sigmas[-1])
            if isinstance(denoise_output, dict) and "layer_logits" in denoise_output:
                self.append_layer_trace(
                    denoise_output,
                    denoise_step_index=self.sigmas.shape[0] - 1,
                    block_idx=block_idx,
                    sigma=self.sigmas[-1],
                    layer_logits=layer_logits,
                    layer_denoise_step_indices=layer_denoise_step_indices,
                    layer_block_indices=layer_block_indices,
                    layer_indices_trace=layer_indices_trace,
                    layer_positions=layer_positions,
                    layers_in_block_trace=layers_in_block_trace,
                    layer_sigmas=layer_sigmas,
                )
            sample = {
                "logits": logits,
                "intermediate_logits": torch.stack(intermediate_logits),
                "block_indices": torch.tensor(
                    block_indices, device=x.device, dtype=torch.long
                ),
                "sigmas": torch.stack(sigmas_trace).to(x.device),
            }
            if layer_logits:
                sample.update(
                    {
                        "layer_logits": torch.cat(layer_logits, dim=0),
                        "layer_denoise_step_indices": torch.tensor(
                            layer_denoise_step_indices,
                            device=x.device,
                            dtype=torch.long,
                        ),
                        "layer_block_indices": torch.tensor(
                            layer_block_indices, device=x.device, dtype=torch.long
                        ),
                        "layer_indices": torch.tensor(
                            layer_indices_trace, device=x.device, dtype=torch.long
                        ),
                        "layer_positions": torch.tensor(
                            layer_positions, device=x.device, dtype=torch.long
                        ),
                        "layers_in_block": torch.tensor(
                            layers_in_block_trace, device=x.device, dtype=torch.long
                        ),
                        "layer_sigmas": torch.stack(layer_sigmas).to(x.device),
                    }
                )
            return sample
        return logits
