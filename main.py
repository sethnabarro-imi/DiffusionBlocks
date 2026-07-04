import os
import argparse
import datetime
import json
import platform
import shlex
import socket
import subprocess
import sys
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy

from data import load_data
from model import load_model, parse_shared_denoising_block_ranges

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _run_command(command):
    try:
        result = subprocess.run(
            command,
            cwd=os.path.dirname(__file__),
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _git_metadata():
    status = _run_command(["git", "status", "--short"])
    return {
        "commit": _run_command(["git", "rev-parse", "HEAD"]),
        "branch": _run_command(["git", "branch", "--show-current"]),
        "is_dirty": bool(status),
        "status_short": status,
    }


def _cuda_metadata():
    devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "capability": [props.major, props.minor],
                }
            )
    return {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": devices,
    }


def _environment_metadata():
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "LOCAL_RANK",
        "RANK",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ID",
        "WANDB_DIR",
        "WANDB_MODE",
        "WORLD_SIZE",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _synthetic_metadata(args):
    if getattr(args, "data_name", None) != "synthetic-teacher":
        return None
    keys = [
        "synthetic_target_type",
        "synthetic_num_train",
        "synthetic_num_test",
        "synthetic_input_dim",
        "synthetic_num_classes",
        "synthetic_target_dim",
        "synthetic_teacher_depth",
        "synthetic_teacher_width",
        "synthetic_teacher_seed",
        "synthetic_train_seed",
        "synthetic_test_seed",
        "synthetic_teacher_activation",
        "synthetic_label_noise",
        "synthetic_target_noise_std",
        "synthetic_enable_checkpointing",
        "synthetic_enable_wandb",
    ]
    return {key: getattr(args, key) for key in keys}


def checkpointing_enabled(args):
    return args.data_name != "synthetic-teacher" or args.synthetic_enable_checkpointing


def wandb_logging_enabled(args):
    return args.data_name != "synthetic-teacher" or args.synthetic_enable_wandb


def _write_json_once(path, data):
    if os.path.exists(path):
        stem, ext = os.path.splitext(path)
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        path = f"{stem}-{timestamp}{ext}"
    with open(path, "w") as f:
        json.dump(data, f, default=str, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _checkpoint_hyperparameters(ckpt_path):
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return {}
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return checkpoint.get("hyper_parameters", {})


def _get_hyperparameter(hparams, key):
    if key in hparams:
        return hparams[key]
    args = hparams.get("args")
    if isinstance(args, dict):
        return args.get(key)
    if hasattr(args, key):
        return getattr(args, key)
    return None


def apply_checkpoint_defaults(args):
    if args.ckpt_path is None:
        return
    if args.model_type != "dblock":
        return
    if args.epsilon_seed is not None:
        return
    hparams = _checkpoint_hyperparameters(args.ckpt_path)
    epsilon_seed = _get_hyperparameter(hparams, "epsilon_seed")
    if epsilon_seed is not None:
        args.epsilon_seed = int(epsilon_seed)
        print(f"Loaded epsilon_seed from checkpoint: {args.epsilon_seed}")


def write_run_metadata(args, logdir):
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    os.makedirs(logdir, exist_ok=True)
    args_dict = vars(args).copy()
    metadata = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": " ".join(
            shlex.quote(part) for part in [sys.executable, *sys.argv]
        ),
        "argv": sys.argv,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "git": _git_metadata(),
        "cuda": _cuda_metadata(),
        "environment": _environment_metadata(),
        "synthetic": _synthetic_metadata(args),
        "args": args_dict,
    }
    args_path = _write_json_once(os.path.join(logdir, "args.json"), args_dict)
    metadata_path = _write_json_once(
        os.path.join(logdir, "run_metadata.json"), metadata
    )
    print(f"Wrote run metadata: {metadata_path}")
    print(f"Wrote args: {args_path}")


def write_eval_results(args, data, logdir, ckpt_path, split_results):
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    payload = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_name": args.data_name,
        "dataset_source": data.data_name,
        "model_type": args.model_type,
        "task_type": getattr(args, "task_type", "classification"),
        "ckpt_path": ckpt_path,
        "ece_num_bins": args.ece_num_bins,
        "input_noise_std": args.input_noise_std,
        "epsilon_seed": args.epsilon_seed,
        "classification_loss_type": getattr(
            args, "classification_loss_type", "cross_entropy"
        ),
        "label_smoothing": getattr(args, "label_smoothing", 0.0),
        "multiclass_hinge_margin": getattr(args, "multiclass_hinge_margin", 1.0),
        "classifier_head_type": getattr(args, "classifier_head_type", "linear"),
        "cosine_classifier_scale": getattr(args, "cosine_classifier_scale", 16.0),
        "one_hot_mse_top_k": getattr(args, "one_hot_mse_top_k", None),
        "num_prediction_samples": args.num_prediction_samples,
        "prediction_average": args.prediction_average,
        "dblock_interblock_transition": getattr(
            args, "dblock_interblock_transition", "euler"
        ),
        "dblock_denoising_space": getattr(args, "dblock_denoising_space", "embedding"),
        "sequential_denoising_training": getattr(
            args, "sequential_denoising_training", False
        ),
        "hybrid_block0_independent_training": getattr(
            args, "hybrid_block0_independent_training", False
        ),
        "shared_denoising_block": getattr(args, "shared_denoising_block", False),
        "shared_denoising_block_index": getattr(
            args, "shared_denoising_block_index", 0
        ),
        "shared_denoising_block_ranges": getattr(
            args, "shared_denoising_block_ranges", ""
        ),
        "trace_block_layers": getattr(args, "trace_block_layers", False),
        "trace_intermediate_predictions": (
            getattr(args, "trace_intermediate_predictions", False)
            or getattr(args, "trace_block_layers", False)
        ),
        "trace_oracle_noise_predictions": getattr(
            args, "trace_oracle_noise_predictions", False
        ),
        "trace_prediction_examples": getattr(args, "trace_prediction_examples", 16),
        "synthetic": _synthetic_metadata(args),
        "splits": {},
    }
    for split, metrics in split_results.items():
        if getattr(args, "task_type", "classification") == "regression":
            split_payload = {
                key: value
                for key, value in metrics.items()
                if key
                not in [
                    "intermediate_predictions",
                    "oracle_noise_predictions",
                ]
            }
        else:
            split_payload = {
                "accuracy": metrics.get("acc"),
                "f1": metrics.get("f1"),
                "ece": metrics.get("ece"),
                "log_likelihood": metrics.get("log_likelihood"),
                "ece_bins": metrics.get("ece_bins", []),
            }
        if "intermediate_predictions" in metrics:
            split_payload["intermediate_predictions"] = metrics[
                "intermediate_predictions"
            ]
        if "oracle_noise_predictions" in metrics:
            split_payload["oracle_noise_predictions"] = metrics[
                "oracle_noise_predictions"
            ]
        payload["splits"][split] = split_payload
    path = _write_json_once(os.path.join(logdir, "eval_results.json"), payload)
    print(f"Wrote eval results: {path}")


def move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def compute_eval_split_results(model, split):
    metrics = model.get_eval_metrics(split)
    split_results = {}
    for metric_name, metric in metrics.items():
        with metric.sync_context():
            value = metric.compute().detach().float().cpu().item()
        split_results[model.strip_metric_prefix(metric_name)] = value

    calibration_metric = model.get_calibration_metric(split)
    if calibration_metric is not None:
        with calibration_metric.sync_context():
            stats = calibration_metric.calibration_stats()
        split_results["ece_bins"] = model.calibration_rows(stats)

    if getattr(model, "trace_intermediate_predictions", False):
        split_results["intermediate_predictions"] = (
            model.intermediate_prediction_results(split)
        )
    if getattr(model, "trace_oracle_noise_predictions", False):
        split_results["oracle_noise_predictions"] = (
            model.oracle_noise_prediction_results(split)
        )
    return split_results


def reset_eval_split_metrics(model, split):
    model.get_eval_metrics(split).reset()
    if getattr(model, "trace_intermediate_predictions", False):
        model.get_intermediate_prediction_metric(split).reset()
        model.intermediate_prediction_examples_by_split[split] = []
        if getattr(model, "trace_block_layers", False):
            model.get_layer_prediction_metric(split).reset()
            model.layer_prediction_examples_by_split[split] = []
    if getattr(model, "trace_oracle_noise_predictions", False):
        model.get_oracle_noise_prediction_metric(split).reset()
        model.oracle_noise_prediction_examples_by_split[split] = []
        if getattr(model, "trace_block_layers", False):
            model.get_oracle_noise_layer_prediction_metric(split).reset()
            model.oracle_noise_layer_prediction_examples_by_split[split] = []


def run_current_model_split_evaluation(model, dataloader, split):
    previous_eval_split = model.eval_split
    model.eval_split = split
    reset_eval_split_metrics(model, split)
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, model.device)
            model_kwargs = model.get_model_kwargs(batch)
            model.shared_step(
                batch,
                step=split,
                return_metrics=True,
                **model_kwargs,
            )
    split_results = compute_eval_split_results(model, split)
    model.eval_split = previous_eval_split
    return split_results


def write_blockwise_training_eval_results(
    args,
    data,
    logdir,
    split_results,
    epoch,
    step,
):
    if int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return
    payload = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_name": args.data_name,
        "dataset_source": data.data_name,
        "model_type": args.model_type,
        "task_type": getattr(args, "task_type", "classification"),
        "epoch": epoch,
        "global_step": step,
        "ece_num_bins": args.ece_num_bins,
        "input_noise_std": args.input_noise_std,
        "epsilon_seed": args.epsilon_seed,
        "classification_loss_type": getattr(
            args, "classification_loss_type", "cross_entropy"
        ),
        "label_smoothing": getattr(args, "label_smoothing", 0.0),
        "multiclass_hinge_margin": getattr(args, "multiclass_hinge_margin", 1.0),
        "classifier_head_type": getattr(args, "classifier_head_type", "linear"),
        "cosine_classifier_scale": getattr(args, "cosine_classifier_scale", 16.0),
        "one_hot_mse_top_k": getattr(args, "one_hot_mse_top_k", None),
        "num_prediction_samples": args.num_prediction_samples,
        "prediction_average": args.prediction_average,
        "dblock_interblock_transition": getattr(
            args, "dblock_interblock_transition", "euler"
        ),
        "dblock_denoising_space": getattr(args, "dblock_denoising_space", "embedding"),
        "sequential_denoising_training": getattr(
            args, "sequential_denoising_training", False
        ),
        "hybrid_block0_independent_training": getattr(
            args, "hybrid_block0_independent_training", False
        ),
        "trace_block_layers": getattr(args, "trace_block_layers", False),
        "trace_intermediate_predictions": True,
        "trace_oracle_noise_predictions": getattr(
            args, "trace_oracle_noise_predictions", False
        ),
        "trace_prediction_examples": getattr(args, "trace_prediction_examples", 16),
        "synthetic": _synthetic_metadata(args),
        "splits": {},
    }
    for split, metrics in split_results.items():
        if getattr(args, "task_type", "classification") == "regression":
            split_payload = {
                key: value
                for key, value in metrics.items()
                if key
                not in [
                    "intermediate_predictions",
                    "oracle_noise_predictions",
                ]
            }
        else:
            split_payload = {
                "accuracy": metrics.get("acc"),
                "f1": metrics.get("f1"),
                "ece": metrics.get("ece"),
                "log_likelihood": metrics.get("log_likelihood"),
                "ece_bins": metrics.get("ece_bins", []),
            }
        if "intermediate_predictions" in metrics:
            split_payload["intermediate_predictions"] = metrics[
                "intermediate_predictions"
            ]
        if "oracle_noise_predictions" in metrics:
            split_payload["oracle_noise_predictions"] = metrics[
                "oracle_noise_predictions"
            ]
        payload["splits"][split] = split_payload
    path = _write_json_once(
        os.path.join(logdir, f"blockwise_eval_epoch_{epoch:05d}.json"),
        payload,
    )
    print(f"Wrote periodic blockwise eval results: {path}")


class PeriodicBlockwiseEvalCallback(Callback):
    def __init__(self, data, args, logdir):
        super().__init__()
        self.data = data
        self.args = args
        self.logdir = logdir

    def on_train_epoch_end(self, trainer, pl_module):
        frequency = self.args.blockwise_eval_every_n_epochs
        if frequency <= 0:
            return
        epoch = trainer.current_epoch + 1
        if epoch % frequency != 0:
            return

        was_training = pl_module.training
        pl_module.eval()
        split_results = {}
        if "train_eval" in self.data.datasets:
            trainer.print(f"Running blockwise train_eval diagnostics at epoch {epoch}")
            split_results["train_eval"] = run_current_model_split_evaluation(
                pl_module,
                self.data.train_eval_dataloader(),
                "train_eval",
            )
        if self.data.test_key is not None:
            trainer.print(f"Running blockwise test diagnostics at epoch {epoch}")
            split_results["test"] = run_current_model_split_evaluation(
                pl_module,
                self.data.test_dataloader(),
                "test",
            )
        if was_training:
            pl_module.train()

        write_blockwise_training_eval_results(
            self.args,
            self.data,
            self.logdir,
            split_results,
            epoch=epoch,
            step=trainer.global_step,
        )


def run_train_test_evaluation(trainer, model, data, ckpt_path, args, logdir):
    model.eval_results_by_split = {}
    data.setup("test")
    if "train_eval" in data.datasets:
        model.eval_split = "train_eval"
        trainer.test(model, data.train_eval_dataloader(), ckpt_path=ckpt_path)
    if data.test_key is not None:
        model.eval_split = "test"
        trainer.test(model, data.test_dataloader(), ckpt_path=ckpt_path)
    write_eval_results(
        args,
        data,
        logdir,
        ckpt_path,
        model.eval_results_by_split,
    )


def validate_args(args):
    if args.num_hidden_layers < 1:
        raise ValueError("--num_hidden_layers must be at least 1")
    for key in ["attention_probs_dropout_prob", "hidden_dropout_prob"]:
        value = getattr(args, key)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{key} must be between 0 and 1")
    if args.blockwise_eval_every_n_epochs < 0:
        raise ValueError("--blockwise_eval_every_n_epochs must be non-negative")
    if args.label_smoothing < 0.0 or args.label_smoothing > 1.0:
        raise ValueError("--label_smoothing must be between 0 and 1")
    if (
        args.label_smoothing > 0.0
        and args.classification_loss_type != "cross_entropy"
    ):
        raise ValueError(
            "--label_smoothing requires --classification_loss_type cross_entropy"
        )
    if args.multiclass_hinge_margin <= 0.0:
        raise ValueError("--multiclass_hinge_margin must be positive")
    if args.cosine_classifier_scale <= 0.0:
        raise ValueError("--cosine_classifier_scale must be positive")
    if args.one_hot_mse_top_k is not None:
        if args.one_hot_mse_top_k < 0:
            raise ValueError("--one_hot_mse_top_k must be non-negative")
        if args.classification_loss_type != "one_hot_mse":
            raise ValueError(
                "--one_hot_mse_top_k requires --classification_loss_type one_hot_mse"
            )
    if args.blockwise_eval_every_n_epochs > 0 and args.model_type != "dblock":
        raise ValueError("--blockwise_eval_every_n_epochs is only supported for dblock")
    if args.trace_oracle_noise_predictions and args.model_type != "dblock":
        raise ValueError("--trace_oracle_noise_predictions is only supported for dblock")
    if args.sequential_denoising_training and args.model_type != "dblock":
        raise ValueError("--sequential_denoising_training is only supported for dblock")
    if args.dblock_interblock_transition != "euler" and args.model_type != "dblock":
        raise ValueError("--dblock_interblock_transition is only supported for dblock")
    if args.dblock_denoising_space != "embedding" and args.model_type != "dblock":
        raise ValueError("--dblock_denoising_space is only supported for dblock")
    if args.hybrid_block0_independent_training and args.model_type != "dblock":
        raise ValueError(
            "--hybrid_block0_independent_training is only supported for dblock"
        )
    if args.shared_denoising_block and args.model_type != "dblock":
        raise ValueError("--shared_denoising_block is only supported for dblock")
    if args.shared_denoising_block_ranges and args.model_type != "dblock":
        raise ValueError(
            "--shared_denoising_block_ranges is only supported for dblock"
        )
    if args.shared_denoising_block and args.shared_denoising_block_ranges:
        raise ValueError(
            "--shared_denoising_block and --shared_denoising_block_ranges are "
            "mutually exclusive"
        )
    if args.num_blocks < 1:
        raise ValueError("--num_blocks must be at least 1")
    if args.shared_denoising_block_index < 0:
        raise ValueError("--shared_denoising_block_index must be non-negative")
    if args.shared_denoising_block_index >= args.num_blocks:
        raise ValueError(
            "--shared_denoising_block_index must be smaller than --num_blocks"
        )
    parse_shared_denoising_block_ranges(
        args.shared_denoising_block_ranges,
        args.num_blocks,
    )
    if args.dblock_training_objective != "classification":
        if args.model_type != "dblock":
            raise ValueError("--dblock_training_objective is only supported for dblock")
        if args.dblock_latent_loss_weight < 0.0:
            raise ValueError("--dblock_latent_loss_weight must be non-negative")
        if args.dblock_prediction_loss_weight < 0.0:
            raise ValueError("--dblock_prediction_loss_weight must be non-negative")
        if (
            args.dblock_latent_loss_weight == 0.0
            and args.dblock_prediction_loss_weight == 0.0
        ):
            raise ValueError(
                "at least one of --dblock_latent_loss_weight or "
                "--dblock_prediction_loss_weight must be positive"
            )
        if args.dblock_training_objective in [
            "residual_next_latent",
            "residual_to_clean",
        ]:
            if not args.sequential_denoising_training:
                raise ValueError(
                    f"--dblock_training_objective {args.dblock_training_objective} requires "
                    "--sequential_denoising_training"
                )
            if args.hybrid_block0_independent_training:
                raise ValueError(
                    f"--dblock_training_objective {args.dblock_training_objective} is not "
                    "currently supported with --hybrid_block0_independent_training"
                )
    if (
        args.dblock_interblock_transition != "euler"
        and args.dblock_training_objective
        not in ["classification", "residual_to_clean"]
    ):
        raise ValueError(
            "--dblock_interblock_transition direct modes are currently supported "
            "only with --dblock_training_objective classification or "
            "residual_to_clean"
        )
    if args.dblock_denoising_space == "logits":
        if getattr(args, "task_type", "classification") != "classification":
            raise ValueError("--dblock_denoising_space logits requires classification")
        if args.dblock_training_objective != "classification":
            raise ValueError(
                "--dblock_denoising_space logits is currently supported only with "
                "--dblock_training_objective classification"
            )
    if args.data_name == "synthetic-teacher":
        if args.synthetic_num_train < 1:
            raise ValueError("--synthetic_num_train must be at least 1")
        if args.synthetic_num_test < 1:
            raise ValueError("--synthetic_num_test must be at least 1")
        if args.synthetic_input_dim < 1:
            raise ValueError("--synthetic_input_dim must be at least 1")
        if args.synthetic_num_classes < 2:
            raise ValueError("--synthetic_num_classes must be at least 2")
        if args.synthetic_target_dim < 1:
            raise ValueError("--synthetic_target_dim must be at least 1")
        if args.synthetic_teacher_depth < 1:
            raise ValueError("--synthetic_teacher_depth must be at least 1")
        if args.synthetic_teacher_width < 1:
            raise ValueError("--synthetic_teacher_width must be at least 1")
        if args.synthetic_label_noise < 0.0 or args.synthetic_label_noise > 1.0:
            raise ValueError("--synthetic_label_noise must be between 0 and 1")
        if args.synthetic_target_noise_std < 0.0:
            raise ValueError("--synthetic_target_noise_std must be non-negative")
        if (
            args.synthetic_target_type == "continuous"
            and args.prediction_average == "probability"
        ):
            args.prediction_average = "logit"


def main(args):
    validate_args(args)
    if args.hybrid_block0_independent_training:
        args.sequential_denoising_training = True
    if args.blockwise_eval_every_n_epochs > 0:
        args.trace_intermediate_predictions = True
    L.seed_everything(args.seed)

    data = load_data(args)
    args.image_size = data.image_size
    args.num_labels = data.num_labels
    args.task_type = getattr(data, "task_type", "classification")
    apply_checkpoint_defaults(args)
    model = load_model(args)
    if args.ckpt_path is not None:
        nowname = os.path.basename(os.path.dirname(args.ckpt_path))
    else:
        now = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9), name="JST")
        ).strftime("%Y-%m-%dT%H-%M-%S")
        nowname = now + f"-{args.model_type}" + args.postfix
        if nowname.startswith("_"):
            nowname = nowname[1:]
    print("Experiment Name:", nowname)
    logdir = os.path.join("logs", nowname)
    write_run_metadata(args, logdir)
    use_checkpointing = checkpointing_enabled(args)
    use_wandb_logging = wandb_logging_enabled(args)
    logger = (
        WandbLogger(
            project=f"dblocks-{args.data_name}",
            name=nowname,
            version=nowname,
            offline=args.debug,
            save_dir=logdir,
            # group=f"{args.data_name}",
        )
        if use_wandb_logging
        else False
    )
    callbacks = []
    if use_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=logdir,
                monitor="val/acc" if data.val_key is not None else None,
                mode="max",
                save_top_k=args.save_top_k,
                save_on_train_epoch_end=True,
                every_n_epochs=args.save_every_n_epochs
                if data.val_key is None
                else None,
                save_last=True,
            )
        )
    if use_wandb_logging:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if args.blockwise_eval_every_n_epochs > 0:
        callbacks.append(PeriodicBlockwiseEvalCallback(data, args, logdir))
    max_epochs = args.num_epochs
    if args.model_type == "dblock" and not args.sequential_denoising_training:
        # In the original independent-block objective, each optimizer step trains
        # only one sampled block, so multiply epochs to keep per-block updates
        # comparable with a full-network ViT epoch count.
        max_epochs = args.num_epochs * args.num_blocks
    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=args.save_every_n_epochs,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        strategy=DDPStrategy(find_unused_parameters=args.model_type == "dblock")
        if args.devices > 1
        else "auto",
        devices=args.devices,
        logger=logger,
        enable_checkpointing=use_checkpointing,
        num_sanity_val_steps=0,
        # precision="bf16-mixed",
    )
    if args.stage == "train":
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
        run_train_test_evaluation(
            trainer,
            model,
            data,
            ckpt_path="best" if use_checkpointing else None,
            args=args,
            logdir=logdir,
        )
    else:
        assert args.ckpt_path is not None
        run_train_test_evaluation(
            trainer, model, data, ckpt_path=args.ckpt_path, args=args, logdir=logdir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, default="train", choices=["train", "test"])
    parser.add_argument("data_name", type=str, default="cifar100")
    parser.add_argument(
        "--model_type", type=str, default="vit", choices=["vit", "dblock"]
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--add_rand_aug", action="store_true")
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=12,
        help="number of transformer layers in the ViT backbone",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="dropout probability applied to attention probabilities",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="dropout probability applied to hidden states",
    )
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--ece_num_bins", type=int, default=15)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument(
        "--scheduler_specific_kwargs",
        type=json.loads,
        default=None,
        help="specific kwargs for the scheduler",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument(
        "--classification_loss_type",
        type=str,
        default="cross_entropy",
        choices=[
            "cross_entropy",
            "one_hot_mse",
            "brier_score",
            "multiclass_hinge",
            "squared_multiclass_hinge",
        ],
        help=(
            "classification loss used for training. cross_entropy keeps the "
            "current logit CE objective; one_hot_mse treats the class-vector "
            "output as a direct one-hot prediction; brier_score applies MSE to "
            "softmax probabilities; multiclass_hinge and "
            "squared_multiclass_hinge use a max-competing-class margin loss"
        ),
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help=(
            "label smoothing factor for cross-entropy training; 0 keeps hard "
            "targets, e.g. 0.1 trains against 90%% true-class mass and spreads "
            "the remainder over other classes"
        ),
    )
    parser.add_argument(
        "--one_hot_mse_top_k",
        type=int,
        default=None,
        help=(
            "if set with --classification_loss_type one_hot_mse, compute MSE "
            "only over the union of the true class and the top-N raw predicted "
            "class outputs; unset keeps full-vector one-hot MSE"
        ),
    )
    parser.add_argument(
        "--multiclass_hinge_margin",
        type=float,
        default=1.0,
        help=(
            "margin used by --classification_loss_type multiclass_hinge and "
            "squared_multiclass_hinge"
        ),
    )
    parser.add_argument(
        "--classifier_head_type",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help=(
            "classifier head parameterization. cosine normalizes features and "
            "class weights, then multiplies cosine similarities by a fixed scale"
        ),
    )
    parser.add_argument(
        "--cosine_classifier_scale",
        type=float,
        default=16.0,
        help="fixed logit scale for --classifier_head_type cosine",
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--postfix", type=str, default="", help="postfix for the experiment name"
    )
    # dblock
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--num_prediction_samples", type=int, default=1)
    parser.add_argument(
        "--prediction_average",
        type=str,
        default="probability",
        choices=["probability", "logit"],
    )
    parser.add_argument(
        "--dblock_interblock_transition",
        type=str,
        default="euler",
        choices=[
            "euler",
            "direct_denoised",
            "direct_denoised_plus_noise",
            "direct_denoised_plus_rescaled_noise",
        ],
        help=(
            "state transition between DBlock denoising steps for the "
            "classification objective. euler keeps the scheduled diffusion "
            "Euler update; direct_denoised feeds the predicted clean embedding "
            "directly to the next block; direct_denoised_plus_noise feeds the "
            "predicted clean embedding after adding fresh noise at the next "
            "scheduled sigma; direct_denoised_plus_rescaled_noise reuses one "
            "Gaussian noise sample per example and rescales it by each next "
            "scheduled sigma"
        ),
    )
    parser.add_argument(
        "--dblock_denoising_space",
        type=str,
        default="embedding",
        choices=["embedding", "logits"],
        help=(
            "state space for classification DBlock denoising. embedding keeps "
            "the original noised label-embedding state; logits noises a "
            "num-label class-vector state directly and projects it into the "
            "ViT block input"
        ),
    )
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--class_dropout_prob", type=float, default=0.0)
    parser.add_argument(
        "--sequential_denoising_training",
        action="store_true",
        help=(
            "train DBlock by running the full denoising chain in each training "
            "step, feeding each block the Euler-updated output from the previous "
            "block; off keeps the original independent-block training objective"
        ),
    )
    parser.add_argument(
        "--hybrid_block0_independent_training",
        action="store_true",
        help=(
            "train DBlock with block 0 using the original independent block-0 "
            "noise objective, then feed its Euler-updated state into the "
            "sequential denoising chain for the remaining blocks; implies "
            "--sequential_denoising_training"
        ),
    )
    parser.add_argument(
        "--shared_denoising_block",
        action="store_true",
        help=(
            "route every DBlock denoising/noise step through the same block "
            "network instead of selecting a separate block by sigma; this "
            "composes with sequential and residual DBlock objectives"
        ),
    )
    parser.add_argument(
        "--shared_denoising_block_index",
        type=int,
        default=0,
        help=(
            "which block network to reuse when --shared_denoising_block is set"
        ),
    )
    parser.add_argument(
        "--shared_denoising_block_ranges",
        type=str,
        default="",
        help=(
            "comma-separated half-open block sharing ranges. Each entry is "
            "start:end or start:end:network_index; start:end reuses the "
            "network at start for denoising block indices start through end-1, "
            "for example 0:4,4:8,8:12"
        ),
    )
    parser.add_argument(
        "--dblock_training_objective",
        type=str,
        default="classification",
        choices=["classification", "residual_next_latent", "residual_to_clean"],
        help=(
            "DBlock training objective. classification keeps the existing "
            "logit/clean-target denoising objective; residual_next_latent makes "
            "each block predict a hidden-size residual delta and applies "
            "z_next_hat = z + delta_hat toward the next scheduled latent state; "
            "residual_to_clean makes each block predict r_hat = z_clean - z "
            "and updates with z_next = z + alpha * r_hat"
        ),
    )
    parser.add_argument(
        "--dblock_residual_readout_type",
        type=str,
        default="classifier",
        choices=["classifier", "label_embedding_cosine"],
        help=(
            "readout used for residual DBlock objectives. classifier applies the "
            "configured classifier head to the predicted clean latent; with the "
            "default --classifier_head_type linear this is a dense projection. "
            "label_embedding_cosine keeps the previous cosine-similarity readout "
            "against the learned label embedding table"
        ),
    )
    parser.add_argument(
        "--dblock_latent_loss_weight",
        type=float,
        default=1.0,
        help="weight on residual latent MSE for residual DBlock objectives",
    )
    parser.add_argument(
        "--dblock_prediction_loss_weight",
        type=float,
        default=1.0,
        help=(
            "weight on cross-entropy prediction loss for residual DBlock "
            "objectives; set to 0 to recover latent-only residual training"
        ),
    )
    parser.add_argument(
        "--epsilon_seed",
        type=int,
        default=None,
        help="if set, reset epsilon sampling to this seed for every dblock noise draw",
    )
    parser.add_argument(
        "--trace_intermediate_predictions",
        action="store_true",
        help="record per-denoising-step DBlock prediction diagnostics during eval",
    )
    parser.add_argument(
        "--trace_block_layers",
        action="store_true",
        help="also record predictions after each transformer layer inside each DBlock",
    )
    parser.add_argument(
        "--trace_oracle_noise_predictions",
        action="store_true",
        help=(
            "record oracle+noise DBlock diagnostics: each block predicts from "
            "true label embeddings plus the corresponding scheduled noise"
        ),
    )
    parser.add_argument(
        "--trace_prediction_examples",
        type=int,
        default=16,
        help="number of per-example DBlock prediction traces to include in eval JSON",
    )
    parser.add_argument(
        "--blockwise_eval_every_n_epochs",
        type=int,
        default=0,
        help=(
            "during training, run blockwise prediction diagnostics on train_eval "
            "and test every N Lightning epochs; 0 disables this"
        ),
    )
    # synthetic teacher dataset
    parser.add_argument(
        "--synthetic_target_type",
        type=str,
        default="classification",
        choices=["classification", "continuous"],
        help="target family for synthetic-teacher",
    )
    parser.add_argument("--synthetic_num_train", type=int, default=4096)
    parser.add_argument("--synthetic_num_test", type=int, default=2048)
    parser.add_argument(
        "--synthetic_input_dim",
        type=int,
        default=128,
        help="number of sampled Gaussian input features before padding to an image",
    )
    parser.add_argument(
        "--synthetic_num_classes",
        type=int,
        default=10,
        help="number of teacher argmax classes for synthetic-teacher",
    )
    parser.add_argument(
        "--synthetic_target_dim",
        type=int,
        default=4,
        help="number of continuous target units for synthetic-teacher regression",
    )
    parser.add_argument(
        "--synthetic_teacher_depth",
        type=int,
        default=3,
        help="number of linear layers in the frozen synthetic teacher MLP",
    )
    parser.add_argument(
        "--synthetic_teacher_width",
        type=int,
        default=256,
        help="hidden width of the frozen synthetic teacher MLP",
    )
    parser.add_argument("--synthetic_teacher_seed", type=int, default=123)
    parser.add_argument("--synthetic_train_seed", type=int, default=1000)
    parser.add_argument("--synthetic_test_seed", type=int, default=2000)
    parser.add_argument(
        "--synthetic_teacher_activation",
        type=str,
        default="gelu",
        choices=["gelu", "relu", "tanh"],
    )
    parser.add_argument("--synthetic_label_noise", type=float, default=0.0)
    parser.add_argument("--synthetic_target_noise_std", type=float, default=0.0)
    parser.add_argument(
        "--synthetic_enable_checkpointing",
        action="store_true",
        help=(
            "opt into checkpoint files for synthetic-teacher diagnostics; "
            "image datasets still checkpoint by default"
        ),
    )
    parser.add_argument(
        "--synthetic_enable_wandb",
        action="store_true",
        help=(
            "opt into W&B logging for synthetic-teacher diagnostics; image "
            "datasets still use W&B by default"
        ),
    )
    args = parser.parse_args()
    main(args)
