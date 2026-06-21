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
from model import load_model

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
        "ckpt_path": ckpt_path,
        "ece_num_bins": args.ece_num_bins,
        "input_noise_std": args.input_noise_std,
        "epsilon_seed": args.epsilon_seed,
        "num_prediction_samples": args.num_prediction_samples,
        "prediction_average": args.prediction_average,
        "trace_block_layers": getattr(args, "trace_block_layers", False),
        "trace_intermediate_predictions": (
            getattr(args, "trace_intermediate_predictions", False)
            or getattr(args, "trace_block_layers", False)
        ),
        "trace_prediction_examples": getattr(args, "trace_prediction_examples", 16),
        "splits": {},
    }
    for split, metrics in split_results.items():
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
    with calibration_metric.sync_context():
        stats = calibration_metric.calibration_stats()
    split_results["ece_bins"] = model.calibration_rows(stats)

    if getattr(model, "trace_intermediate_predictions", False):
        split_results["intermediate_predictions"] = (
            model.intermediate_prediction_results(split)
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
        "epoch": epoch,
        "global_step": step,
        "ece_num_bins": args.ece_num_bins,
        "input_noise_std": args.input_noise_std,
        "epsilon_seed": args.epsilon_seed,
        "num_prediction_samples": args.num_prediction_samples,
        "prediction_average": args.prediction_average,
        "trace_block_layers": getattr(args, "trace_block_layers", False),
        "trace_intermediate_predictions": True,
        "trace_prediction_examples": getattr(args, "trace_prediction_examples", 16),
        "splits": {},
    }
    for split, metrics in split_results.items():
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
    if args.blockwise_eval_every_n_epochs > 0 and args.model_type != "dblock":
        raise ValueError("--blockwise_eval_every_n_epochs is only supported for dblock")


def main(args):
    validate_args(args)
    if args.blockwise_eval_every_n_epochs > 0:
        args.trace_intermediate_predictions = True
    L.seed_everything(args.seed)

    data = load_data(args)
    args.image_size = data.image_size
    args.num_labels = data.num_labels
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
    logger = WandbLogger(
        project=f"dblocks-{args.data_name}",
        name=nowname,
        version=nowname,
        offline=args.debug,
        save_dir=logdir,
        # group=f"{args.data_name}",
    )
    callbacks = [
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
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.blockwise_eval_every_n_epochs > 0:
        callbacks.append(PeriodicBlockwiseEvalCallback(data, args, logdir))
    trainer = L.Trainer(
        max_epochs=args.num_epochs
        if args.model_type != "dblock"
        else args.num_epochs
        * args.num_blocks,  # to align total number of iterations across the entire network because one step corresponds to one block
        check_val_every_n_epoch=args.save_every_n_epochs,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        strategy=DDPStrategy(find_unused_parameters=args.model_type == "dblock")
        if args.devices > 1
        else "auto",
        devices=args.devices,
        logger=logger,
        num_sanity_val_steps=0,
        # precision="bf16-mixed",
    )
    if args.stage == "train":
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
        run_train_test_evaluation(
            trainer, model, data, ckpt_path="best", args=args, logdir=logdir
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
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--class_dropout_prob", type=float, default=0.0)
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
    args = parser.parse_args()
    main(args)
