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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
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
        "num_prediction_samples": args.num_prediction_samples,
        "prediction_average": args.prediction_average,
        "splits": {},
    }
    for split, metrics in split_results.items():
        payload["splits"][split] = {
            "accuracy": metrics.get("acc"),
            "f1": metrics.get("f1"),
            "ece": metrics.get("ece"),
            "log_likelihood": metrics.get("log_likelihood"),
            "ece_bins": metrics.get("ece_bins", []),
        }
    path = _write_json_once(os.path.join(logdir, "eval_results.json"), payload)
    print(f"Wrote eval results: {path}")


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


def main(args):
    L.seed_everything(args.seed)

    data = load_data(args)
    args.image_size = data.image_size
    args.num_labels = data.num_labels
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
    trainer = L.Trainer(
        max_epochs=args.num_epochs
        if args.model_type != "dblock"
        else args.num_epochs
        * args.num_blocks,  # to align total number of iterations across the entire network because one step corresponds to one block
        check_val_every_n_epoch=args.save_every_n_epochs,
        callbacks=[
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
        ],
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
    args = parser.parse_args()
    main(args)
