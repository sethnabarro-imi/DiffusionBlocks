# DiffusionBlocks (ICLR 2026)

<div align="center">
<img alt="overview" src="./overview.jpg" title="overview">
</div>

> We propose ***DiffusionBlocks***, a principled framework that partitions transformers into independently trainable blocks, reducing memory requirements proportionally while maintaining competitive performance across diverse architectures and tasks.

This is an official implementation of *[DiffusionBlocks](https://arxiv.org/abs/2506.14202)* on image classification using Vision Transformers (ViT).

## Installation

Please install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then, run:

```bash
# Install dependencies
uv sync

# make sure to login huggingface and wandb
uv run huggingface-cli login
uv run wandb login
```

We conducted our experiments in the following environment: Python Version 3.12 and CUDA Version 12.2 H100.

### Remote machine setup

On a fresh remote checkout, run:

```bash
scripts/prepare_remote.sh
source .remote.env
uv run huggingface-cli login
uv run wandb login
```

The script installs `uv` if needed, creates scratch directories for logs,
Hugging Face caches, and W&B files, writes `.remote.env`, symlinks `logs` to
scratch when it can do so safely, and runs `uv sync --frozen`.

## Training

The model checkpoints are saved in `logs` folder. Each run also writes
`args.json` and `run_metadata.json` into its run directory with the parsed
arguments, command, git state, hostname, Python version, and CUDA device
information. If a run directory already has metadata, later invocations write
timestamped metadata files instead of overwriting the original files.

**Baseline (ViT):**

```bash
uv run main.py train cifar100 --model_type vit
```

**DiffusionBlocks:**

```bash
uv run main.py train cifar100 --model_type dblock
```

* **NOTE:** the total epochs in DiffusionBlocks is multiplied by the number of blocks to align the total number of iterations with the baseline as one step in DiffusionBlocks corresponds to training for one block.

<details>

In the base setting, we don't reply on techniques such as heavy data augmentation. In case you want to see the performance with heavy data augmentation and learning rate scheduler, run as follows:

**Baseline (ViT):**

```bash
BATCH_SIZE=128
EPOCHS=1000
POSTFIX="-rand-augment"
WARMUP_STEPS=3900
MODEL_TYPE="dblock"
srun uv run main.py train cifar100 \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE --num_epochs $EPOCHS --postfix=$POSTFIX \
    --scheduler_type cosine_with_min_lr --num_warmup_steps $WARMUP_STEPS --lr 5e-4 \
    --scheduler_specific_kwargs '{"min_lr": 5e-5}' \
    --add_rand_aug
```

**DiffusionBlocks:**

```bash
BATCH_SIZE=128
EPOCHS=1000
POSTFIX="-rand-augment"
WARMUP_STEPS=$((3900 * 3)) # 3 indicates the number of blocks
MODEL_TYPE="dblock"
srun uv run main.py train cifar100 \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE --num_epochs $EPOCHS --postfix=$POSTFIX \
    --scheduler_type cosine_with_min_lr --num_warmup_steps $WARMUP_STEPS --lr 5e-4 \
    --scheduler_specific_kwargs '{"min_lr": 5e-5}' \
    --add_rand_aug
```

</details>

### Synthetic teacher experiments

For a minimal controlled task, use `synthetic-teacher`. Inputs are sampled as
Gaussian vectors, padded into `3 x 32 x 32` tensors for the existing ViT/DBlock
model, and labeled by a frozen random MLP teacher. Synthetic runs write JSON
diagnostics under `logs/<run-name>/`, but checkpointing and W&B logging are off
by default.

```bash
uv run main.py train synthetic-teacher \
    --model_type dblock \
    --num_blocks 6 \
    --num_hidden_layers 12 \
    --num_epochs 50 \
    --batch_size 128 \
    --synthetic_num_train 4096 \
    --synthetic_num_test 2048 \
    --synthetic_input_dim 128 \
    --synthetic_num_classes 10 \
    --synthetic_teacher_depth 3 \
    --synthetic_teacher_width 256 \
    --sequential_denoising_training \
    --trace_oracle_noise_predictions \
    --blockwise_eval_every_n_epochs 10 \
    --postfix=-synthetic-teacher
```

To compare categorical cross-entropy with a one-hot MSE likelihood, add:

```bash
--classification_loss_type one_hot_mse
```

To train cross-entropy with smoothed classification targets, keep the default
loss type and add:

```bash
--label_smoothing 0.1
```

To use alternatives that avoid unbounded cross-entropy margins:

```bash
--classification_loss_type brier_score
--classification_loss_type multiclass_hinge
--classification_loss_type squared_multiclass_hinge
```

The hinge losses use `--multiclass_hinge_margin 1.0` by default.

To replace the linear classifier with a fixed-scale cosine classifier, add:

```bash
--classifier_head_type cosine --cosine_classifier_scale 16.0
```

To ablate the inter-block state transition during sequential DBlock denoising,
replace the default Euler update with a direct handoff of the predicted clean
embedding:

```bash
--sequential_denoising_training --dblock_interblock_transition direct_denoised
```

To test whether re-adding scheduled noise between blocks helps, compare against:

```bash
--sequential_denoising_training --dblock_interblock_transition direct_denoised_plus_noise
```

To re-add the same Gaussian noise sample at every step, rescaled by the next
scheduled sigma, use:

```bash
--sequential_denoising_training --dblock_interblock_transition direct_denoised_plus_rescaled_noise
```

For continuous teacher targets, switch the target type and set the output
dimension:

```bash
uv run main.py train synthetic-teacher \
    --model_type dblock \
    --synthetic_target_type continuous \
    --synthetic_target_dim 8 \
    --num_blocks 6 \
    --num_hidden_layers 12 \
    --num_epochs 50 \
    --sequential_denoising_training \
    --trace_oracle_noise_predictions \
    --blockwise_eval_every_n_epochs 10 \
    --postfix=-synthetic-regression
```

To opt back into heavier training artifacts for synthetic runs, add:

```bash
--synthetic_enable_checkpointing --synthetic_enable_wandb
```

After training, turn a blockwise eval JSON file into a CSV and accuracy plot:

```bash
uv run scripts/plot_blockwise_eval.py \
    logs/<run-name>/blockwise_eval_epoch_00050.json
```

### Toy 1D regression DBlock

For a very small local diagnostic that does not use Lightning, W&B,
checkpoints, images, or the ViT code path, run:

```bash
uv run python toy_1d_regression_dblock.py \
    --epochs 1000 \
    --num_blocks 6 \
    --num_train 512 \
    --num_test 512 \
    --observation_noise_std 0.0
```

This trains a tiny DBlock-style denoising chain on a 1D synthetic regression
task and writes:

```text
results/toy_1d_regression/<timestamp>/config.json
results/toy_1d_regression/<timestamp>/metrics.json
results/toy_1d_regression/<timestamp>/blockwise_metrics.csv
results/toy_1d_regression/<timestamp>/blockwise_rmse.png
results/toy_1d_regression/<timestamp>/predictions_by_block.png
results/toy_1d_regression/<timestamp>/test_rmse_by_block_eval_cycles.csv
results/toy_1d_regression/<timestamp>/test_rmse_by_block_eval_cycles.png
results/toy_1d_regression/<timestamp>/test_latent_distance_by_block_eval_cycles.csv
results/toy_1d_regression/<timestamp>/test_latent_distance_by_block_eval_cycles.png
results/toy_1d_regression/<timestamp>/test_latent_distance_by_block_eval_cycles.svg
results/toy_1d_regression/<timestamp>/train_latent_distance_by_block_eval_cycles.csv
results/toy_1d_regression/<timestamp>/train_latent_distance_by_block_eval_cycles.png
results/toy_1d_regression/<timestamp>/train_latent_distance_by_block_eval_cycles.svg
results/toy_1d_regression/<timestamp>/train_loss_by_block_eval_cycles.csv
results/toy_1d_regression/<timestamp>/train_loss_by_block_eval_cycles.png
results/toy_1d_regression/<timestamp>/train_loss_by_block_eval_cycles.svg
results/toy_1d_regression/<timestamp>/prediction_curves_by_block_eval_cycles.csv
results/toy_1d_regression/<timestamp>/prediction_curves_by_block_eval_cycles.png
results/toy_1d_regression/<timestamp>/prediction_uncertainty_by_block.csv
results/toy_1d_regression/<timestamp>/prediction_uncertainty_by_block.png
results/toy_1d_regression/<timestamp>/prediction_uncertainty_by_block.svg
```

Use `--observation_noise_std` to add Gaussian observation noise to both train
and test targets. The old `--target_noise_std` name is kept as an alias.
Use `--function sin_cos_bifurcation` for a bifurcated regression dataset where
training examples include both `sin(x)` and `cos(x)` evaluated at each sampled
training input location. Test examples are independently sampled from either
branch with equal probability. Prediction plots draw both ground-truth branches.
Use `--initial_noise_std 1.0` to make the initial sequential denoising state
standard normal; when omitted, it uses the previous default
`sqrt(1 + sigma[0]^2)` scaling.
Use `--interblock_transition direct_denoised` to feed each block's predicted
clean latent directly into the next block instead of taking an Euler step. The
toy script also supports `direct_denoised_plus_noise` and
`direct_denoised_plus_rescaled_noise` for the matching re-noising ablations.
Use `--block_objective_pattern first_prediction_then_residual` to train block
0 with decoded prediction MSE and every later block with residual-next-latent
loss. Use `--block_objective_pattern alternating_prediction_residual` to train
even-indexed blocks with prediction loss and odd-indexed blocks with residual
loss.
Use `--prediction_uncertainty_samples` to control how many random initial-noise
draws are used for the final per-block Gaussian prediction interval plot; the
default is 50, and 0 skips it. For bifurcated datasets, this plot shows the
individual sampled prediction curves with low alpha instead of Gaussian bands.

## Evaluation

**Baseline (ViT):**

```bash
CKPT_PATH="logs/path-to-last.ckpt"
uv run main.py test cifar100 --model_type vit --ckpt_path $CKPT
```

**DiffusionBlocks:**

```bash
CKPT_PATH="logs/path-to-last.ckpt"
uv run main.py test cifar100 --model_type dblock --ckpt_path $CKPT
```

## Acknowledgement

The implementation of Vision Transformer in [vit.py](./vit.py) is based on [HuggingFace Transformers](https://github.com/huggingface/transformers). And, the implementation of EDM is based on [Stability-AI/generative-models](https://github.com/Stability-AI/generative-models).  
We are grateful for their work.

## Citation

To cite our work, please use the following BibTeX:

```bibtex
@inproceedings{shing2026diffusionblocks,
  title     = {DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation},
  author.   = {Makoto Shing and Masanori Koyama and Takuya Akiba},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=pwVSmK71cS}
}
```
