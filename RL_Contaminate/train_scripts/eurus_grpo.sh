set -x

# NOTE: change to your root dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Add repo root and verl source trees so local packages resolve.
export PYTHONPATH="$ROOT:$ROOT/RL_Contaminate/verl:$ROOT/RL_Contaminate/verl/verl:$PYTHONPATH"

# Force using only 4 GPUs on this node (override by setting CUDA_VISIBLE_DEVICES).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

ray stop

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

# Hugging Face caching (avoid per-rank redownload / NFS lock issues)
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Default to the standard HF cache location (user mentioned it's already populated).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Model selection.
export MODEL_REPO_ID="${MODEL_REPO_ID:-PRIME-RL/Eurus-2-7B-PRIME}"

# Model weights are expected to already be present in your HF cache.
# If you want to force offline usage, set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
export MODEL_PATH="$MODEL_REPO_ID"
export DATA_DIR="$ROOT/RL_Contaminate/data"
export EXP_NAME="Eurus-2-7B-PRIME-EURUS"
export CKPT_DIR="$ROOT/RL_Contaminate/checkpoints/$EXP_NAME"

export WANDB_PROJECT="EURUS_GRPO_RLMIA"

# Allow overriding which dataset variant to train on.
# Defaults match the notebook outputs under RL_Contaminate/data/eurus_mia/.
export TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/eurus_mia/eurus_mia_train_1k.parquet}"
export VAL_FILE="${VAL_FILE:-$DATA_DIR/eurus_mia/eurus_mia_test_100.parquet}"

cd "$ROOT/RL_Contaminate/verl/verl/"

# LOGGER="['console']" 
LOGGER="['console','wandb']"

# export WANDB_MODE="offline"

python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=4096 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=${LOGGER} \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=False \
    +trainer.save_by_epoch=True \
    +trainer.save_final_checkpoint=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=0.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.total_epochs=2 $@ 2>&1

# --- Push epoch checkpoints to Hugging Face Hub ---
# This relies on MIX trainer saving epoch checkpoints under:
#   ${trainer.default_local_dir}/epoch_{epoch}_step_{global_step}/actor/
# and saving them in HuggingFace format (save_pretrained + tokenizer).

python3 - <<'PY'
import os
import re
import sys
import subprocess
from pathlib import Path

ckpt_dir = Path(os.environ.get("CKPT_DIR", ""))
if not ckpt_dir:
    print("CKPT_DIR not set; skipping HF push.", file=sys.stderr)
    raise SystemExit(0)

if not ckpt_dir.exists():
    print(f"CKPT_DIR does not exist: {ckpt_dir}. Skipping HF push.", file=sys.stderr)
    raise SystemExit(0)

repo_prefix = os.environ.get("HF_REPO_PREFIX", "talzoomanzoo/eurus_grpo_rlmia_epoch_")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not token:
    # allow huggingface-cli login token
    try:
        from huggingface_hub import HfFolder  # type: ignore

        token = HfFolder.get_token()
    except Exception:
        token = None

if not token:
    raise SystemExit(
        "HF token not found. Set HF_TOKEN (or HUGGINGFACE_TOKEN) or run `huggingface-cli login`."
    )

try:
    from huggingface_hub import HfApi  # type: ignore
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import HfApi  # type: ignore

api = HfApi(token=token)

epoch_re = re.compile(r"^epoch_(\d+)_step_(\d+)$")
epoch_dirs = []
for p in ckpt_dir.iterdir():
    if not p.is_dir():
        continue
    m = epoch_re.match(p.name)
    if not m:
        continue
    epoch_num = int(m.group(1))
    step_num = int(m.group(2))
    actor_dir = p / "actor"
    if actor_dir.is_dir():
        epoch_dirs.append((epoch_num, step_num, actor_dir))

epoch_dirs.sort(key=lambda t: (t[0], t[1]))
if not epoch_dirs:
    print(f"No epoch checkpoints found under {ckpt_dir}", file=sys.stderr)
    raise SystemExit(0)

for epoch_num, step_num, actor_dir in epoch_dirs:
    repo_id = f"{repo_prefix}{epoch_num}"
    print(f"Pushing epoch {epoch_num} (step {step_num}) from {actor_dir} -> {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(actor_dir),
        path_in_repo=".",
        commit_message=f"Upload actor checkpoint (epoch={epoch_num}, step={step_num})",
    )

print("Done pushing epoch checkpoints.")
PY
