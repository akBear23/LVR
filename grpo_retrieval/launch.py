"""
Stage-2 GRPO training for LVR+Retrieval on M3-VQA.

Usage (single GPU):
    python train.py \
        --model_id /path/to/stage1_checkpoint \
        --data_path /mnt/data/lannth/mLAnR/lvr/grpo_retrieval/m3vqa_train.json \
        --kb_path   /mnt/data/lannth/mLAnR/M3-VQA/encyclopedic_kb_wiki.json \
        --output_dir ./grpo_retrieval_checkpoints \
        --lvr_steps 8 --retrieval_top_k 3 --num_generations 4

Usage (multi-GPU with DeepSpeed):
    deepspeed train.py --deepspeed ../scripts/zero2.json [same args]
"""

import sys
import os
import pathlib

# Must be set before CUDA is initialized (before any torch.cuda call).
# expandable_segments prevents OOM from allocator fragmentation when
# large tensors are allocated/freed repeatedly across per-sample generation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from dataclasses import dataclass, field
from transformers import AutoProcessor, AutoConfig, HfArgumentParser

_GRPO_DIR = os.path.dirname(os.path.abspath(__file__))
_LVR_ROOT = os.path.abspath(os.path.join(_GRPO_DIR, ".."))
# Do NOT put _GRPO_DIR first — it contains launch.py which would shadow src/train package
sys.path.insert(0, _LVR_ROOT)
sys.path.insert(1, os.path.join(_LVR_ROOT, "src"))
from model.qwen_lvr_model import QwenWithLVR
from params import DataArguments, ModelArguments, GRPOArguments

# Import grpo_dataset directly to avoid dataset/__init__.py loading missing modules
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "grpo_dataset", os.path.join(_LVR_ROOT, "src", "dataset", "grpo_dataset.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_grpo_data_module = _mod.make_grpo_data_module
from train.train_utils import safe_save_model_for_hf_trainer, normalize_special_tokens
from train.monkey_patch_forward_lvr_rl import replace_qwen2_5_with_mixed_modality_forward_lvr_rl
from train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb

from retrieval import PassageIndex

# Import our local trainer (not lvr/src/trainer package)
_spec3 = _ilu.spec_from_file_location(
    "grpo_retrieval_trainer", os.path.join(os.path.dirname(__file__), "trainer.py")
)
_mod3 = _ilu.module_from_spec(_spec3)
_spec3.loader.exec_module(_mod3)
RetrievalGRPOTrainer = _mod3.RetrievalGRPOTrainer

from transformers import CLIPModel, CLIPProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Extra args for retrieval
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalArguments:
    kb_path: str = field(
        default="/mnt/data/lannth/mLAnR/M3-VQA/encyclopedic_kb_wiki.json",
        metadata={"help": "Path to M3-VQA encyclopedic KB JSON."},
    )
    clip_model: str = field(
        default="openai/clip-vit-large-patch14",
        metadata={"help": "CLIP model for passage/image encoding."},
    )
    clip_cache_dir: str = field(
        default="/mnt/data/lannth/mLAnR/lvr/grpo_retrieval/.cache",
        metadata={"help": "Directory for caching CLIP passage embeddings."},
    )
    retrieval_model_path: str = field(
        default="intfloat/e5-base-v2",
        metadata={"help": "Dense text encoder for <search> query retrieval (AutoRefine-style)."},
    )
    retrieval_top_k: int = field(default=3, metadata={"help": "Documents retrieved per step."})
    box_threshold: float = field(default=0.65, metadata={"help": "Attention bbox threshold."})
    box_padding:   int   = field(default=10,   metadata={"help": "Bbox padding in pixels."})
    retrieval_reward_weight: float = field(
        default=0.2,
        metadata={"help": "Weight for the retrieval hit bonus reward."},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_model(model, training_args, compute_dtype, device):
    model.visual.to(dtype=compute_dtype, device=device)
    set_requires_grad(model.visual.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(model.visual.merger.parameters(), not training_args.freeze_merger)
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments, RetrievalArguments))
    model_args, data_args, training_args, retrieval_args = parser.parse_args_into_dataclasses()
    training_args.use_liger_loss = False

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model_pth = training_args.checkpoint_name or model_args.model_id

    # ── Build passage index BEFORE loading the 7B model ───────────────────
    # CLIP + E5 are loaded here, embeddings streamed to disk, then both
    # models are moved to CPU so the 7B model can claim their GPU memory.
    print("Loading CLIP model …")
    clip_model = CLIPModel.from_pretrained(
        retrieval_args.clip_model, torch_dtype=torch.float16
    ).to(training_args.device)
    clip_processor = CLIPProcessor.from_pretrained(retrieval_args.clip_model)
    clip_model.eval()

    passage_index = PassageIndex(
        kb_path              = retrieval_args.kb_path,
        clip_model           = clip_model,
        clip_processor       = clip_processor,
        cache_dir            = retrieval_args.clip_cache_dir,
        text_key             = "title",
        retrieval_model_path = retrieval_args.retrieval_model_path,
    )

    # Move retrieval models to CPU after index build; they move back to GPU
    # inside retrieve_by_image / DenseEncoder.encode when called during rollout.
    clip_model.cpu()
    passage_index.dense_encoder.model.cpu()
    torch.cuda.empty_cache()

    # Normalise special tokens so the model can output them
    tokens_to_normalize = {"<|lvr_start|>", "<|lvr_end|>", "<|lvr|>", "<|lvr_latent_end|>"}
    normalize_special_tokens(model_pth, tokens_to_normalize)

    # Patch the forward function for RL (stage 2)
    replace_qwen2_5_with_mixed_modality_forward_lvr_rl()

    config = AutoConfig.from_pretrained(model_pth, trust_remote_code=True)
    model = QwenWithLVR.from_pretrained(
        model_pth, config=config,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
    )

    if model_args.lvr_head:
        model._init_lvr_head(lvr_head_type=model_args.lvr_head_type)

    replace_qwen_2_5_vl_patch_emb()
    configure_model(model, training_args, compute_dtype, training_args.device)

    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_pth)

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset_module = make_grpo_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
    )

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = RetrievalGRPOTrainer(
        # retrieval kwargs
        passage_index            = passage_index,
        lvr_steps                = training_args.lvr_steps,
        top_k                    = retrieval_args.retrieval_top_k,
        box_threshold            = retrieval_args.box_threshold,
        box_padding              = retrieval_args.box_padding,
        retrieval_reward_weight  = retrieval_args.retrieval_reward_weight,
        # standard QwenGRPOTrainer kwargs
        model                    = model,
        ref_model_pth            = model_pth,
        train_dataset            = dataset_module["train_dataset"],
        eval_dataset             = dataset_module["eval_dataset"],
        reward_funcs             = [],      # rewards handled inside _generate_and_score_completions
        processing_class         = processor,
        args                     = training_args,
    )

    resume = bool(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
