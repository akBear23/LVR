"""
Visualise which image regions the LVR tokens attend to / retrieve on M3-VQA samples.

Two modes
---------
  --mode similarity  (default, faithful to h_t=v_t training objective)
      Cosine similarity between last_position_hidden_state h_t and every image
      patch embedding.  Directly mirrors what the LVR loss optimises.
      Uses flash_attention_2 → faster.

  --mode attention
      Self-attention weights from each LVR generation step to image tokens.
      Shows the reasoning trajectory through the image.
      Requires attn_implementation=eager → slower.

Usage (from /mnt/data/lannth/mLAnR/lvr/):
    python analysis/lvr_attention_viz.py \\
        --model_path /mnt/data/lannth/mLAnR/checkpoints/LVR-7B \\
        --mode similarity \\
        --steps 8 --num_samples 10
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from transformers.generation.configuration_utils import GenerationConfig

# ── GenerationConfig patch (same as evaluate_m3vqa.py) ───────────────────────
_orig_from_model_config = GenerationConfig.from_model_config.__func__
def _patched_from_model_config(cls, model_config):
    for attr in ("decoder", "encoder"):
        val = getattr(model_config, attr, None)
        if isinstance(val, dict):
            d = val
            setattr(model_config, attr,
                    type("_DictConfig", (), {"to_dict": lambda self, _d=d: _d})())
    return _orig_from_model_config(cls, model_config)
GenerationConfig.from_model_config = classmethod(_patched_from_model_config)

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from qwen_vl_utils import process_vision_info

SPATIAL_MERGE = 2   # Qwen2.5-VL always uses 2×2 spatial merging


# ─────────────────────────────────────────────────────────────────────────────
# Answer correctness  (same logic as evaluate_m3vqa.py)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower()

def is_correct(generated_text: str, gold_answers: list) -> bool:
    pred = normalize(generated_text)
    return any(normalize(gold) in pred for gold in gold_answers)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path, mode: str = "similarity"):
    """
    mode='similarity' → flash_attention_2  (no weights needed, faster)
    mode='attention'  → eager              (must return attention weights)
    """
    config = AutoConfig.from_pretrained(model_path)
    replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=config.lvr_head)

    attn_impl = "eager" if mode == "attention" else "flash_attention_2"
    print(f"  attn_implementation = {attn_impl}")

    model = QwenWithLVR.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation=attn_impl,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Shared input preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_inputs(processor, img_path, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text",  "text": question},
    ]}]
    text_fmt = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[text_fmt], images=img_inputs, videos=vid_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    image_pad_id     = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_mask       = (inputs.input_ids[0] == image_pad_id)
    num_image_tokens = image_mask.sum().item()

    T, tok_h, tok_w  = inputs["image_grid_thw"][0].tolist()
    eff_h = int(tok_h) // SPATIAL_MERGE
    eff_w = int(tok_w) // SPATIAL_MERGE
    assert eff_h * eff_w == num_image_tokens, (
        f"grid mismatch: {eff_h}×{eff_w}={eff_h*eff_w} vs {num_image_tokens} tokens")

    return inputs, image_mask, eff_h, eff_w


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 – Similarity maps  (faithful to h_t = v_t training objective)
# ─────────────────────────────────────────────────────────────────────────────

def get_lvr_similarity_maps(model, processor, img_path, question, steps,
                             decoding_strategy="steps"):
    """
    For each LVR step t, compute cosine_similarity(h_t, v_img) over all image
    patches.  h_t is last_position_hidden_state captured via a forward hook;
    v_img are the image patch embeddings from the visual encoder.

    This directly mirrors the LVR training loss: MSE(h_{t-1}, v_t).
    """
    inputs, image_mask, eff_h, eff_w = _prepare_inputs(processor, img_path, question)
    input_len    = inputs.input_ids.shape[1]
    lvr_token_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")

    # ── 1. extract image patch embeddings from the visual encoder ─────────────
    with torch.no_grad():
        image_embeds = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        image_embeds = torch.cat(image_embeds, dim=0).float().cpu()  # (N_img, H)

    # ── 2. hook to capture last_position_hidden_state at every generation step ─
    captured = []

    def _hook(module, inp, output):
        if (hasattr(output, "last_position_hidden_state")
                and output.last_position_hidden_state is not None):
            captured.append(output.last_position_hidden_state[0].float().detach().cpu())

    handle = model.register_forward_hook(_hook)

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=512,
            decoding_strategy=decoding_strategy,
            lvr_steps=[steps],
            return_dict_in_generate=True,
        )
    handle.remove()

    generated_tokens = gen_out.sequences[0, input_len:]
    generated_text   = processor.decode(generated_tokens, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)

    lvr_step_indices = (generated_tokens == lvr_token_id).nonzero(as_tuple=True)[0]
    if lvr_step_indices.numel() == 0:
        print("  ⚠  No <|lvr_latent_end|> tokens generated.")
        return [], None, Image.open(img_path).convert("RGB"), generated_text, eff_h, eff_w

    # ── 3. cosine similarity between h_t and every image patch ────────────────
    # captured[step] = last_position_hidden_state at generation step `step`
    # The step that generated <|lvr_latent_end|> token at index s used captured[s] as its
    # QUERY (it was the hidden state passed in from the previous step).
    per_step_grids = []
    for s in lvr_step_indices:
        step_idx = s.item()
        if step_idx >= len(captured):
            break
        h_t  = captured[step_idx]                                     # (H,)
        sims = F.cosine_similarity(h_t.unsqueeze(0), image_embeds, dim=-1)  # (N_img,)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        per_step_grids.append(sims.numpy().reshape(eff_h, eff_w))

    if not per_step_grids:
        return [], None, Image.open(img_path).convert("RGB"), generated_text, eff_h, eff_w

    avg_grid = np.stack(per_step_grids).mean(axis=0)
    return per_step_grids, avg_grid, Image.open(img_path).convert("RGB"), generated_text, eff_h, eff_w


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 – Attention maps  (reasoning trajectory through the image)
# ─────────────────────────────────────────────────────────────────────────────

def get_lvr_attention_maps(model, processor, img_path, question, steps,
                            decoding_strategy="steps"):
    """
    For each LVR step, extract attention weights (averaged over layers & heads)
    from the current LVR query to every image token in the KV-cache.
    Requires attn_implementation='eager'.
    """
    inputs, image_mask, eff_h, eff_w = _prepare_inputs(processor, img_path, question)
    input_len    = inputs.input_ids.shape[1]
    lvr_token_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=512,
            decoding_strategy=decoding_strategy,
            lvr_steps=[steps],
            return_dict_in_generate=True,
            output_attentions=True,
        )

    generated_tokens = gen_out.sequences[0, input_len:]
    generated_text   = processor.decode(generated_tokens, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)

    lvr_step_indices = (generated_tokens == lvr_token_id).nonzero(as_tuple=True)[0]
    if lvr_step_indices.numel() == 0:
        print("  ⚠  No <|lvr_latent_end|> tokens generated.")
        return [], None, Image.open(img_path).convert("RGB"), generated_text, eff_h, eff_w

    def _step_attn(step_idx: int) -> torch.Tensor:
        layer_maps = []
        for layer_attn in gen_out.attentions[step_idx]:
            avg = layer_attn[0].mean(dim=0).squeeze(0).float()  # (seq_so_far,)
            layer_maps.append(avg[:input_len][image_mask])       # (N_img,)
        return torch.stack(layer_maps).mean(dim=0)               # (N_img,)

    per_step_1d    = [_step_attn(s.item()).cpu().numpy() for s in lvr_step_indices]
    per_step_grids = [a.reshape(eff_h, eff_w) for a in per_step_1d]
    avg_grid       = np.stack(per_step_grids).mean(axis=0)

    return per_step_grids, avg_grid, Image.open(img_path).convert("RGB"), generated_text, eff_h, eff_w


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_heatmap(grid: np.ndarray, img_w: int, img_h: int,
                sigma_rel: float = 0.03) -> np.ndarray:
    up    = cv2.resize(grid.astype(np.float32), (img_w, img_h),
                       interpolation=cv2.INTER_LINEAR)
    sigma = max(1, int(min(img_h, img_w) * sigma_rel))
    up    = gaussian_filter(up, sigma=sigma)
    mn, mx = up.min(), up.max()
    return (up - mn) / (mx - mn + 1e-8)


def _overlay(image: Image.Image, heatmap: np.ndarray,
             alpha: float = 0.50, cmap: str = "jet") -> np.ndarray:
    img_arr  = np.array(image.convert("RGB")).astype(np.float32)
    heat_rgb = (plt.get_cmap(cmap)(heatmap)[:, :, :3] * 255).astype(np.float32)
    return ((1 - alpha) * img_arr + alpha * heat_rgb).clip(0, 255).astype(np.uint8)


def _top_region_box(heatmap: np.ndarray, threshold: float = 0.65):
    mask = (heatmap >= threshold).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Summary
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_fig(avg_grid, per_step_grids, image, sample,
                     generated_text, out_path, mode: str = "similarity"):
    heatmap = _to_heatmap(avg_grid, image.width, image.height)
    overlay = _overlay(image, heatmap)

    mode_label = "cosine sim  h_t · v_img" if mode == "similarity" else "attention weights"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original image", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"LVR {mode_label}  ({len(per_step_grids)} steps avg)", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    box = _top_region_box(heatmap)
    if box is not None:
        x1, y1, x2, y2 = box
        axes[2].add_patch(mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="lime", facecolor="none"))
    axes[2].set_title("Overlay  (green = top region)", fontsize=11)
    axes[2].axis("off")

    q_short    = sample["question"][:90] + ("…" if len(sample["question"]) > 90 else "")
    pred_short = generated_text.replace("\n", " ")[:140]
    fig.suptitle(
        f"[{sample['data_id']}]  Q: {q_short}\n"
        f"Gold: {sample['answers']}   |   Pred: {pred_short}",
        fontsize=8, y=1.01
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  summary  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Per-step evolution
# ─────────────────────────────────────────────────────────────────────────────

def save_per_step_fig(per_step_grids, image, sample, out_path,
                      mode: str = "similarity", max_cols: int = 8):
    n    = len(per_step_grids)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(2.5 * cols, 2.5 * rows + 0.6),
                              squeeze=False)

    for idx, grid in enumerate(per_step_grids):
        r, c    = divmod(idx, cols)
        hmap    = _to_heatmap(grid, image.width, image.height)
        overlay = _overlay(image, hmap, alpha=0.55)
        axes[r][c].imshow(overlay)
        axes[r][c].set_title(f"step {idx+1}", fontsize=8)
        axes[r][c].axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    mode_label = "similarity (h_t · v_img)" if mode == "similarity" else "attention"
    fig.suptitle(
        f"[{sample['data_id']}]  Per-step LVR {mode_label} evolution  ({n} steps)",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  per-step → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--questions",   default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--image_dir",   default="/mnt/data/lannth/mLAnR/M3-VQA/images")
    parser.add_argument("--output_dir",  default="/mnt/data/lannth/mLAnR/results/attention_viz")
    parser.add_argument("--steps",       type=int, default=8)
    parser.add_argument("--decoding_strategy", default="steps", choices=["steps", "latent"])
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--start_idx",   type=int, default=0)
    parser.add_argument("--mode",        default="similarity",
                        choices=["similarity", "attention"],
                        help="similarity: cosine sim h_t·v_img (faithful to training); "
                             "attention: self-attention weights (reasoning trajectory)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    questions = []
    with open(args.questions) as f:
        for line in f:
            questions.append(json.loads(line))
    subset = questions[args.start_idx : args.start_idx + args.num_samples]
    print(f"Loaded {len(subset)} samples  (idx {args.start_idx}–{args.start_idx+len(subset)-1})")
    print(f"Mode: {args.mode}\n")

    print(f"Loading model from {args.model_path} …")
    model, processor = load_model(args.model_path, mode=args.mode)
    print("Model loaded.\n")

    # pick the extraction function
    extract_fn = (get_lvr_similarity_maps if args.mode == "similarity"
                  else get_lvr_attention_maps)

    n_correct = 0
    n_wrong   = 0
    n_no_lvr  = 0

    for i, sample in enumerate(tqdm(subset, desc="Visualising")):
        img_path = os.path.join(args.image_dir, sample["image_id"])
        print(f"\n[{i+1}/{len(subset)}] {sample['data_id']}  —  {sample['question'][:60]}…")

        per_step_grids, avg_grid, image, gen_text, eff_h, eff_w = extract_fn(
            model, processor, img_path, sample["question"],
            steps=args.steps, decoding_strategy=args.decoding_strategy,
        )

        if avg_grid is None:
            print("  Skipping (no LVR tokens).")
            n_no_lvr += 1
            continue

        # ── correctness filter ────────────────────────────────────────────────
        correct = is_correct(gen_text, sample["answers"])
        if correct:
            n_correct += 1
            print(f"  ✓ correct  — saving …")
            continue
        else:
            n_wrong += 1
            print(f"  ✗ wrong    — skipping  "
                  f"(gold: {sample['answers']}  |  pred: {gen_text[:80].strip()!r})")
            # continue

        stem = os.path.join(args.output_dir,
                            f"{sample['data_id']}_steps{args.steps:03d}_{args.mode}")

        save_summary_fig(avg_grid, per_step_grids, image, sample, gen_text,
                         stem + "_summary.png", mode=args.mode)

        save_per_step_fig(per_step_grids, image, sample,
                          stem + "_perstep.png", mode=args.mode)

        # np.save(stem + "_avg_grid.npy",      avg_grid)
        # np.save(stem + "_perstep_grids.npy", np.stack(per_step_grids))

    total = n_correct + n_wrong + n_no_lvr
    print(f"\n{'='*50}")
    print(f"  Mode             : {args.mode}")
    print(f"  Correct (saved)  : {n_correct}/{total}")
    print(f"  Wrong (skipped)  : {n_wrong}/{total}")
    print(f"  No LVR (skipped) : {n_no_lvr}/{total}")
    print(f"  Output dir       : {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
