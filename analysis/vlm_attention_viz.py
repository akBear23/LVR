"""
Visualise where the *vanilla* Qwen-2.5-VL model attends when answering M3-VQA questions.
Baseline companion to lvr_attention_viz.py — no LVR, pure pretrained VLM.

Modes
-----
  --mode answer    (default)
      Average attention from the first <answer_tokens> generated tokens back to
      every image-pad token, averaged over all layers & heads.
      Most direct read of "what the model looks at when producing the answer".

  --mode rollout
      Attention-rollout (Abnar & Zuidema 2020): multiply attention matrices
      layer-by-layer (with residual identity) on the prompt-only forward pass.
      Captures how information flows end-to-end from early to late layers.

  --mode per_token
      One heatmap per generated token — shows how image attention evolves
      across the entire generation.  Useful for spotting when the model
      "re-reads" the image.

  --mode layer
      Split layers into thirds (bottom / mid / top) and compare their
      attention for the answer token.  Reveals early-global vs. late-focused
      attention patterns.

  --mode heads
      Per-head heatmaps for the answer token at a chosen layer.
      Reveals head specialisation (text, entities, edges, regions…).

  --mode all
      Run all five modes for every sample.

Usage (from /mnt/data/lannth/mLAnR/):
    python lvr/analysis/vlm_attention_viz.py \\
        --mode answer \\
        --num_samples 10 \\
        --answer_tokens 5

    # compare layer patterns
    python lvr/analysis/vlm_attention_viz.py --mode layer --num_samples 20

    # head specialisation on last layer
    python lvr/analysis/vlm_attention_viz.py --mode heads --heads_layer -1
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SPATIAL_MERGE = 2
DEFAULT_MODEL  = "Qwen/Qwen2.5-VL-7B-Instruct"


# ─────────────────────────────────────────────────────────────────────────────
# Answer correctness
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.strip().lower()

def is_correct(generated_text: str, gold_answers: list) -> bool:
    pred = normalize(generated_text)
    return any(normalize(gold) in pred for gold in gold_answers)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    print(f"  Loading {model_path} with eager attention …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="eager",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Input preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_inputs(processor, img_path: str, question: str):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text",  "text": question},
    ]}]
    text_fmt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[text_fmt], images=img_inputs, videos=vid_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    image_pad_id     = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_mask       = (inputs.input_ids[0] == image_pad_id)
    num_image_tokens = image_mask.sum().item()

    T, tok_h, tok_w = inputs["image_grid_thw"][0].tolist()
    eff_h = int(tok_h) // SPATIAL_MERGE
    eff_w = int(tok_w) // SPATIAL_MERGE
    assert eff_h * eff_w == num_image_tokens, (
        f"grid mismatch: {eff_h}×{eff_w}={eff_h*eff_w} vs {num_image_tokens}")

    return inputs, image_mask, eff_h, eff_w


# ─────────────────────────────────────────────────────────────────────────────
# Generation with attention output
# ─────────────────────────────────────────────────────────────────────────────

def run_generation(model, processor, inputs, max_new_tokens: int = 64):
    """Run model.generate with output_attentions=True.  Returns (gen_out, tokens, input_len)."""
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_attentions=True,
        )
    gen_tokens = gen_out.sequences[0, input_len:]
    return gen_out, gen_tokens, input_len


# ─────────────────────────────────────────────────────────────────────────────
# Low-level attention extractor
# ─────────────────────────────────────────────────────────────────────────────

def _attn_to_image(step_attns, image_mask, input_len: int,
                   layer_slice: slice = None) -> np.ndarray:
    """
    Average attention from the last query token to image-pad positions.

    step_attns : tuple of (batch, heads, 1, kv_len) tensors, one per layer.
                 This is gen_out.attentions[t] for generation step t.
    Returns a 1-D float32 array of length = num_image_tokens.
    """
    n_layers = len(step_attns)
    if layer_slice is None:
        layer_slice = slice(0, n_layers)

    maps = []
    for layer_attn in step_attns[layer_slice]:
        # step 0 is the prefill pass: (heads, input_len, input_len)
        # step t>0 uses KV-cache:     (heads, 1, input_len+t)
        # [-1] picks the last query row in both cases
        avg      = layer_attn[0].mean(dim=0).float()  # (query, kv_len)
        last     = avg[-1]                             # (kv_len,)
        img_attn = last[:input_len][image_mask]        # (N_img,)
        maps.append(img_attn.cpu().numpy())

    a = np.stack(maps).mean(axis=0)
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 – Answer-token attention
# ─────────────────────────────────────────────────────────────────────────────

def get_answer_attention(gen_out, image_mask, input_len: int,
                         answer_tokens: int = 5):
    """
    Average attention-to-image over the first `answer_tokens` generated tokens,
    all layers & heads.  Returns (avg_1d, per_token_list).
    """
    n_use   = min(answer_tokens, len(gen_out.attentions))
    per_tok = [_attn_to_image(gen_out.attentions[t], image_mask, input_len)
               for t in range(n_use)]
    avg = np.stack(per_tok).mean(axis=0)
    return avg, per_tok


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 – Attention rollout on prefill
# ─────────────────────────────────────────────────────────────────────────────

def get_rollout_attention(model, inputs, image_mask) -> np.ndarray:
    """
    Attention rollout (Abnar & Zuidema 2020).
    Forward pass on the prompt → multiply (A + I) / row-sum across layers.
    Returns normalised 1-D attention over image tokens at the last query position.
    """
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    seq_len = out.attentions[0].shape[-1]
    rollout = torch.eye(seq_len, dtype=torch.float32)

    for layer_attn in out.attentions:
        A = layer_attn[0].mean(dim=0).float().cpu()        # (seq, seq)
        A = A + torch.eye(seq_len, dtype=torch.float32)    # add residual
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = A @ rollout

    img_attn = rollout[-1][image_mask.cpu()].numpy()
    mn, mx   = img_attn.min(), img_attn.max()
    return (img_attn - mn) / (mx - mn + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 – Per-token attention
# ─────────────────────────────────────────────────────────────────────────────

def get_per_token_attention(gen_out, image_mask, input_len: int,
                             max_tokens: int = 32) -> list:
    """One normalised attention-to-image 1D array per generated token."""
    n_steps = min(len(gen_out.attentions), max_tokens)
    return [_attn_to_image(gen_out.attentions[t], image_mask, input_len)
            for t in range(n_steps)]


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4 – Layer-bucket comparison
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_attention(gen_out, image_mask, input_len: int,
                        answer_tokens: int = 3) -> dict:
    """
    Split layers into thirds (bottom / mid / top) and return per-bucket
    average attention-to-image.  Returns {"bottom": 1d, "mid": 1d, "top": 1d}.
    """
    n_layers = len(gen_out.attentions[0])
    third    = max(1, n_layers // 3)
    buckets  = {
        "bottom": slice(0, third),
        "mid":    slice(third, 2 * third),
        "top":    slice(2 * third, n_layers),
    }
    n_steps = min(answer_tokens, len(gen_out.attentions))
    result  = {}
    for name, sl in buckets.items():
        maps = [_attn_to_image(gen_out.attentions[t], image_mask, input_len,
                               layer_slice=sl)
                for t in range(n_steps)]
        result[name] = np.stack(maps).mean(axis=0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Mode 5 – Per-head attention
# ─────────────────────────────────────────────────────────────────────────────

def get_head_attention(gen_out, image_mask, input_len: int,
                        answer_tokens: int = 3, layer: int = -1) -> np.ndarray:
    """
    Per-head attention-to-image from a single `layer`, averaged over the first
    `answer_tokens` generation steps.  Returns (num_heads, N_img) float32.
    """
    n_layers = len(gen_out.attentions[0])
    l_idx    = layer % n_layers
    n_steps  = min(answer_tokens, len(gen_out.attentions))
    n_heads  = gen_out.attentions[0][l_idx].shape[1]
    n_img    = image_mask.sum().item()

    per_head = np.zeros((n_heads, n_img), dtype=np.float32)
    for t in range(n_steps):
        layer_attn = gen_out.attentions[t][l_idx]           # (1, heads, 1, kv_len)
        for h in range(n_heads):
            attn_h = (layer_attn[0, h, 0, :input_len][image_mask]
                      .float().cpu().numpy())
            per_head[h] += attn_h
    per_head /= n_steps

    # normalise each head independently so weak heads still show structure
    mn = per_head.min(axis=1, keepdims=True)
    mx = per_head.max(axis=1, keepdims=True)
    return (per_head - mn) / (mx - mn + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_heatmap(grid1d: np.ndarray, eff_h: int, eff_w: int,
                img_w: int, img_h: int, sigma_rel: float = 0.03) -> np.ndarray:
    grid  = grid1d.reshape(eff_h, eff_w)
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


def _fig_title(sample, gen_text, prefix=""):
    status   = "✓" if is_correct(gen_text, sample["answers"]) else "✗"
    q_short  = sample["question"][:90] + ("…" if len(sample["question"]) > 90 else "")
    pred_str = gen_text.replace("\n", " ")[:120]
    return (f"{prefix}[{status}] [{sample['data_id']}]  Q: {q_short}\n"
            f"Gold: {sample['answers']}  |  Pred: {pred_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure savers
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_fig(attn1d, image, sample, gen_text, eff_h, eff_w,
                     out_path: str, mode_label: str = ""):
    heatmap = _to_heatmap(attn1d, eff_h, eff_w, image.width, image.height)
    overlay = _overlay(image, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original image", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Attention to image tokens\n{mode_label}", fontsize=9)
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

    fig.suptitle(_fig_title(sample, gen_text), fontsize=8, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  summary  → {out_path}")


def save_per_token_fig(per_tok_1d, gen_tokens, processor, image, sample,
                        eff_h, eff_w, gen_text, out_path: str, max_cols: int = 8):
    n    = len(per_tok_1d)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    decoded = [processor.decode([t]) for t in gen_tokens[:n]]

    fig, axes = plt.subplots(rows, cols,
                              figsize=(2.5 * cols, 2.5 * rows + 0.8),
                              squeeze=False)
    for idx, (a1d, tok_str) in enumerate(zip(per_tok_1d, decoded)):
        r, c  = divmod(idx, cols)
        hmap  = _to_heatmap(a1d, eff_h, eff_w, image.width, image.height)
        ovl   = _overlay(image, hmap, alpha=0.55)
        axes[r][c].imshow(ovl)
        axes[r][c].set_title(repr(tok_str)[:14], fontsize=7)
        axes[r][c].axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(_fig_title(sample, gen_text,
                            prefix=f"Per-token attention ({n} tokens)  "),
                 fontsize=8, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  per-token → {out_path}")


def save_layer_fig(layer_maps: dict, image, sample, eff_h, eff_w,
                    gen_text, out_path: str):
    buckets = list(layer_maps.items())
    n_cols  = len(buckets) * 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * len(buckets), 4.5))
    axes = axes.flatten()

    for col, (name, a1d) in enumerate(buckets):
        hmap = _to_heatmap(a1d, eff_h, eff_w, image.width, image.height)
        ovl  = _overlay(image, hmap, alpha=0.55)
        axes[col * 2].imshow(hmap, cmap="jet", vmin=0, vmax=1)
        axes[col * 2].set_title(f"{name} layers\nheatmap", fontsize=8)
        axes[col * 2].axis("off")
        axes[col * 2 + 1].imshow(ovl)
        axes[col * 2 + 1].set_title(f"{name} layers\noverlay", fontsize=8)
        axes[col * 2 + 1].axis("off")

    fig.suptitle(_fig_title(sample, gen_text,
                            prefix="Layer-bucket attention comparison  "),
                 fontsize=8, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  layer    → {out_path}")


def save_heads_fig(head_maps_1d: np.ndarray, image, sample, eff_h, eff_w,
                    gen_text, out_path: str, max_heads: int = 16, layer: int = -1):
    n_heads = min(head_maps_1d.shape[0], max_heads)
    cols    = min(n_heads, 8)
    rows    = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(2.5 * cols, 2.5 * rows + 0.8),
                              squeeze=False)
    for h in range(n_heads):
        r, c = divmod(h, cols)
        hmap = _to_heatmap(head_maps_1d[h], eff_h, eff_w, image.width, image.height)
        ovl  = _overlay(image, hmap, alpha=0.55)
        axes[r][c].imshow(ovl)
        axes[r][c].set_title(f"head {h}", fontsize=7)
        axes[r][c].axis("off")

    for h in range(n_heads, rows * cols):
        r, c = divmod(h, cols)
        axes[r][c].axis("off")

    l_label = f"layer {layer}" if layer >= 0 else "last layer"
    fig.suptitle(_fig_title(sample, gen_text,
                            prefix=f"Per-head attention  ({l_label}, {n_heads} heads)  "),
                 fontsize=8, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  heads    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path",     default=DEFAULT_MODEL,
                        help=f"HF model id or local path  (default: {DEFAULT_MODEL})")
    parser.add_argument("--questions",      default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--image_dir",      default="/mnt/data/lannth/mLAnR/M3-VQA/images")
    parser.add_argument("--output_dir",     default="/mnt/data/lannth/mLAnR/results/vlm_attention_viz")
    parser.add_argument("--mode",           default="answer",
                        choices=["answer", "rollout", "per_token", "layer", "heads", "all"],
                        help=("answer: avg attn from first N answer tokens | "
                              "rollout: layer-by-layer rollout on prompt | "
                              "per_token: one heatmap per generated token | "
                              "layer: bottom/mid/top layer comparison | "
                              "heads: per-head maps for one layer | "
                              "all: run every mode"))
    parser.add_argument("--answer_tokens",  type=int, default=5,
                        help="Tokens to average for 'answer'/'layer'/'heads' modes")
    parser.add_argument("--per_token_max",  type=int, default=32,
                        help="Max tokens to visualise in 'per_token' mode")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_samples",    type=int, default=10)
    parser.add_argument("--start_idx",      type=int, default=0)
    parser.add_argument("--heads_layer",    type=int, default=-1,
                        help="Layer index for 'heads' mode (-1 = last layer)")
    parser.add_argument("--max_heads",      type=int, default=16,
                        help="Max attention heads to show per figure")
    parser.add_argument("--filter",         default="all",
                        choices=["all", "correct_only", "wrong_only"],
                        help="Save results only for correct/wrong/all predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    questions = []
    with open(args.questions) as f:
        for line in f:
            questions.append(json.loads(line))
    subset = questions[args.start_idx : args.start_idx + args.num_samples]
    print(f"Loaded {len(subset)} samples  (idx {args.start_idx}–{args.start_idx+len(subset)-1})")
    print(f"Mode: {args.mode}  |  filter: {args.filter}\n")

    print(f"Loading model from {args.model_path} …")
    model, processor = load_model(args.model_path)
    print("Model loaded.\n")

    modes  = (["answer", "rollout", "per_token", "layer", "heads"]
              if args.mode == "all" else [args.mode])
    stats  = {"correct": 0, "wrong": 0, "skipped": 0}

    for i, sample in enumerate(tqdm(subset, desc="Visualising")):
        img_path = os.path.join(args.image_dir, sample["image_id"])
        print(f"\n[{i+1}/{len(subset)}] {sample['data_id']}  —  {sample['question']}")

        inputs, image_mask, eff_h, eff_w = prepare_inputs(
            processor, img_path, sample["question"])
        input_len = inputs.input_ids.shape[1]
        image     = Image.open(img_path).convert("RGB")
        stem      = os.path.join(args.output_dir, f"{sample['data_id']}_{args.mode}")

        # rollout-only: no generation needed
        if modes == ["rollout"]:
            attn1d = get_rollout_attention(model, inputs, image_mask)
            save_summary_fig(attn1d, image, sample, "[rollout — no generation]",
                              eff_h, eff_w, stem + "_rollout_summary.png",
                              mode_label="rollout (all layers × heads)")
            stats["correct"] += 1   # can't judge correctness without generation
            continue

        # all other modes require generation
        gen_out, gen_tokens, _ = run_generation(
            model, processor, inputs, max_new_tokens=args.max_new_tokens)
        gen_text = processor.decode(gen_tokens, skip_special_tokens=True)

        correct = is_correct(gen_text, sample["answers"])
        print("Ground truth: ", sample["answers"])
        status  = "correct" if correct else "wrong"
        stats[status] += 1
        print(f"  {'✓' if correct else '✗'} {status}  |  pred: {gen_text.strip()!r}")

        # apply filter
        if args.filter == "correct_only" and not correct:
            stats["skipped"] += 1
            continue
        if args.filter == "wrong_only" and correct:
            stats["skipped"] += 1
            continue

        for mode in modes:
            if mode == "answer":
                attn1d, _ = get_answer_attention(
                    gen_out, image_mask, input_len, args.answer_tokens)
                save_summary_fig(
                    attn1d, image, sample, gen_text, eff_h, eff_w,
                    stem + "_answer_summary.png",
                    mode_label=f"avg first {args.answer_tokens} tokens, all layers & heads")

            elif mode == "rollout":
                attn1d = get_rollout_attention(model, inputs, image_mask)
                save_summary_fig(
                    attn1d, image, sample, gen_text, eff_h, eff_w,
                    stem + "_rollout_summary.png",
                    mode_label="attention rollout (Abnar & Zuidema 2020)")

            elif mode == "per_token":
                per_tok = get_per_token_attention(
                    gen_out, image_mask, input_len, args.per_token_max)
                save_per_token_fig(per_tok, gen_tokens, processor, image,
                                    sample, eff_h, eff_w, gen_text,
                                    stem + "_pertok.png")

            elif mode == "layer":
                layer_maps = get_layer_attention(
                    gen_out, image_mask, input_len, args.answer_tokens)
                save_layer_fig(layer_maps, image, sample, eff_h, eff_w,
                                gen_text, stem + "_layer.png")

            elif mode == "heads":
                head_maps = get_head_attention(
                    gen_out, image_mask, input_len,
                    args.answer_tokens, args.heads_layer)
                save_heads_fig(head_maps, image, sample, eff_h, eff_w, gen_text,
                                stem + "_heads.png",
                                max_heads=args.max_heads, layer=args.heads_layer)

    total = stats["correct"] + stats["wrong"]
    print(f"\n{'='*55}")
    print(f"  Model   : {args.model_path}")
    print(f"  Mode    : {args.mode}  |  filter: {args.filter}")
    print(f"  Correct : {stats['correct']}/{total}")
    print(f"  Wrong   : {stats['wrong']}/{total}")
    if stats["skipped"]:
        print(f"  Skipped : {stats['skipped']} (filtered out)")
    print(f"  Output  : {args.output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
