"""
Test whether LVR hidden states / CLIP image regions can retrieve relevant KB documents.

Two retrieval modes
-------------------
  lvr_ht   : query = h_t (last_position_hidden_state at each LVR step, mean-pooled)
              passages encoded with LLM text backbone (same hidden space as h_t)

  clip_box : query = CLIP image embedding of the high-attention region (bounding box
             derived from LVR cosine-similarity heatmap)
             passages encoded with CLIP text encoder
             → both query and passages live in CLIP's joint image-text space

Usage (from /mnt/data/lannth/mLAnR/lvr/):
    # LVR h_t mode (original)
    python analysis/lvr_retrieval_eval.py --model_path <ckpt> --steps 8

    # CLIP box mode (single box, original)
    python analysis/lvr_retrieval_eval.py --model_path <ckpt> --steps 8 \\
        --retrieval_mode clip_box --clip_model openai/clip-vit-large-patch14

    # Multi-box: connected components on averaged heatmap (up to 3 entities)
    python analysis/lvr_retrieval_eval.py --model_path <ckpt> --steps 8 \\
        --retrieval_mode clip_multibox --max_boxes 3 --box_fusion max

    # Multi-box: one box per LVR step, NMS-deduped (temporal/semantic entities)
    python analysis/lvr_retrieval_eval.py --model_path <ckpt> --steps 8 \\
        --retrieval_mode clip_stepbox --max_boxes 3 --box_fusion max

    # Encode passages only (useful to pre-cache before eval loop)
    python analysis/lvr_retrieval_eval.py --model_path <ckpt> \\
        --retrieval_mode clip_box --encode_only
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from transformers.generation.configuration_utils import GenerationConfig

# ── GenerationConfig patch ────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=config.lvr_head)
    model = QwenWithLVR.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


# ─────────────────────────────────────────────────────────────────────────────
# Pool building
# ─────────────────────────────────────────────────────────────────────────────

def build_pool(kb, questions, num_distractors=50_000, seed=42):
    """
    Returns
    -------
    passages : list of dicts
        {"pid": str, "url": str, "sec_id": int, "text": str, "is_gold": bool}
    gold_pids_per_question : list of set[str]
        For each question, the set of passage-IDs that are correct answers.
    """
    rng = random.Random(seed)

    # collect all gold (url, sec_id) pairs across the full dataset
    gold_url_set = set()
    gold_pid_set = set()
    for q in questions:
        for url, sec_ids in zip(q.get("evidence_urls", []),
                                q.get("evidence_section_ids", [])):
            gold_url_set.add(url)
            for s in sec_ids:
                gold_pid_set.add(f"{url}|{s}")

    # build passages from ALL sections of gold articles
    passages = []
    pid_set  = set()
    for url in sorted(gold_url_set):
        entry = kb.get(url, {})
        art_title  = entry.get("title", "")
        sec_texts  = entry.get("section_texts", [])
        sec_titles = entry.get("section_titles", [])
        for sec_id, text in enumerate(sec_texts):
            text = text.strip()
            if not text:
                continue
            pid  = f"{url}|{sec_id}"
            stitle = sec_titles[sec_id] if sec_id < len(sec_titles) else ""
            formatted = f"{art_title} – {stitle}\n{text}" if stitle else f"{art_title}\n{text}"
            title_only = f"{art_title} – {stitle}" if stitle else art_title
            passages.append({
                "pid":     pid,
                "url":     url,
                "sec_id":  sec_id,
                "text":    formatted,
                "title":   title_only,
                "is_gold": pid in gold_pid_set,
            })
            pid_set.add(pid)

    print(f"  Gold articles: {len(gold_url_set)}  |  "
          f"sections from gold articles: {len(passages)}  |  "
          f"gold sections: {len(gold_pid_set)}")

    # add random distractor sections from non-gold articles
    if num_distractors > 0:
        non_gold_urls = [u for u in kb if u not in gold_url_set]
        rng.shuffle(non_gold_urls)
        added = 0
        for url in non_gold_urls:
            if added >= num_distractors:
                break
            entry = kb[url]
            art_title  = entry.get("title", "")
            sec_texts  = entry.get("section_texts", [])
            sec_titles = entry.get("section_titles", [])
            for sec_id, text in enumerate(sec_texts):
                if added >= num_distractors:
                    break
                text = text.strip()
                if not text:
                    continue
                pid = f"{url}|{sec_id}"
                if pid in pid_set:
                    continue
                stitle = sec_titles[sec_id] if sec_id < len(sec_titles) else ""
                formatted  = f"{art_title} – {stitle}\n{text}" if stitle else f"{art_title}\n{text}"
                title_only = f"{art_title} – {stitle}" if stitle else art_title
                passages.append({
                    "pid":     pid,
                    "url":     url,
                    "sec_id":  sec_id,
                    "text":    formatted,
                    "title":   title_only,
                    "is_gold": False,
                })
                pid_set.add(pid)
                added += 1
        print(f"  Added {added} distractor sections  |  total pool: {len(passages)}")

    # build per-question gold pid sets (only over passages in our pool)
    gold_pids_per_question = []
    for q in questions:
        pids = set()
        for url, sec_ids in zip(q.get("evidence_urls", []),
                                q.get("evidence_section_ids", [])):
            for s in sec_ids:
                pid = f"{url}|{s}"
                if pid in pid_set:
                    pids.add(pid)
        gold_pids_per_question.append(pids)

    return passages, gold_pids_per_question


# ─────────────────────────────────────────────────────────────────────────────
# Passage encoding
# ─────────────────────────────────────────────────────────────────────────────

def encode_passages(model, processor, passages, batch_size=32, max_length=256,
                    cache_dir=None):
    """
    Encode each passage with mean-pool of the last hidden layer.

    Qwen2.5-VL integrates vision and language in a single transformer
    (model.model is Qwen2_5_VLForConditionalGeneration; model.model.model is
    Qwen2_5_VLModel).  For text-only inputs we call model.model.model directly
    to skip the monkey-patched VL forward and avoid image-processing overhead.

    Returns np.ndarray  (N, H)
    """
    emb_path = None
    idx_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        emb_path = os.path.join(cache_dir, "passage_embeddings.npy")
        idx_path = os.path.join(cache_dir, "passage_index.json")

    if emb_path and os.path.exists(emb_path) and os.path.exists(idx_path):
        print(f"Loading cached passage embeddings from {emb_path}")
        embeddings = np.load(emb_path)
        with open(idx_path) as f:
            cached_pids = json.load(f)
        current_pids = [p["pid"] for p in passages]
        if cached_pids == current_pids:
            print(f"  Cache hit: {embeddings.shape[0]} passages, dim={embeddings.shape[1]}")
            return embeddings
        else:
            print("  Cache PID mismatch — re-encoding.")

    texts = [p["text"] for p in passages]
    all_embeddings = []

    # QwenWithLVR extends Qwen2_5_VLForConditionalGeneration directly, so:
    #   model                       = QwenWithLVR  (monkey-patched forward)
    #   model.model                 = Qwen2_5_VLModel  (vision + language combined)
    #   model.model.language_model  = Qwen2_5_VLTextModel  (pure LLM backbone)
    #
    # The training code runs: self.model.language_model(...) to produce h_t.
    # For document encoding we use the same module so query and passages share
    # the identical hidden space — no CLIP, no projection gap.
    inner = model.model.language_model   # Qwen2_5_VLTextModel
    tok   = processor.tokenizer
    device = next(inner.parameters()).device

    inner.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding passages"):
        batch_texts = texts[i : i + batch_size]
        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            out = inner(
                input_ids=input_ids,
                attention_mask=attn_mask,
                return_dict=True,
            )
        last_hidden = out.last_hidden_state.float()           # (B, L, H)
        mask = attn_mask.unsqueeze(-1).float().to(last_hidden.device)
        pooled = (last_hidden * mask).sum(1) / mask.sum(1)   # (B, H)
        all_embeddings.append(pooled.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)   # (N, H)

    if emb_path:
        np.save(emb_path, embeddings)
        with open(idx_path, "w") as f:
            json.dump([p["pid"] for p in passages], f)
        print(f"Saved embeddings → {emb_path}  shape={embeddings.shape}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# LVR query extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_lvr_hidden_states(model, processor, img_path, question, steps,
                           decoding_strategy="steps"):
    """
    Run LVR generation and return a list of h_t tensors (one per <|lvr|> step)
    plus the generated text.
    h_t = last_position_hidden_state captured via forward hook at each step.
    """
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text",  "text": question},
    ]}]
    text_fmt = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[text_fmt], images=img_inputs, videos=vid_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    input_len    = inputs.input_ids.shape[1]
    lvr_token_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")

    # hook: capture last_position_hidden_state at every forward call
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

    h_list = []
    for s in lvr_step_indices:
        idx = s.item()
        if idx < len(captured):
            h_list.append(captured[idx])   # (H,)

    return h_list, generated_text


def aggregate_query(h_list, strategy="mean"):
    """Combine per-step h_t tensors into a single (H,) query vector."""
    if not h_list:
        return None
    stacked = torch.stack(h_list, dim=0)   # (T, H)
    if strategy == "mean":
        return stacked.mean(dim=0)
    elif strategy == "last":
        return stacked[-1]
    elif strategy == "first":
        return stacked[0]
    elif strategy == "max":
        return stacked.max(dim=0).values
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# CLIP-box retrieval
# ─────────────────────────────────────────────────────────────────────────────

SPATIAL_MERGE = 2

def _top_region_box(heatmap: np.ndarray, threshold: float = 0.65):
    """
    Return (x1, y1, x2, y2) bounding box of all pixels >= threshold,
    or None if no such pixels exist.  Same logic as lvr_attention_viz.py.
    """
    mask = (heatmap >= threshold).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def load_clip_model(clip_model_name="openai/clip-vit-large-patch14"):
    from transformers import CLIPModel, CLIPProcessor
    print(f"Loading CLIP model: {clip_model_name} …")
    clip_model = CLIPModel.from_pretrained(clip_model_name,
                                           torch_dtype=torch.float16).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()
    print("CLIP loaded.")
    return clip_model, clip_processor


def encode_passages_clip(clip_model, clip_processor, passages,
                          batch_size=256, cache_dir=None, text_key="title"):
    """
    Encode passage texts with CLIP's text encoder.
    text_key selects which passage field to encode: "title" (article – section name,
    fits comfortably in CLIP's 77-token limit) or "text" (full section, truncated).
    Returns np.ndarray (N, D_clip).
    """
    emb_path = idx_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        emb_path = os.path.join(cache_dir, f"passage_embeddings_clip_{text_key}.npy")
        idx_path = os.path.join(cache_dir, f"passage_index_clip_{text_key}.json")

    if emb_path and os.path.exists(emb_path) and os.path.exists(idx_path):
        print(f"Loading cached CLIP passage embeddings ({text_key}) from {emb_path}")
        embeddings = np.load(emb_path)
        with open(idx_path) as f:
            cached_pids = json.load(f)
        if cached_pids == [p["pid"] for p in passages]:
            print(f"  Cache hit: {embeddings.shape}")
            return embeddings
        print("  Cache PID mismatch — re-encoding.")

    device = next(clip_model.parameters()).device
    all_embs = []
    desc = f"Encoding passages (CLIP text, key={text_key})"
    for i in tqdm(range(0, len(passages), batch_size), desc=desc):
        batch = [p[text_key] for p in passages[i : i + batch_size]]
        enc = clip_processor(
            text=batch, return_tensors="pt",
            padding=True, truncation=True, max_length=77,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            feats = clip_model.get_text_features(**enc)   # (B, D)
        all_embs.append(feats.float().cpu().numpy())

    embeddings = np.concatenate(all_embs, axis=0)   # (N, D)

    if emb_path:
        np.save(emb_path, embeddings)
        with open(idx_path, "w") as f:
            json.dump([p["pid"] for p in passages], f)
        print(f"Saved CLIP passage embeddings → {emb_path}  shape={embeddings.shape}")

    return embeddings


def get_box_and_clip_query(model, processor, clip_model, clip_processor,
                            img_path, question, steps,
                            decoding_strategy="steps",
                            box_threshold=0.65, box_padding=10):
    """
    1. Run LVR generation, build per-step cosine-similarity heatmaps.
    2. Average heatmaps → bounding box of high-attention region.
    3. Crop image to that box (+ padding).
    4. Encode the crop with CLIP image encoder.

    Returns (clip_embedding: np.ndarray shape (D,), generated_text, box_xyxy or None)
    """
    image = Image.open(img_path).convert("RGB")
    W, H = image.size

    # ── 1. Build inputs ───────────────────────────────────────────────────────
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text",  "text": question},
    ]}]
    text_fmt = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[text_fmt], images=img_inputs, videos=vid_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    input_len    = inputs.input_ids.shape[1]
    lvr_token_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")

    # ── 2. Extract image patch embeddings (in LLM space, used for sim maps) ──
    with torch.no_grad():
        image_embeds_list = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
    image_embeds = torch.cat(image_embeds_list, dim=0).float().cpu()  # (N_patches, H)

    thw    = inputs["image_grid_thw"][0]
    eff_h  = int(thw[1].item()) // SPATIAL_MERGE
    eff_w  = int(thw[2].item()) // SPATIAL_MERGE

    # ── 3. Generate with hook ─────────────────────────────────────────────────
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

    if len(lvr_step_indices) == 0:
        return None, generated_text, None

    # ── 4. Build per-step similarity grids and average ────────────────────────
    per_step_grids = []
    for s in lvr_step_indices:
        idx = s.item()
        if idx >= len(captured):
            continue
        h_t  = captured[idx]   # (H,)
        sims = F.cosine_similarity(h_t.unsqueeze(0), image_embeds, dim=-1)  # (N,)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        per_step_grids.append(sims.numpy().reshape(eff_h, eff_w))

    if not per_step_grids:
        return None, generated_text, None

    avg_grid = np.mean(per_step_grids, axis=0)   # (eff_h, eff_w)

    # ── 5. Upsample to image pixel space ─────────────────────────────────────
    heatmap = np.array(
        Image.fromarray((avg_grid * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # ── 6. Bounding box → crop ────────────────────────────────────────────────
    box = _top_region_box(heatmap, threshold=box_threshold)
    if box is None:
        return None, generated_text, None

    x1, y1, x2, y2 = box
    # add padding and clamp to image bounds
    x1 = max(0, x1 - box_padding)
    y1 = max(0, y1 - box_padding)
    x2 = min(W, x2 + box_padding)
    y2 = min(H, y2 + box_padding)

    # guard against degenerate crops
    if x2 <= x1 or y2 <= y1:
        return None, generated_text, None

    crop = image.crop((x1, y1, x2, y2))

    # ── 7. Encode crop with CLIP ──────────────────────────────────────────────
    device = next(clip_model.parameters()).device
    clip_inp = clip_processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_feat = clip_model.get_image_features(**clip_inp)   # (1, D)
    clip_feat = clip_feat.float().cpu().numpy().squeeze(0)       # (D,)

    return clip_feat, generated_text, (x1, y1, x2, y2)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-box CLIP retrieval  (new: multi-entity support)
# ─────────────────────────────────────────────────────────────────────────────
#
# Two new retrieval modes address the case where a question references multiple
# distinct visual entities (e.g. "What is the relationship between the person
# on the left and the monument behind them?"):
#
#   clip_multibox  – connected components on the AVERAGED heatmap finds distinct
#                    spatial blobs; each blob → one CLIP embedding (spatial view)
#
#   clip_stepbox   – each LVR step attends to a potentially different entity;
#                    one box per step, NMS-deduplicated (temporal/semantic view)
#
# Both produce a list of CLIP embeddings fed to retrieve_multi_query(), which
# fuses per-box similarities with max (OR-style) or mean pooling.
# ─────────────────────────────────────────────────────────────────────────────

def _save_box_viz(img_path, boxes, out_path, question="", mode="clip_box"):
    """
    Draw bounding boxes on the original image and save as JPEG.

    boxes : list of (x1, y1, x2, y2) in pixel space (already padded).
    Each box gets a distinct colour; the mode label and truncated question
    are stamped at the top-left corner.
    """
    from PIL import ImageDraw
    BOX_COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
    LINE_W = 3

    image = Image.open(img_path).convert("RGB")
    vis   = image.copy()
    draw  = ImageDraw.Draw(vis)

    for bi, box in enumerate(boxes):
        color = BOX_COLORS[bi % len(BOX_COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_W)
        label = f"box{bi}"
        # white background behind label for readability
        tw, th = (len(label) * 7 + 4), 14
        draw.rectangle([x1 + LINE_W, y1 + LINE_W, x1 + LINE_W + tw, y1 + LINE_W + th],
                       fill="black")
        draw.text((x1 + LINE_W + 2, y1 + LINE_W + 1), label, fill=color)

    # stamp mode + question at top of image
    header = f"[{mode}] {question[:90]}"
    draw.rectangle([0, 0, vis.width, 18], fill=(0, 0, 0, 180))
    draw.text((4, 2), header, fill="white")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vis.save(out_path, quality=92)


def _connected_component_boxes(heatmap: np.ndarray, threshold: float = 0.65,
                                max_boxes: int = 3, min_area: int = 16):
    """
    Find distinct high-attention regions via connected components on the heatmap.
    Returns list of (x1, y1, x2, y2) tuples sorted by region area descending,
    capped at max_boxes; regions with fewer than min_area pixels are skipped.
    """
    from scipy import ndimage as ndi
    mask = (heatmap >= threshold).astype(np.uint8)
    labeled, n_components = ndi.label(mask)
    boxes = []
    for comp_id in range(1, n_components + 1):
        ys, xs = np.where(labeled == comp_id)
        area = len(xs)
        if area < min_area:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), area))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in boxes[:max_boxes]]


def _iou(box_a, box_b):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(box_a[0], box_b[0]);  iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2]);  iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (a_area + b_area - inter + 1e-8)


def _nms_boxes(boxes, iou_threshold=0.5):
    """Remove overlapping boxes: keep first, suppress any later box with IoU > threshold."""
    kept = []
    for box in boxes:
        if not any(_iou(box, k) > iou_threshold for k in kept):
            kept.append(box)
    return kept


def _encode_crop_clip(image, box, clip_model, clip_processor, box_padding=10):
    """
    Crop image to box (+ padding), encode with CLIP image encoder.
    Returns (D,) float32 numpy array, or None for degenerate crops.
    """
    W, H = image.size
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - box_padding);  y1 = max(0, y1 - box_padding)
    x2 = min(W, x2 + box_padding);  y2 = min(H, y2 + box_padding)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image.crop((x1, y1, x2, y2))
    device = next(clip_model.parameters()).device
    clip_inp = clip_processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = clip_model.get_image_features(**clip_inp)   # (1, D)
    return feat.float().cpu().numpy().squeeze(0)           # (D,)


def _run_lvr_and_build_grids(model, processor, img_path, question, steps,
                               decoding_strategy):
    """
    Shared helper: run LVR generation and build per-step cosine-similarity grids.
    Returns (per_step_grids, pil_image, generated_text).
    per_step_grids is a list of (eff_h, eff_w) float32 arrays (one per LVR step).
    """
    image = Image.open(img_path).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text",  "text": question},
    ]}]
    text_fmt = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(text=[text_fmt], images=img_inputs, videos=vid_inputs,
                       padding=True, return_tensors="pt").to("cuda")

    input_len    = inputs.input_ids.shape[1]
    lvr_token_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")

    with torch.no_grad():
        image_embeds_list = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
    image_embeds = torch.cat(image_embeds_list, dim=0).float().cpu()

    thw   = inputs["image_grid_thw"][0]
    eff_h = int(thw[1].item()) // SPATIAL_MERGE
    eff_w = int(thw[2].item()) // SPATIAL_MERGE

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

    per_step_grids = []
    for s in lvr_step_indices:
        idx = s.item()
        if idx >= len(captured):
            continue
        h_t  = captured[idx]
        sims = F.cosine_similarity(h_t.unsqueeze(0), image_embeds, dim=-1)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        per_step_grids.append(sims.numpy().reshape(eff_h, eff_w))

    return per_step_grids, image, generated_text


def get_multi_box_clip_query(model, processor, clip_model, clip_processor,
                              img_path, question, steps,
                              decoding_strategy="steps",
                              box_threshold=0.65, box_padding=10,
                              max_boxes=3, min_area=16):
    """
    Multi-entity CLIP retrieval via connected components on the averaged heatmap.

    Strategy
    --------
    The averaged per-step similarity grid (same as the single-box mode) is fed to
    connected-component analysis instead of a single bounding-box extraction.
    Each distinct above-threshold blob likely corresponds to a separate visual
    entity referenced by the question.

    Returns
    -------
    clip_feats     : list[np.ndarray]   – (D,) vector per found region (may be [])
    generated_text : str
    boxes          : list[tuple]        – padded pixel-space (x1,y1,x2,y2) per region
    """
    per_step_grids, image, generated_text = _run_lvr_and_build_grids(
        model, processor, img_path, question, steps, decoding_strategy)

    if not per_step_grids:
        return [], generated_text, []

    W, H = image.size
    avg_grid = np.mean(per_step_grids, axis=0)
    heatmap = np.array(
        Image.fromarray((avg_grid * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    raw_boxes = _connected_component_boxes(
        heatmap, threshold=box_threshold, max_boxes=max_boxes, min_area=min_area)

    clip_feats, final_boxes = [], []
    for box in raw_boxes:
        feat = _encode_crop_clip(image, box, clip_model, clip_processor, box_padding)
        if feat is not None:
            clip_feats.append(feat)
            final_boxes.append(box)

    return clip_feats, generated_text, final_boxes


def get_per_step_clip_query(model, processor, clip_model, clip_processor,
                             img_path, question, steps,
                             decoding_strategy="steps",
                             box_threshold=0.65, box_padding=10,
                             max_boxes=3, iou_threshold=0.5):
    """
    Multi-entity CLIP retrieval using per-LVR-step bounding boxes.

    Strategy
    --------
    Different LVR steps attend to semantically different visual entities.
    A bounding box is extracted from each step's individual heatmap; overlapping
    boxes are deduplicated via NMS; up to max_boxes unique regions are kept.

    Returns
    -------
    clip_feats     : list[np.ndarray]   – (D,) vector per unique region (may be [])
    generated_text : str
    boxes          : list[tuple]        – padded pixel-space (x1,y1,x2,y2) per region
    """
    per_step_grids, image, generated_text = _run_lvr_and_build_grids(
        model, processor, img_path, question, steps, decoding_strategy)

    if not per_step_grids:
        return [], generated_text, []

    W, H = image.size
    per_step_boxes = []
    for grid in per_step_grids:
        heatmap = np.array(
            Image.fromarray((grid * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        ).astype(np.float32) / 255.0
        box = _top_region_box(heatmap, threshold=box_threshold)
        if box is not None:
            per_step_boxes.append(box)

    unique_boxes = _nms_boxes(per_step_boxes, iou_threshold=iou_threshold)[:max_boxes]

    clip_feats, final_boxes = [], []
    for box in unique_boxes:
        feat = _encode_crop_clip(image, box, clip_model, clip_processor, box_padding)
        if feat is not None:
            clip_feats.append(feat)
            final_boxes.append(box)

    return clip_feats, generated_text, final_boxes


def retrieve_multi_query(query_vecs: list, passage_embeddings: np.ndarray,
                          top_k: int = 100, fusion: str = "max"):
    """
    Cosine-similarity retrieval with multiple query vectors (multi-entity).

    fusion="max"  : score = max_i sim(passage, query_i)
                    correct when ANY entity match is sufficient (OR-style multi-hop)
    fusion="mean" : score = mean_i sim(passage, query_i)
                    conservative; useful when all entities should contribute
    fusion="sum"  : score = sum_i sim(passage, query_i)
                    equivalent to mean for ranking but amplifies multi-entity hits

    Returns top_idx (sorted desc), top_sims.
    """
    if not query_vecs:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    P = passage_embeddings / (np.linalg.norm(passage_embeddings, axis=1, keepdims=True) + 1e-8)
    sims_per_query = []
    for q in query_vecs:
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        sims_per_query.append(P @ q_norm)              # (N,)
    sims_matrix = np.stack(sims_per_query, axis=0)     # (num_boxes, N)

    if fusion == "max":
        sims = sims_matrix.max(axis=0)
    elif fusion == "mean":
        sims = sims_matrix.mean(axis=0)
    elif fusion == "sum":
        sims = sims_matrix.sum(axis=0)
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion}")

    top_idx = np.argpartition(sims, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    return top_idx, sims[top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query_vec: np.ndarray, passage_embeddings: np.ndarray,
             top_k: int = 100):
    """
    Compute cosine similarity and return top-k passage indices (sorted desc).
    Both query_vec and passage_embeddings should be float32.
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    P = passage_embeddings / (np.linalg.norm(passage_embeddings, axis=1, keepdims=True) + 1e-8)
    sims = P @ q                              # (N,)
    top_idx = np.argpartition(sims, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    return top_idx, sims[top_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(ranks_list, K_list=(1, 5, 10, 20, 50, 100)):
    """
    ranks_list : list of int  – 1-indexed rank of first gold hit per question
                                (0 if not found in top-100)
    """
    n = len(ranks_list)
    metrics = {}
    for k in K_list:
        metrics[f"R@{k}"] = sum(1 for r in ranks_list if 0 < r <= k) / n
    metrics["MRR"] = sum((1 / r) if r > 0 else 0 for r in ranks_list) / n
    return metrics


def print_metrics(metrics, prefix=""):
    line = "  ".join(f"{k}={v*100:.1f}" for k, v in metrics.items())
    print(f"{prefix}{line}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval_eval(model, processor, questions, passages, passage_embeddings,
                       gold_pids_per_question, image_dir, steps, decoding_strategy,
                       aggregation_strategies, top_k, out_path, debug_samples=3,
                       retrieval_mode="lvr_ht",
                       clip_model=None, clip_processor=None,
                       box_threshold=0.65, box_padding=10,
                       box_fusion="max", max_boxes=3, min_area=16, iou_threshold=0.5,
                       save_boxes=False, save_boxes_n=0, viz_dir=None):
    """
    For each question: retrieve relevant passages and compute Recall@K / MRR.

    retrieval_mode="lvr_ht"      : query = aggregated h_t hidden states
    retrieval_mode="clip_box"    : query = CLIP image embedding of the single
                                   high-attention bounding box (averaged heatmap)
    retrieval_mode="clip_multibox": up to max_boxes CLIP embeddings from distinct
                                   spatial blobs (connected components on avg heatmap)
    retrieval_mode="clip_stepbox" : up to max_boxes CLIP embeddings, one per LVR
                                   step after NMS deduplication
    """
    pid_to_idx = {p["pid"]: i for i, p in enumerate(passages)}
    K_list     = [1, 5, 10, 20, 50, min(100, top_k)]

    # pre-normalise passage embeddings once for fast cosine similarity
    p_norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True) + 1e-8
    passage_embeddings_normed = passage_embeddings / p_norms

    # per-strategy, per-hop accumulators
    results_per_strategy = {s: [] for s in aggregation_strategies}
    hop_results          = {s: {} for s in aggregation_strategies}

    is_clip_mode       = (retrieval_mode == "clip_box")
    is_multi_clip_mode = retrieval_mode in ("clip_multibox", "clip_stepbox")

    # single-/multi-box CLIP modes use a fixed strategy name (no h_t aggregation loop)
    if is_clip_mode:
        aggregation_strategies = ["clip_box"]
        results_per_strategy   = {"clip_box": []}
        hop_results            = {"clip_box": {}}
    elif is_multi_clip_mode:
        aggregation_strategies = [retrieval_mode]
        results_per_strategy   = {retrieval_mode: []}
        hop_results            = {retrieval_mode: {}}

    skipped   = 0
    n_empty_q = 0   # no query (empty h_list or no box found)
    n_empty_g = 0   # no gold pids in pool
    per_question_records = []

    for qi, (q, gold_pids) in enumerate(
            tqdm(zip(questions, gold_pids_per_question), total=len(questions),
                 desc=f"Retrieval eval [{retrieval_mode}]")):

        img_path = os.path.join(image_dir, q["image_id"])

        # ── get query vector ──────────────────────────────────────────────────
        query_vecs = []   # used by multi-box modes
        if is_clip_mode:
            clip_feat, gen_text, box = get_box_and_clip_query(
                model, processor, clip_model, clip_processor,
                img_path, q["question"], steps=steps,
                decoding_strategy=decoding_strategy,
                box_threshold=box_threshold, box_padding=box_padding,
            )
            query_vec  = clip_feat          # np.ndarray (D,) or None
            h_list     = [clip_feat] if clip_feat is not None else []
        elif is_multi_clip_mode:
            if retrieval_mode == "clip_multibox":
                query_vecs, gen_text, boxes = get_multi_box_clip_query(
                    model, processor, clip_model, clip_processor,
                    img_path, q["question"], steps=steps,
                    decoding_strategy=decoding_strategy,
                    box_threshold=box_threshold, box_padding=box_padding,
                    max_boxes=max_boxes, min_area=min_area,
                )
            else:   # clip_stepbox
                query_vecs, gen_text, boxes = get_per_step_clip_query(
                    model, processor, clip_model, clip_processor,
                    img_path, q["question"], steps=steps,
                    decoding_strategy=decoding_strategy,
                    box_threshold=box_threshold, box_padding=box_padding,
                    max_boxes=max_boxes, iou_threshold=iou_threshold,
                )
            h_list    = query_vecs   # non-empty → valid query
            query_vec = None
            box       = boxes[0] if boxes else None   # first box for compat
        else:
            h_list, gen_text = get_lvr_hidden_states(
                model, processor, img_path, q["question"],
                steps=steps, decoding_strategy=decoding_strategy,
            )
            query_vec = None   # computed per-strategy below
            box       = None

        # ── save box visualisation ────────────────────────────────────────────
        if save_boxes and (save_boxes_n == 0 or qi < save_boxes_n):
            viz_boxes = None
            if is_clip_mode and box is not None:
                viz_boxes = [box]
            elif is_multi_clip_mode and boxes:
                viz_boxes = boxes
            if viz_boxes:
                data_id  = str(q.get("data_id", qi))
                safe_id  = data_id.replace("/", "_").replace("\\", "_")
                out_name = f"q{qi:05d}_{safe_id}.jpg"
                out_path_viz = os.path.join(viz_dir or "box_viz", retrieval_mode, out_name)
                _save_box_viz(img_path, viz_boxes, out_path_viz,
                              question=q["question"], mode=retrieval_mode)
        # ─────────────────────────────────────────────────────────────────────

        # ── debug: first N questions ──────────────────────────────────────────
        if qi < debug_samples:
            print(f"\n{'─'*60}")
            print(f"[DEBUG Q{qi}] mode={retrieval_mode}  data_id={q.get('data_id')}"
                  f"  hop={q.get('question_hop')}")
            print(f"  question : {q['question'][:120]}")
            if is_clip_mode:
                print(f"  box      : {box}")
                print(f"  clip_feat: {'ok' if clip_feat is not None else 'None'}"
                      + (f"  norm={np.linalg.norm(clip_feat):.3f}" if clip_feat is not None else ""))
            elif is_multi_clip_mode:
                print(f"  boxes ({len(boxes)}):")
                for bi, bx in enumerate(boxes):
                    print(f"    box[{bi}]: {bx}"
                          + (f"  norm={np.linalg.norm(query_vecs[bi]):.3f}" if bi < len(query_vecs) else ""))
            else:
                print(f"  h_list   : {len(h_list)} LVR steps captured")
                for si, h in enumerate(h_list[:4]):
                    print(f"    h[{si}] norm={h.norm().item():.3f}"
                          f"  mean={h.mean().item():.4f}")
            print(f"  gold_pids: {len(gold_pids)}")
            for gpid in list(gold_pids)[:3]:
                in_pool = gpid in pid_to_idx
                print(f"    '{gpid}'  in_pool={in_pool}")
                if in_pool:
                    g_idx = pid_to_idx[gpid]
                    g_emb = passage_embeddings[g_idx]
                    if is_clip_mode:
                        q_probe = query_vec
                    elif is_multi_clip_mode:
                        q_probe = query_vecs[0] if query_vecs else None
                    else:
                        q_probe = aggregate_query(h_list, "mean").numpy() if h_list else None
                    if q_probe is not None:
                        qn = q_probe / (np.linalg.norm(q_probe) + 1e-8)
                        gn = g_emb   / (np.linalg.norm(g_emb)   + 1e-8)
                        print(f"      cosine_sim(query[0], gold)={float(gn @ qn):.4f}"
                              f"  text: {passages[g_idx]['text'][:70]}")
            if h_list and gold_pids:
                if is_multi_clip_mode:
                    top_idx_d, top_sims_d = retrieve_multi_query(
                        query_vecs, passage_embeddings, top_k=5, fusion=box_fusion)
                elif is_clip_mode:
                    top_idx_d, top_sims_d = retrieve(query_vec, passage_embeddings, top_k=5)
                else:
                    q_probe = aggregate_query(h_list, "mean").numpy()
                    top_idx_d, top_sims_d = retrieve(q_probe, passage_embeddings, top_k=5)
                print(f"  top-5 retrieved:")
                for rp, (tidx, tsim) in enumerate(zip(top_idx_d, top_sims_d), 1):
                    is_gold = passages[tidx]["pid"] in gold_pids
                    print(f"    {rp}. sim={tsim:.4f}  gold={is_gold}"
                          f"  {passages[tidx]['text'][:70]}")
                if not is_multi_clip_mode:
                    q_probe = (query_vec if is_clip_mode
                               else aggregate_query(h_list, "mean").numpy())
                    qn = q_probe / (np.linalg.norm(q_probe) + 1e-8)
                    all_sims = passage_embeddings_normed @ qn
                    print(f"  sim stats: min={all_sims.min():.4f}  max={all_sims.max():.4f}"
                          f"  mean={all_sims.mean():.4f}  std={all_sims.std():.4f}")
            print(f"  gen_text : {gen_text[:200]}")
        # ──────────────────────────────────────────────────────────────────────

        if not h_list:    # covers both empty h_list and clip_feat=None
            n_empty_q += 1
            skipped   += 1
            for s in aggregation_strategies:
                results_per_strategy[s].append(0)
            continue

        if not gold_pids:
            n_empty_g += 1
            skipped   += 1
            for s in aggregation_strategies:
                results_per_strategy[s].append(0)
            continue

        record = {"data_id": q["data_id"], "hop": q.get("question_hop", -1)}
        hop    = q.get("question_hop", -1)

        for strategy in aggregation_strategies:
            if is_clip_mode:
                top_idx, top_sims = retrieve(query_vec, passage_embeddings, top_k=top_k)
            elif is_multi_clip_mode:
                top_idx, top_sims = retrieve_multi_query(
                    query_vecs, passage_embeddings, top_k=top_k, fusion=box_fusion)
            else:
                q_vec = aggregate_query(h_list, strategy=strategy).numpy()
                top_idx, top_sims = retrieve(q_vec, passage_embeddings, top_k=top_k)

            rank = 0
            for rank_pos, idx in enumerate(top_idx, start=1):
                if passages[idx]["pid"] in gold_pids:
                    rank = rank_pos
                    break

            results_per_strategy[strategy].append(rank)
            record[f"rank_{strategy}"] = rank

            if hop not in hop_results[strategy]:
                hop_results[strategy][hop] = []
            hop_results[strategy][hop].append(rank)

        per_question_records.append(record)

    print(f"\n[skip breakdown] empty_query={n_empty_q}  empty_gold={n_empty_g}"
          f"  total_skipped={skipped}")

    # ── save per-question results ──────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(per_question_records, f, indent=2)
    print(f"\nSaved per-question results → {out_path}")

    # ── print metrics ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Pool size: {len(passages)} passages  |  "
          f"Evaluated: {len(questions)-skipped}/{len(questions)}  |  "
          f"LVR steps: {steps}  |  decoding: {decoding_strategy}")
    print(f"{'='*70}\n")

    summary = {}
    for strategy in aggregation_strategies:
        ranks = results_per_strategy[strategy]
        m     = compute_metrics(ranks, K_list)
        summary[strategy] = {"overall": m, "by_hop": {}}
        print(f"── aggregation = {strategy} ──")
        print_metrics(m, prefix="  Overall:  ")

        for hop in sorted(hop_results[strategy].keys()):
            hm = compute_metrics(hop_results[strategy][hop], K_list)
            summary[strategy]["by_hop"][str(hop)] = hm
            print_metrics(hm, prefix=f"  Hop {hop}:    ")
        print()

    # save summary
    stem = out_path.replace(".json", "_summary.json")
    with open(stem, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {stem}")
    print("="*70)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--kb_path",       default="/mnt/data/lannth/mLAnR/M3-VQA/encyclopedic_kb_wiki.json")
    parser.add_argument("--questions",     default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--image_dir",     default="/mnt/data/lannth/mLAnR/M3-VQA/images")
    parser.add_argument("--output_dir",    default="/mnt/data/lannth/mLAnR/results/retrieval_eval")
    parser.add_argument("--steps",         type=int, default=8)
    parser.add_argument("--decoding_strategy", default="steps", choices=["steps", "latent"])
    parser.add_argument("--num_samples",   type=int, default=None,
                        help="Evaluate on first N questions (None = all)")
    parser.add_argument("--num_distractors", type=int, default=50_000,
                        help="Non-gold sections to add to the pool")
    parser.add_argument("--encode_batch",  type=int, default=32)
    parser.add_argument("--max_passage_len", type=int, default=256,
                        help="Max tokens per passage during encoding")
    parser.add_argument("--top_k",         type=int, default=100,
                        help="How many passages to retrieve per query")
    parser.add_argument("--aggregation",   nargs="+",
                        default=["mean", "last", "first"],
                        choices=["mean", "last", "first", "max"],
                        help="h_t aggregation strategies (lvr_ht mode only)")
    parser.add_argument("--encode_only",   action="store_true",
                        help="Only build and encode the passage pool, then exit")
    parser.add_argument("--debug_samples", type=int, default=3,
                        help="Print verbose diagnostics for first N questions")
    # ── CLIP-box mode ──────────────────────────────────────────────────────────
    parser.add_argument("--retrieval_mode", default="lvr_ht",
                        choices=["lvr_ht", "clip_box", "clip_multibox", "clip_stepbox"],
                        help=("lvr_ht: use h_t hidden states; "
                              "clip_box: single CLIP box from averaged heatmap; "
                              "clip_multibox: multiple CLIP boxes via connected components; "
                              "clip_stepbox: one CLIP box per LVR step (NMS-deduped)"))
    parser.add_argument("--clip_model",    default="openai/clip-vit-large-patch14",
                        help="HuggingFace CLIP model name (clip_* modes)")
    parser.add_argument("--box_threshold", type=float, default=0.65,
                        help="Similarity threshold for bounding box extraction")
    parser.add_argument("--box_padding",   type=int, default=10,
                        help="Pixels of padding added around each bounding box")
    parser.add_argument("--clip_text_key", default="title",
                        choices=["title", "text"],
                        help="Passage field encoded by CLIP: 'title' (article–section name)"
                             " or 'text' (full section, truncated to 77 tokens)")
    # ── multi-box options ──────────────────────────────────────────────────────
    parser.add_argument("--max_boxes",     type=int, default=3,
                        help="Max CLIP boxes per image (clip_multibox / clip_stepbox)")
    parser.add_argument("--box_fusion",    default="max",
                        choices=["max", "mean", "sum"],
                        help="How to fuse per-box similarity scores: "
                             "max (OR-style, any entity match) or mean/sum (all entities)")
    parser.add_argument("--min_area",      type=int, default=16,
                        help="Min blob area in pixels to keep (clip_multibox mode)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="NMS IoU threshold for deduplicating step boxes (clip_stepbox)")
    # ── box visualisation ──────────────────────────────────────────────────────
    parser.add_argument("--save_boxes",   action="store_true",
                        help="Save annotated images showing which region(s) were used for retrieval")
    parser.add_argument("--save_boxes_n", type=int, default=0,
                        help="Only save the first N box visualisations (0 = save all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load questions ─────────────────────────────────────────────────────────
    print("Loading questions …")
    questions = [json.loads(l) for l in open(args.questions)]
    if args.num_samples:
        questions = questions[:args.num_samples]
    print(f"  {len(questions)} questions")

    # ── load KB ───────────────────────────────────────────────────────────────
    print("Loading KB …")
    with open(args.kb_path) as f:
        kb = json.load(f)
    print(f"  {len(kb)} articles")

    # ── build passage pool ────────────────────────────────────────────────────
    print("Building passage pool …")
    all_questions = [json.loads(l) for l in open(args.questions)]
    passages, gold_pids_per_question_all = build_pool(
        kb, all_questions, num_distractors=args.num_distractors)
    gold_pids_per_question = gold_pids_per_question_all[:len(questions)]

    # ── load LVR model ────────────────────────────────────────────────────────
    print(f"\nLoading LVR model from {args.model_path} …")
    model, processor = load_model(args.model_path)
    print("LVR model loaded.\n")

    # ── load CLIP model (all clip_* modes) ────────────────────────────────────
    clip_model_obj = clip_proc_obj = None
    if args.retrieval_mode in ("clip_box", "clip_multibox", "clip_stepbox"):
        clip_model_obj, clip_proc_obj = load_clip_model(args.clip_model)

    # ── encode passages ───────────────────────────────────────────────────────
    if args.retrieval_mode in ("clip_box", "clip_multibox", "clip_stepbox"):
        cache_dir = os.path.join(args.output_dir, "passage_cache",
                                 f"clip_{args.clip_model.replace('/', '_')}"
                                 f"_dist{args.num_distractors}")
        passage_embeddings = encode_passages_clip(
            clip_model_obj, clip_proc_obj, passages,
            batch_size=args.encode_batch,
            cache_dir=cache_dir,
            text_key=args.clip_text_key,
        )
    else:
        cache_dir = os.path.join(args.output_dir, "passage_cache",
                                 f"pool_dist{args.num_distractors}_len{args.max_passage_len}")
        passage_embeddings = encode_passages(
            model, processor, passages,
            batch_size=args.encode_batch,
            max_length=args.max_passage_len,
            cache_dir=cache_dir,
        )

    if args.encode_only:
        print("Encoding done. Exiting (--encode_only).")
        return

    # ── run retrieval evaluation ───────────────────────────────────────────────
    is_clip_variant = args.retrieval_mode in ("clip_box", "clip_multibox", "clip_stepbox")
    if is_clip_variant:
        mode_tag = f"{args.retrieval_mode}_{args.clip_text_key}"
        if args.retrieval_mode != "clip_box":
            mode_tag += f"_boxes{args.max_boxes}_{args.box_fusion}"
    else:
        mode_tag = args.retrieval_mode
    out_name = (f"{mode_tag}_steps{args.steps:03d}_{args.decoding_strategy}"
                f"_n{len(questions)}_dist{args.num_distractors}.json")
    out_path = os.path.join(args.output_dir, out_name)

    viz_dir = os.path.join(args.output_dir, "box_viz")

    run_retrieval_eval(
        model, processor, questions, passages, passage_embeddings,
        gold_pids_per_question,
        image_dir=args.image_dir,
        steps=args.steps,
        decoding_strategy=args.decoding_strategy,
        aggregation_strategies=args.aggregation,
        top_k=args.top_k,
        out_path=out_path,
        debug_samples=args.debug_samples,
        retrieval_mode=args.retrieval_mode,
        clip_model=clip_model_obj,
        clip_processor=clip_proc_obj,
        box_threshold=args.box_threshold,
        box_padding=args.box_padding,
        box_fusion=args.box_fusion,
        max_boxes=args.max_boxes,
        min_area=args.min_area,
        iou_threshold=args.iou_threshold,
        save_boxes=args.save_boxes,
        save_boxes_n=args.save_boxes_n,
        viz_dir=viz_dir,
    )


if __name__ == "__main__":
    main()
