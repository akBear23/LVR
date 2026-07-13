"""
Retrieval module for GRPO+LVR training.

Supports two query modes:
  - "clip_image": encode a cropped image region with CLIP (uses LVR attention bbox)
  - "dense_text": encode a <search> query string with a dense E5/BGE encoder (AutoRefine-style)

The passage index is built once from the M3-VQA KB and cached to disk.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional
from transformers import AutoTokenizer, AutoModel


SPATIAL_MERGE = 2   # Qwen2.5-VL merges 2x2 patches


# ─────────────────────────────────────────────────────────────────────────────
# Index building
# ─────────────────────────────────────────────────────────────────────────────

def load_kb(kb_path: str) -> dict:
    print(f"Loading KB from {kb_path} …")
    with open(kb_path) as f:
        return json.load(f)


def build_passage_list(kb: dict) -> list[dict]:
    """Flatten KB into a list of passage dicts with title and text fields."""
    passages = []
    for url, art in kb.items():
        art_title    = art.get("title", url.split("/")[-1])
        sec_texts    = art.get("section_texts", [])
        sec_titles   = art.get("section_titles", [])
        for sec_id, text in enumerate(sec_texts):
            stitle = sec_titles[sec_id] if sec_id < len(sec_titles) else ""
            title  = f"{art_title} – {stitle}" if stitle else art_title
            passages.append({
                "pid":   f"{url}#{sec_id}",
                "url":   url,
                "title": title,
                "text":  text,
            })
    print(f"  {len(passages)} passages from {len(kb)} articles")
    return passages


def encode_passages_clip(
    clip_model,
    clip_processor,
    passages: list[dict],
    batch_size: int = 256,
    cache_dir: Optional[str] = None,
    text_key: str = "title",
) -> np.ndarray:
    """Encode passage texts with CLIP's text encoder. Streams directly to disk.

    Returns a memory-mapped (N, D) float32 array so the full matrix stays on
    disk and is only paged into RAM on access.
    """
    emb_path = idx_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        emb_path = os.path.join(cache_dir, f"passage_clip_{text_key}.npy")
        idx_path = os.path.join(cache_dir, f"passage_clip_{text_key}_pids.json")

    if emb_path and os.path.exists(emb_path) and os.path.exists(idx_path):
        with open(idx_path) as f:
            cached_pids = json.load(f)
        if cached_pids == [p["pid"] for p in passages]:
            embs = np.load(emb_path, mmap_mode="r")
            print(f"Cache hit: {emb_path}  shape={embs.shape}")
            return embs
        print("Cache PID mismatch — re-encoding.")

    device = next(clip_model.parameters()).device

    # Probe embedding dim with the first batch.
    probe_batch = [p[text_key] for p in passages[:1]]
    probe_enc = clip_processor(text=probe_batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
    probe_enc = {k: v.to(device) for k, v in probe_enc.items()}
    with torch.no_grad():
        probe_feat = clip_model.get_text_features(**probe_enc)
    D = probe_feat.shape[1]

    # Allocate memory-mapped output — written batch-by-batch, never fully in RAM.
    N = len(passages)
    if emb_path:
        embs = np.lib.format.open_memmap(emb_path, mode="w+", dtype="float32", shape=(N, D))
    else:
        embs = np.empty((N, D), dtype="float32")

    offset = 0
    for i in tqdm(range(0, N, batch_size), desc=f"Encoding passages (CLIP/{text_key})"):
        batch = [p[text_key] for p in passages[i : i + batch_size]]
        enc = clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            feats = clip_model.get_text_features(**enc)
        feats_np = feats.float().cpu().numpy()
        embs[offset : offset + len(feats_np)] = feats_np
        offset += len(feats_np)

    if emb_path:
        embs.flush()
        with open(idx_path, "w") as f:
            json.dump([p["pid"] for p in passages], f)
        print(f"Saved → {emb_path}  shape={embs.shape}")
        # Re-open read-only so the OS can evict pages freely during training.
        embs = np.load(emb_path, mmap_mode="r")

    return embs


# ─────────────────────────────────────────────────────────────────────────────
# Dense text retrieval (AutoRefine-style: E5/BGE encoder + cosine index)
# ─────────────────────────────────────────────────────────────────────────────

class DenseEncoder:
    """Transformer-based text encoder matching AutoRefine's Encoder class."""

    def __init__(
        self,
        model_name_or_path: str,
        pooling_method: str = "mean",
        max_length: int = 256,
        fp16: bool = False,
    ):
        self.model_name = model_name_or_path
        self.pooling_method = pooling_method
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.eval()
        if fp16:
            self.model = self.model.half()
        self.model.cuda()  # starts on GPU; moved to CPU after index build in launch.py

    def _pool(self, output, attention_mask) -> torch.Tensor:
        if self.pooling_method == "mean":
            hidden = output.last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            return hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling_method == "cls":
            return output.last_hidden_state[:, 0]
        elif self.pooling_method == "pooler":
            return output.pooler_output
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        elif "bge" in self.model_name.lower() and is_query:
            texts = [
                f"Represent this sentence for searching relevant passages: {t}"
                for t in texts
            ]

        enc = self.tokenizer(
            texts, max_length=self.max_length,
            padding=True, truncation=True, return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.model(**enc, return_dict=True)
        emb = self._pool(out, enc["attention_mask"])
        if "dpr" not in self.model_name.lower():
            emb = F.normalize(emb, dim=-1)
        return emb.float().cpu().numpy().astype(np.float32)


def encode_passages_dense(
    encoder: DenseEncoder,
    passages: list[dict],
    batch_size: int = 128,
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """Encode passages with a dense text encoder. Streams directly to disk.

    Each passage is encoded as 'title\\ntext' (full content, not title-only).
    Returns a memory-mapped (N, D) float32 array.
    """
    model_slug = Path(encoder.model_name).name.replace("/", "_")
    emb_path = idx_path = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        emb_path = os.path.join(cache_dir, f"passage_dense_{model_slug}.npy")
        idx_path = os.path.join(cache_dir, f"passage_dense_{model_slug}_pids.json")

    if emb_path and os.path.exists(emb_path) and os.path.exists(idx_path):
        with open(idx_path) as f:
            cached_pids = json.load(f)
        if cached_pids == [p["pid"] for p in passages]:
            embs = np.load(emb_path, mmap_mode="r")
            print(f"Cache hit: {emb_path}  shape={embs.shape}")
            return embs
        print("Cache PID mismatch — re-encoding.")

    # Probe embedding dim.
    probe = encoder.encode([passages[0]["title"] + "\n" + passages[0]["text"]], is_query=False)
    D = probe.shape[1]

    N = len(passages)
    if emb_path:
        embs = np.lib.format.open_memmap(emb_path, mode="w+", dtype="float32", shape=(N, D))
    else:
        embs = np.empty((N, D), dtype="float32")

    offset = 0
    for i in tqdm(range(0, N, batch_size), desc=f"Encoding passages (dense/{model_slug})"):
        batch = [
            p["title"] + "\n" + p["text"]
            for p in passages[i : i + batch_size]
        ]
        batch_emb = encoder.encode(batch, is_query=False)
        embs[offset : offset + len(batch_emb)] = batch_emb
        offset += len(batch_emb)

    if emb_path:
        embs.flush()
        with open(idx_path, "w") as f:
            json.dump([p["pid"] for p in passages], f)
        print(f"Saved → {emb_path}  shape={embs.shape}")
        embs = np.load(emb_path, mmap_mode="r")

    return embs


# ─────────────────────────────────────────────────────────────────────────────
# Query extraction from LVR hidden states
# ─────────────────────────────────────────────────────────────────────────────

def _top_region_box(heatmap: np.ndarray, threshold: float = 0.65):
    mask = (heatmap >= threshold).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def clip_query_from_lvr_states(
    h_list: list[np.ndarray],          # list of (D_llm,) arrays, one per LVR step
    image_embeds: np.ndarray,           # (N_patches, D_llm) visual patch embeddings in LLM space
    thw: tuple,                         # (T, H, W) patch grid from image_grid_thw
    image: Image.Image,
    clip_model,
    clip_processor,
    box_threshold: float = 0.65,
    box_padding:   int   = 10,
) -> Optional[np.ndarray]:
    """
    1. Compute per-step cosine similarity maps between each h_t and image patches.
    2. Average → bounding box of high-attention region.
    3. Crop image to bbox.
    4. CLIP-encode the crop.

    Returns CLIP image embedding (D_clip,) or None if no valid region found.
    """
    W, H = image.size
    eff_h = int(thw[1]) // SPATIAL_MERGE
    eff_w = int(thw[2]) // SPATIAL_MERGE

    per_step_grids = []
    for h_t in h_list:
        h_t_t = torch.from_numpy(h_t).float()
        patches = torch.from_numpy(image_embeds).float()
        sims = F.cosine_similarity(h_t_t.unsqueeze(0), patches, dim=-1)  # (N,)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        per_step_grids.append(sims.numpy().reshape(eff_h, eff_w))

    if not per_step_grids:
        return None

    avg_grid = np.mean(per_step_grids, axis=0)
    heatmap = np.array(
        Image.fromarray((avg_grid * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    box = _top_region_box(heatmap, threshold=box_threshold)
    if box is None:
        return None

    x1, y1, x2, y2 = box
    x1 = max(0, x1 - box_padding)
    y1 = max(0, y1 - box_padding)
    x2 = min(W, x2 + box_padding)
    y2 = min(H, y2 + box_padding)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image.crop((x1, y1, x2, y2))
    device = next(clip_model.parameters()).device
    inp = clip_processor(images=crop, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = clip_model.get_image_features(**inp)   # (1, D)
    return feat.float().cpu().numpy().squeeze(0)       # (D,)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query_vec: np.ndarray,
    passage_embeddings: np.ndarray,
    top_k: int = 3,
    chunk_size: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity search over a memory-mapped embedding matrix.

    Processes embeddings in chunks so only ~1.5 GB is resident at a time
    instead of materializing a full normalized copy of the 46 GB mmap array.
    Returns (top_k_indices, top_k_scores) sorted descending.
    """
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    N = len(passage_embeddings)
    k = min(top_k, N)

    # Keep a running heap of (score, global_idx) for the top-k seen so far.
    top_scores = np.full(k, -np.inf, dtype=np.float32)
    top_indices = np.zeros(k, dtype=np.int64)

    for start in range(0, N, chunk_size):
        chunk = np.array(passage_embeddings[start : start + chunk_size], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        chunk_norm = chunk / (norms + 1e-8)
        sims = chunk_norm @ q  # (chunk_size,)

        # Merge chunk top-k with global top-k
        global_idx = np.arange(start, start + len(sims), dtype=np.int64)
        combined_sims = np.concatenate([top_scores, sims])
        combined_idx = np.concatenate([top_indices, global_idx])
        if len(combined_sims) > k:
            part = np.argpartition(-combined_sims, k)[:k]
            order = part[np.argsort(-combined_sims[part])]
        else:
            order = np.argsort(-combined_sims)
        top_scores = combined_sims[order]
        top_indices = combined_idx[order]

    return top_indices, top_scores


# ─────────────────────────────────────────────────────────────────────────────
# Passage index (singleton for reuse across rollouts)
# ─────────────────────────────────────────────────────────────────────────────

class PassageIndex:
    """Pre-built passage index, reused across all rollouts in a training run.

    Visual retrieval uses CLIP (image crop → passage title embeddings).
    Text retrieval uses a dense E5/BGE encoder (query text → title+body embeddings),
    matching AutoRefine's retrieval method.
    """

    def __init__(
        self,
        kb_path: str,
        clip_model,
        clip_processor,
        cache_dir: Optional[str] = None,
        text_key: str = "title",
        retrieval_model_path: str = "intfloat/e5-base-v2",
    ):
        kb = load_kb(kb_path)
        self.passages = build_passage_list(kb)

        # CLIP index — used for visual retrieval (image crop query)
        self.embeddings = encode_passages_clip(
            clip_model, clip_processor, self.passages,
            cache_dir=cache_dir, text_key=text_key,
        )
        self.clip_model     = clip_model
        self.clip_processor = clip_processor

        # Dense index — used for text retrieval (<search> query, AutoRefine-style)
        self.dense_encoder = DenseEncoder(retrieval_model_path)
        self.dense_embeddings = encode_passages_dense(
            self.dense_encoder, self.passages, cache_dir=cache_dir,
        )

    def retrieve_by_image(
        self,
        h_list: list[np.ndarray],
        image_embeds: np.ndarray,
        thw: tuple,
        image: Image.Image,
        top_k: int = 3,
        box_threshold: float = 0.65,
        box_padding:   int   = 10,
    ) -> list[dict]:
        """Retrieve passages using LVR attention bbox → CLIP image crop."""
        query = clip_query_from_lvr_states(
            h_list, image_embeds, thw, image,
            self.clip_model, self.clip_processor,
            box_threshold=box_threshold, box_padding=box_padding,
        )
        if query is None:
            return []
        idx, _ = retrieve(query, self.embeddings, top_k=top_k)
        return [self.passages[i] for i in idx]

    def retrieve_by_text(self, text: str, top_k: int = 3) -> list[dict]:
        """Retrieve passages using a text query → dense E5/BGE encoder (AutoRefine-style)."""
        query = self.dense_encoder.encode([text], is_query=True)[0]  # (D,)
        idx, _ = retrieve(query, self.dense_embeddings, top_k=top_k)
        return [self.passages[i] for i in idx]
