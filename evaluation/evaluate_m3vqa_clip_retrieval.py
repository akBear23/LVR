"""
Evaluate image retrieval on M3-VQA using LVR hidden states at <|lvr_latent_end|>.

Pipeline per sample
-------------------
  1. Run LVR generation with a forward hook to capture h_t
     (last_position_hidden_state) at every <|lvr_latent_end|> token.

  2. Compute cosine similarity between each h_t and every input image patch
     embedding (both live in the 3584-dim LLM hidden space).
     Average across all LVR steps → attention grid (eff_h × eff_w).

  3. Identify the bounding box of the high-attention region; crop the input
     image there.

  4. Encode the crop with CLIP → 512-dim query vector.

  5. Retrieve top-k entries from the pre-built CLIP KB index.

  6. Map retrieved image_urls back to KB entry URLs; check against the
     ground-truth evidence_urls from M3-VQA annotations.

Metrics: Recall@1 / @5 / @10, MRR — overall and per question_hop.

Prerequisites
-------------
  Run build_clip_kb_index.py first:
      python evaluation/build_clip_kb_index.py

Usage (from /mnt/data/lannth/mLAnR/lvr/):
    python evaluation/evaluate_m3vqa_clip_retrieval.py \\
        --model_path /mnt/data/lannth/mLAnR/checkpoints/LVR-7B \\
        --index_dir  /mnt/data/lannth/mLAnR/results/clip_index \\
        --steps 8
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig, CLIPModel, CLIPProcessor
from transformers.generation.configuration_utils import GenerationConfig

# ── GenerationConfig patch (same as other eval scripts) ──────────────────────
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
from src.train.monkey_patch_forward_lvr import (
    replace_qwen2_5_with_mixed_modality_forward_lvr,
)
from qwen_vl_utils import process_vision_info

SPATIAL_MERGE = 2   # Qwen2.5-VL always uses 2×2 spatial merging


# ─────────────────────────────────────────────────────────────────────────────
# URL utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    return url.replace("en.m.wikipedia.org", "en.wikipedia.org").rstrip("/")


# ─────────────────────────────────────────────────────────────────────────────
# LVR model loading + h_t extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_lvr_model(model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        inference_mode=True, lvr_head=config.lvr_head
    )
    model = QwenWithLVR.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def extract_h_t(
    model, processor,
    img_path: str, question: str,
    steps: int, decoding_strategy: str = "steps",
):
    """
    Run LVR generation and capture h_t (last_position_hidden_state) at each
    <|lvr_latent_end|> token.

    Returns
    -------
    h_t_list      : list[(H,) tensor]  — one per LVR step
    image_embeds  : (N_img, H) tensor  — input image patches in LLM space
    eff_h, eff_w  : int                — spatial patch grid dimensions
    generated_text: str
    """
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text",  "text": question},
    ]}]
    text_fmt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_fmt], images=img_inputs, videos=vid_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

    lvr_token_id = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")
    T, tok_h, tok_w = inputs["image_grid_thw"][0].tolist()
    eff_h = int(tok_h) // SPATIAL_MERGE
    eff_w = int(tok_w) // SPATIAL_MERGE

    # image patch embeddings in LLM hidden space — same space as h_t
    with torch.no_grad():
        image_embeds = model.model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        image_embeds = torch.cat(image_embeds, dim=0).float().cpu()  # (N_img, H)

    # hook captures last_position_hidden_state at every generation step
    captured: list[torch.Tensor] = []
    def _hook(module, inp, output):
        if (hasattr(output, "last_position_hidden_state")
                and output.last_position_hidden_state is not None):
            captured.append(
                output.last_position_hidden_state[0].float().detach().cpu()
            )

    handle   = model.register_forward_hook(_hook)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=512,
            decoding_strategy=decoding_strategy,
            lvr_steps=[steps],
            return_dict_in_generate=True,
        )
    handle.remove()

    gen_tokens = gen_out.sequences[0, input_len:]
    gen_text   = processor.decode(gen_tokens, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)

    lvr_indices = (gen_tokens == lvr_token_id).nonzero(as_tuple=True)[0]
    h_t_list = [
        captured[s.item()]
        for s in lvr_indices if s.item() < len(captured)
    ]
    return h_t_list, image_embeds, eff_h, eff_w, gen_text


# ─────────────────────────────────────────────────────────────────────────────
# Attention map → attention-guided crop
# ─────────────────────────────────────────────────────────────────────────────

def compute_attention_map(
    h_t_list: list, image_embeds: torch.Tensor,
    eff_h: int, eff_w: int,
) -> np.ndarray:
    """
    Average cosine-similarity maps of all LVR steps.
    Returns a (eff_h, eff_w) float32 array in [0, 1].
    """
    maps = []
    for h_t in h_t_list:
        sims = F.cosine_similarity(h_t.unsqueeze(0), image_embeds, dim=-1)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        maps.append(sims.numpy().reshape(eff_h, eff_w))
    avg_map = np.stack(maps).mean(axis=0)
    return avg_map.astype(np.float32)


def attention_bbox(avg_map: np.ndarray, threshold: float = 0.5):
    """
    Return (r0, c0, r1, c1) bounding box in patch coordinates of the
    high-attention region.  Falls back to top-25% if no patch crosses
    the threshold.
    """
    mask = avg_map >= threshold
    if mask.sum() == 0:
        thr  = float(np.percentile(avg_map, 75))
        mask = avg_map >= thr
    ys, xs = np.where(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def crop_attention_region(
    image: Image.Image,
    r0: int, c0: int, r1: int, c1: int,
    eff_h: int, eff_w: int,
    min_px: int = 32,
) -> Image.Image:
    """Convert patch-space bbox → pixel-space and crop; falls back to full image."""
    W, H  = image.size
    x0 = max(0, int(c0 * W / eff_w))
    y0 = max(0, int(r0 * H / eff_h))
    x1 = min(W, int(c1 * W / eff_w))
    y1 = min(H, int(r1 * H / eff_h))
    if x1 - x0 < min_px or y1 - y0 < min_px:
        return image
    return image.crop((x0, y0, x1, y1))


# ─────────────────────────────────────────────────────────────────────────────
# CLIP index loading + query
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_index(index_dir: str):
    emb_file  = os.path.join(index_dir, "clip_embeddings.npy")
    meta_file = os.path.join(index_dir, "clip_index_meta.json")
    if not os.path.exists(emb_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(
            f"CLIP index not found in {index_dir}. "
            "Run build_clip_kb_index.py first."
        )
    embeddings = torch.from_numpy(np.load(emb_file)).float()  # (N, D)
    with open(meta_file) as f:
        meta = json.load(f)
    assert len(embeddings) == len(meta)
    print(f"Loaded CLIP index: {len(embeddings)} entries, dim={embeddings.shape[1]}")
    return embeddings, meta


def load_clip_model(clip_model_name: str = "openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model     = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    print(f"Loaded CLIP: {clip_model_name}")
    return clip_model, clip_processor, device


@torch.no_grad()
def clip_encode_image(
    clip_model, clip_processor, image: Image.Image, device: str
) -> torch.Tensor:
    """Returns a normalised (D,) CLIP image feature."""
    inputs = clip_processor(images=[image], return_tensors="pt").to(device)
    feat   = clip_model.get_image_features(**inputs)
    feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].float().cpu()


def retrieve_topk(
    query: torch.Tensor,        # (D,) normalised
    index_embs: torch.Tensor,   # (N, D) normalised
    k: int = 10,
):
    """Cosine-similarity search (both inputs are L2-normalised)."""
    sims    = index_embs @ query                        # (N,)
    topk    = torch.topk(sims, min(k, len(sims)))
    return topk.indices.tolist(), topk.values.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank(retrieved_kb_urls: list, evidence_urls: list) -> float:
    ev_norm = {normalize_url(u) for u in evidence_urls}
    for rank, url in enumerate(retrieved_kb_urls, start=1):
        if normalize_url(url) in ev_norm:
            return 1.0 / rank
    return 0.0


def compute_metrics(results: list, ks=(1, 5, 10)) -> dict:
    valid_results = [r for r in results if not r.get("skipped")]
    N = len(valid_results)
    if N == 0:
        print("No valid results to evaluate.")
        return {}

    recall   = {k: 0 for k in ks}
    mrr_sum  = 0.0
    hop_stats: dict[int, dict] = {}

    for r in valid_results:
        retrieved = r["retrieved_kb_urls"]
        evidence  = r["evidence_urls"]
        hop       = r.get("question_hop", -1)
        rr        = reciprocal_rank(retrieved, evidence)
        mrr_sum  += rr

        if hop not in hop_stats:
            hop_stats[hop] = {"total": 0, "mrr": 0.0, **{k: 0 for k in ks}}
        hop_stats[hop]["total"] += 1
        hop_stats[hop]["mrr"]   += rr

        ev_norm = {normalize_url(u) for u in evidence}
        for k in ks:
            if any(normalize_url(u) in ev_norm for u in retrieved[:k]):
                recall[k]          += 1
                hop_stats[hop][k]  += 1

    print(f"\n{'='*60}")
    print(f"  LVR h_t → Attention Crop → CLIP Retrieval")
    print(f"  Valid samples: {N} / {len(results)}")
    for k in ks:
        print(f"  Recall@{k:2d}: {recall[k]/N*100:6.2f}%  ({recall[k]}/{N})")
    print(f"  MRR      : {mrr_sum/N*100:6.2f}%")
    print(f"\n  Breakdown by question hop:")
    for hop in sorted(hop_stats):
        hs  = hop_stats[hop]
        hn  = hs["total"]
        row = f"  Hop {hop:2d} (n={hn:4d}):"
        for k in ks:
            row += f"  R@{k}={hs[k]/hn*100:.1f}%"
        row += f"  MRR={hs['mrr']/hn*100:.1f}%"
        print(row)
    print(f"{'='*60}\n")

    return {
        "total"       : len(results),
        "valid"       : N,
        "recall"      : {f"@{k}": recall[k] / N for k in ks},
        "mrr"         : mrr_sum / N,
        "hop_breakdown": {
            str(hop): {
                "total"  : hs["total"],
                "mrr"    : hs["mrr"] / hs["total"],
                **{f"recall@{k}": hs[k] / hs["total"] for k in ks},
            }
            for hop, hs in hop_stats.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = os.path.basename(args.model_path)
    out_file = os.path.join(
        args.output_dir,
        f"{run_name}_clip_steps{args.steps:03d}.json",
    )

    # ── load CLIP index ───────────────────────────────────────────────────────
    index_embs, index_meta = load_clip_index(args.index_dir)

    # ── load CLIP model ───────────────────────────────────────────────────────
    clip_model, clip_processor, clip_device = load_clip_model(args.clip_model)

    # ── load LVR model ────────────────────────────────────────────────────────
    print(f"\nLoading LVR model: {args.model_path} …")
    lvr_model, lvr_processor = load_lvr_model(args.model_path)
    print("LVR model loaded.\n")

    # ── load questions ────────────────────────────────────────────────────────
    questions = []
    with open(args.questions) as f:
        for line in f:
            questions.append(json.loads(line))
    if args.max_samples:
        questions = questions[: args.max_samples]
    print(f"Evaluating on {len(questions)} questions")

    # ── resume from cached results ────────────────────────────────────────────
    if os.path.exists(out_file) and not args.rebuild:
        print(f"Loading cached results from {out_file}")
        with open(out_file) as f:
            cached = json.load(f)
        results = cached["results"]
    else:
        results: list[dict] = []

        for item in tqdm(questions, desc="Retrieving"):
            data_id  = item["data_id"]
            img_path = os.path.join(args.image_dir, item["image_id"])
            question = item["question"]
            evidence = item.get("evidence_urls", [])

            if not os.path.exists(img_path):
                results.append({
                    "data_id": data_id, "skipped": True, "reason": "image_not_found",
                })
                continue

            # ── step 1: extract h_t at <|lvr_latent_end|> ────────────────────
            try:
                h_t_list, image_embeds, eff_h, eff_w, gen_text = extract_h_t(
                    lvr_model, lvr_processor, img_path, question,
                    args.steps, args.decoding_strategy,
                )
            except Exception as e:
                results.append({
                    "data_id": data_id, "skipped": True, "reason": f"extract_error: {e}",
                })
                continue

            if not h_t_list:
                results.append({
                    "data_id": data_id, "skipped": True, "reason": "no_lvr_tokens",
                    "generated_text": gen_text,
                })
                continue

            # ── step 2: attention map (h_t × image patches) ──────────────────
            avg_map = compute_attention_map(h_t_list, image_embeds, eff_h, eff_w)
            r0, c0, r1, c1 = attention_bbox(avg_map, threshold=args.attn_threshold)

            # ── step 3: crop input image at high-attention region ─────────────
            pil_img  = Image.open(img_path).convert("RGB")
            crop_img = crop_attention_region(pil_img, r0, c0, r1, c1, eff_h, eff_w)

            # ── step 4: CLIP-encode the crop ──────────────────────────────────
            clip_query = clip_encode_image(
                clip_model, clip_processor, crop_img, clip_device
            )   # (D,)

            # ── step 5: retrieve top-k from CLIP index ────────────────────────
            top_indices, top_scores = retrieve_topk(clip_query, index_embs, k=args.top_k)
            retrieved_meta   = [index_meta[i] for i in top_indices]
            retrieved_kb_urls= [m["kb_url"] for m in retrieved_meta]

            results.append({
                "data_id"          : data_id,
                "image_id"         : item["image_id"],
                "question"         : question,
                "answers"          : item["answers"],
                "evidence_urls"    : evidence,
                "question_hop"     : item.get("question_hop", -1),
                "generated_text"   : gen_text,
                "num_lvr_steps"    : len(h_t_list),
                # retrieval results
                "retrieved_kb_urls": retrieved_kb_urls,
                "retrieved_scores" : top_scores,
                "retrieved_meta"   : retrieved_meta,
                # attention bbox (patch space)
                "attn_bbox_patch"  : [r0, c0, r1, c1],
                "attn_grid_shape"  : [eff_h, eff_w],
                # flatten attention map to list for storage
                "attn_avg_map"     : avg_map.tolist(),
            })

        with open(out_file, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"Saved results → {out_file}")

    # ── compute & save metrics ────────────────────────────────────────────────
    metrics = compute_metrics(results, ks=[1, 5, 10])
    summary_file = out_file.replace(".json", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Summary        → {summary_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M3-VQA image retrieval via LVR h_t → attention crop → CLIP"
    )
    parser.add_argument("--model_path", required=True,
        help="Path to LVR-7B checkpoint")
    parser.add_argument("--index_dir",
        default="/mnt/data/lannth/mLAnR/results/clip_index",
        help="Directory containing clip_embeddings.npy and clip_index_meta.json")
    parser.add_argument("--questions",
        default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--image_dir",
        default="/mnt/data/lannth/mLAnR/M3-VQA/images")
    parser.add_argument("--output_dir",
        default="/mnt/data/lannth/mLAnR/results/clip_retrieval")
    parser.add_argument("--steps", type=int, default=8,
        help="Number of LVR revision steps")
    parser.add_argument("--decoding_strategy", default="steps",
        choices=["steps", "latent"])
    parser.add_argument("--clip_model",
        default="openai/clip-vit-base-patch32")
    parser.add_argument("--top_k", type=int, default=10,
        help="Number of KB images to retrieve per query")
    parser.add_argument("--attn_threshold", type=float, default=0.5,
        help="Normalised cosine-sim threshold for the high-attention region (0–1)")
    parser.add_argument("--max_samples", type=int, default=None,
        help="Truncate the question set for quick testing")
    parser.add_argument("--rebuild", action="store_true",
        help="Ignore cached results and re-run inference")
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()
