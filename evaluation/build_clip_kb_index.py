"""
Build a CLIP index from encyclopedic_kb_wiki.json.

Two modes (--mode):
  image  (default)
      Encode actual KB images with CLIP's image encoder.
      Requires the images to be either downloaded live (needs internet) or
      pre-downloaded with download_kb_images.py (--image_cache_dir).

  text
      Encode KB image captions (image_reference_descriptions + title) with
      CLIP's text encoder.  No internet needed — runs entirely from the
      JSON file.  Because CLIP is trained to align text and image features,
      text embeddings can still retrieve the correct KB entry when the LVR
      crop is used as the image-side query.

Output (saved to --output_dir):
  clip_embeddings.npy      — float32 array (N, D)
  clip_index_meta.json     — list of {kb_url, image_url, caption, title} per row

Usage (from /mnt/data/lannth/mLAnR/lvr/):
    # Text mode — no internet needed, runs in minutes
    python evaluation/build_clip_kb_index.py \\
        --mode text \\
        --output_dir /mnt/data/lannth/mLAnR/results/clip_index_text

    # Image mode — local cache
    python evaluation/build_clip_kb_index.py \\
        --mode image \\
        --image_cache_dir /mnt/data/lannth/mLAnR/results/kb_images \\
        --output_dir /mnt/data/lannth/mLAnR/results/clip_index_image \\
        --num_workers 64 --batch_size 512

Add --rebuild to discard any existing checkpoint and start fresh.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, argparse, random, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import requests

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
        "Gecko/20100101 Firefox/120.0"
    )
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    return url.replace("en.m.wikipedia.org", "en.wikipedia.org").rstrip("/")


def url_to_filename(image_url: str) -> str:
    """Same mapping as download_kb_images.py — md5 hex + .jpg."""
    return hashlib.md5(image_url.encode()).hexdigest() + ".jpg"


def load_evidence_urls(questions_path: str) -> set:
    urls = set()
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            for url in item.get("evidence_urls", []):
                urls.add(normalize_url(url))
    return urls


def load_task_from_cache(args: tuple) -> tuple:
    """Load one image from local cache. Returns (kb_url, title, img_url, pil_or_None)."""
    kb_url, title, img_url, image_cache_dir = args
    local_path = os.path.join(image_cache_dir, url_to_filename(img_url))
    try:
        img = Image.open(local_path).convert("RGB")
        return kb_url, title, img_url, img
    except Exception:
        return kb_url, title, img_url, None


def download_task(args: tuple) -> tuple:
    """Worker function: download one image. Returns (kb_url, title, img_url, pil_or_None)."""
    kb_url, title, img_url = args
    try:
        r = requests.get(img_url, headers=_HEADERS, timeout=15)
        if r.status_code == 200 and r.content:
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return kb_url, title, img_url, img
    except Exception:
        pass
    return kb_url, title, img_url, None


# ─────────────────────────────────────────────────────────────────────────────
# CLIP encoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_image_batch(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    images: list,
    device: str,
) -> np.ndarray:
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()


@torch.no_grad()
def encode_text_batch(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    texts: list[str],
    device: str,
) -> np.ndarray:
    # CLIP truncates at 77 tokens automatically
    inputs = clip_processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().cpu().numpy()


def make_caption_text(title: str, caption: str) -> str:
    """
    Combine KB entry title + image caption into a single CLIP text string.
    Empty captions fall back to just the title.
    """
    caption = caption.strip()
    title   = title.strip()
    if caption:
        return f"{title}: {caption}"
    return title


# ─────────────────────────────────────────────────────────────────────────────
# Text-mode index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_text_index(
    kb_path: str,
    questions_path: str,
    output_dir: str,
    num_distractors: int = 10_000,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    max_captions_per_entry: int = 1,
    batch_size: int = 1024,
):
    """
    Encode image_reference_descriptions (captions) from KB entries with CLIP's
    text encoder.  No internet access or downloaded images required.
    """
    os.makedirs(output_dir, exist_ok=True)
    emb_file  = os.path.join(output_dir, "clip_embeddings.npy")
    meta_file = os.path.join(output_dir, "clip_index_meta.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP: {clip_model_name}  (device={device})")
    clip_model     = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # ── evidence URLs ─────────────────────────────────────────────────────────
    print("Collecting evidence URLs from questions …")
    evidence_urls = load_evidence_urls(questions_path)
    print(f"  {len(evidence_urls)} unique evidence URLs")

    # ── load KB ───────────────────────────────────────────────────────────────
    print(f"Loading KB (may take ~60 s for 16 GB) …")
    with open(kb_path) as f:
        kb = json.load(f)
    print(f"  {len(kb):,} KB entries")

    # ── partition into relevant / distractor ──────────────────────────────────
    relevant, distractors = [], []
    for url, entry in kb.items():
        # require at least one image_url so meta stays compatible with image mode
        if not entry.get("image_urls"):
            continue
        (relevant if normalize_url(url) in evidence_urls else distractors).append(
            (url, entry)
        )

    random.seed(42)
    random.shuffle(distractors)
    candidates = relevant + distractors[:num_distractors]
    print(
        f"  {len(relevant)} relevant + {len(distractors[:num_distractors])} distractor"
        f" = {len(candidates)} entries"
    )

    # ── build flat list of (text, meta) ──────────────────────────────────────
    all_texts: list[str]  = []
    all_meta:  list[dict] = []

    for url, entry in candidates:
        title      = entry.get("title", "")
        img_urls   = entry.get("image_urls", [])
        captions   = entry.get("image_reference_descriptions", [])

        for idx in range(min(max_captions_per_entry, len(img_urls))):
            caption  = captions[idx] if idx < len(captions) else ""
            text     = make_caption_text(title, caption)
            img_url  = img_urls[idx]
            all_texts.append(text)
            all_meta.append({
                "kb_url"   : url,
                "image_url": img_url,
                "caption"  : caption,
                "title"    : title,
            })

    n_empty = sum(1 for t in all_texts if not t.strip())
    print(f"  {len(all_texts)} captions to encode  ({n_empty} empty → title-only)\n")

    # ── GPU encode in batches ─────────────────────────────────────────────────
    all_embeddings: list[np.ndarray] = []

    for start in tqdm(range(0, len(all_texts), batch_size),
                      desc="Text encode", unit="batch"):
        batch = all_texts[start : start + batch_size]
        embs  = encode_text_batch(clip_model, clip_processor, batch, device)
        all_embeddings.append(embs)

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(emb_file, embeddings)
    with open(meta_file, "w") as f:
        json.dump(all_meta, f, indent=2)

    n_relevant_indexed = sum(
        1 for m in all_meta if normalize_url(m["kb_url"]) in evidence_urls
    )
    print(
        f"\nText index complete:"
        f"\n  Total embeddings : {len(embeddings)}"
        f"\n  Relevant indexed : {n_relevant_indexed}"
        f"\n  Embeddings       : {emb_file}"
        f"\n  Metadata         : {meta_file}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers  (used by image mode only)
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_paths(output_dir: str):
    return (
        os.path.join(output_dir, "_ckpt_embeddings.npy"),
        os.path.join(output_dir, "_ckpt_meta.json"),
        os.path.join(output_dir, "_ckpt_done_urls.json"),
    )


def load_checkpoint(output_dir: str):
    emb_ckpt, meta_ckpt, done_ckpt = ckpt_paths(output_dir)
    if not (os.path.exists(emb_ckpt) and os.path.exists(meta_ckpt)
            and os.path.exists(done_ckpt)):
        return [], [], set()
    embs = list(np.load(emb_ckpt))
    with open(meta_ckpt) as f:
        meta = json.load(f)
    with open(done_ckpt) as f:
        done = set(json.load(f))
    print(f"  Resuming from checkpoint: {len(embs)} embeddings, {len(done)} done image-URLs")
    return embs, meta, done


def save_checkpoint(output_dir: str, embeddings: list, meta: list, done_urls: set):
    emb_ckpt, meta_ckpt, done_ckpt = ckpt_paths(output_dir)
    np.save(emb_ckpt, np.stack(embeddings))
    with open(meta_ckpt, "w") as f:
        json.dump(meta, f)
    with open(done_ckpt, "w") as f:
        json.dump(list(done_urls), f)


def finalize(output_dir: str, embeddings: list, meta: list):
    """Write final output and remove checkpoint files."""
    emb_file  = os.path.join(output_dir, "clip_embeddings.npy")
    meta_file = os.path.join(output_dir, "clip_index_meta.json")
    np.save(emb_file, np.stack(embeddings))
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    for p in ckpt_paths(output_dir):
        if os.path.exists(p):
            os.remove(p)
    return emb_file, meta_file


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    kb_path: str,
    questions_path: str,
    output_dir: str,
    num_distractors: int = 10_000,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    max_images_per_entry: int = 1,
    batch_size: int = 512,
    num_workers: int = 64,
    save_every: int = 20,   # checkpoint every N GPU batches
    image_cache_dir: str | None = None,  # if set, load images from local dir
):
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP: {clip_model_name}  (device={device})")
    clip_model     = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # ── evidence URLs ─────────────────────────────────────────────────────────
    print("Collecting evidence URLs from questions …")
    evidence_urls = load_evidence_urls(questions_path)
    print(f"  {len(evidence_urls)} unique evidence URLs")

    # ── load KB ───────────────────────────────────────────────────────────────
    print(f"Loading KB (may take ~60 s for 16 GB) …")
    with open(kb_path) as f:
        kb = json.load(f)
    print(f"  {len(kb):,} KB entries")

    # ── partition into relevant / distractor ──────────────────────────────────
    relevant, distractors = [], []
    for url, entry in kb.items():
        if not entry.get("image_urls"):
            continue
        (relevant if normalize_url(url) in evidence_urls else distractors).append(
            (url, entry)
        )

    random.seed(42)
    random.shuffle(distractors)
    candidates = relevant + distractors[:num_distractors]
    print(
        f"  {len(relevant)} relevant + {len(distractors[:num_distractors])} distractor"
        f" = {len(candidates)} entries to index"
    )

    # ── build flat list of all tasks ──────────────────────────────────────────
    all_tasks: list[tuple] = []
    for url, entry in candidates:
        title = entry.get("title", "")
        for img_url in entry["image_urls"][:max_images_per_entry]:
            if image_cache_dir:
                all_tasks.append((url, title, img_url, image_cache_dir))
            else:
                all_tasks.append((url, title, img_url))

    source = f"local cache ({image_cache_dir})" if image_cache_dir else "network download"
    print(f"  {len(all_tasks)} total images to load from {source}\n")

    # ── resume from checkpoint ────────────────────────────────────────────────
    all_embeddings, all_meta, done_urls = load_checkpoint(output_dir)
    remaining_tasks = [t for t in all_tasks if t[2] not in done_urls]
    print(f"  {len(remaining_tasks)} images remaining after checkpoint\n")

    # ── load/download images in parallel + GPU encode ─────────────────────────
    worker_fn = load_task_from_cache if image_cache_dir else download_task
    n_failed  = 0
    batch_buf_imgs: list[Image.Image] = []
    batch_buf_meta: list[dict]        = []
    batches_since_ckpt = 0

    def flush_gpu_batch():
        nonlocal batches_since_ckpt
        if not batch_buf_imgs:
            return
        embs = encode_image_batch(clip_model, clip_processor, batch_buf_imgs, device)
        for e, m in zip(embs, batch_buf_meta):
            all_embeddings.append(e)
            all_meta.append(m)
            done_urls.add(m["image_url"])
        batch_buf_imgs.clear()
        batch_buf_meta.clear()
        batches_since_ckpt += 1
        if batches_since_ckpt >= save_every:
            save_checkpoint(output_dir, all_embeddings, all_meta, done_urls)
            batches_since_ckpt = 0

    desc = "Load+encode" if image_cache_dir else "Download+encode"
    print(f"Starting ({num_workers} workers, batch_size={batch_size}) …")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(worker_fn, t): t for t in remaining_tasks}
        pbar = tqdm(as_completed(futures), total=len(futures), desc=desc, unit="img")
        for future in pbar:
            kb_url, title, img_url, img = future.result()
            if img is None:
                n_failed += 1
                pbar.set_postfix(failed=n_failed, buffered=len(batch_buf_imgs))
                continue

            batch_buf_imgs.append(img)
            batch_buf_meta.append({"kb_url": kb_url, "image_url": img_url, "title": title})

            if len(batch_buf_imgs) >= batch_size:
                flush_gpu_batch()
                pbar.set_postfix(
                    encoded=len(all_embeddings), failed=n_failed,
                    buffered=len(batch_buf_imgs),
                )

    flush_gpu_batch()   # final partial batch

    if not all_embeddings:
        print("ERROR: No images loaded. Check --image_cache_dir or network connectivity.")
        return

    # ── write final index ─────────────────────────────────────────────────────
    emb_file, meta_file = finalize(output_dir, all_embeddings, all_meta)

    n_relevant_indexed = sum(
        1 for m in all_meta if normalize_url(m["kb_url"]) in evidence_urls
    )
    print(
        f"\nIndex complete:"
        f"\n  Total embeddings : {len(all_embeddings)}"
        f"\n  Relevant indexed : {n_relevant_indexed}"
        f"\n  Download failures: {n_failed}"
        f"\n  Embeddings       : {emb_file}"
        f"\n  Metadata         : {meta_file}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build CLIP index from encyclopedic_kb_wiki.json"
    )
    parser.add_argument("--mode", default="image", choices=["image", "text"],
        help="'image': encode KB images (needs local cache or internet); "
             "'text': encode image captions with CLIP text encoder (no internet needed)")
    parser.add_argument("--kb_path",
        default="/mnt/data/lannth/mLAnR/M3-VQA/encyclopedic_kb_wiki.json")
    parser.add_argument("--questions",
        default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--output_dir",
        default="/mnt/data/lannth/mLAnR/results/clip_index")
    parser.add_argument("--num_distractors", type=int, default=10_000)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--max_images_per_entry", type=int, default=1,
        help="Max images (image mode) or captions (text mode) per KB entry")
    # image-mode only
    parser.add_argument("--batch_size", type=int, default=512,
        help="[image mode] Images per GPU encode call")
    parser.add_argument("--num_workers", type=int, default=64,
        help="[image mode] Parallel load/download threads")
    parser.add_argument("--save_every", type=int, default=20,
        help="[image mode] Checkpoint every N GPU batches")
    parser.add_argument("--image_cache_dir", default=None,
        help="[image mode] Load from local dir (from download_kb_images.py) "
             "instead of downloading from the internet")
    # text-mode only
    parser.add_argument("--text_batch_size", type=int, default=1024,
        help="[text mode] Captions per GPU encode call")
    parser.add_argument("--rebuild", action="store_true",
        help="Discard checkpoint and existing index; start fresh")
    args = parser.parse_args()

    if args.rebuild:
        for fname in (
            "clip_embeddings.npy", "clip_index_meta.json",
            "_ckpt_embeddings.npy", "_ckpt_meta.json", "_ckpt_done_urls.json",
        ):
            p = os.path.join(args.output_dir, fname)
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed {p}")

    emb_file  = os.path.join(args.output_dir, "clip_embeddings.npy")
    meta_file = os.path.join(args.output_dir, "clip_index_meta.json")
    if os.path.exists(emb_file) and os.path.exists(meta_file):
        print(
            f"Index already exists at {args.output_dir}.\n"
            "Pass --rebuild to overwrite."
        )
        return

    if args.mode == "text":
        build_text_index(
            kb_path                = args.kb_path,
            questions_path         = args.questions,
            output_dir             = args.output_dir,
            num_distractors        = args.num_distractors,
            clip_model_name        = args.clip_model,
            max_captions_per_entry = args.max_images_per_entry,
            batch_size             = args.text_batch_size,
        )
    else:
        build_index(
            kb_path              = args.kb_path,
            questions_path       = args.questions,
            output_dir           = args.output_dir,
            num_distractors      = args.num_distractors,
            clip_model_name      = args.clip_model,
            max_images_per_entry = args.max_images_per_entry,
            batch_size           = args.batch_size,
            num_workers          = args.num_workers,
            save_every           = args.save_every,
            image_cache_dir      = args.image_cache_dir,
        )


if __name__ == "__main__":
    main()
