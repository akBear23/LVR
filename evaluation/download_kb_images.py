"""
Download KB images needed for the CLIP index to a local directory.

Run this on any machine with internet access, then copy the output directory
to the GPU server before running build_clip_kb_index.py --image_cache_dir.

What is downloaded
------------------
  - First image of every KB entry whose URL appears in M3-VQA evidence_urls
    (the "relevant" pool — must be indexed for correct retrieval).
  - First image of --num_distractors randomly-sampled other KB entries.
  Together: typically ~15 000 images.

Output
------
  <image_cache_dir>/
      <md5_of_image_url>.jpg      — downloaded images
      download_manifest.json      — {image_url: {"path": ..., "kb_url": ...,
                                                   "title": ..., "ok": bool}}

Resume
------
  Re-running the same command skips images already in download_manifest.json
  (whether they succeeded or failed).  Use --retry_failed to re-attempt
  previously failed URLs.

Usage
-----
  python evaluation/download_kb_images.py \\
      --kb_path   /path/to/encyclopedic_kb_wiki.json \\
      --questions /path/to/quesions.jsonl \\
      --image_cache_dir /path/to/kb_images \\
      --num_distractors 10000 \\
      --num_workers 64
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, argparse, random, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO

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
    """Stable local filename derived from the image URL (md5 hex + .jpg)."""
    return hashlib.md5(image_url.encode()).hexdigest() + ".jpg"


def load_evidence_urls(questions_path: str) -> set:
    urls = set()
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            for url in item.get("evidence_urls", []):
                urls.add(normalize_url(url))
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# Download worker
# ─────────────────────────────────────────────────────────────────────────────

def download_one(task: dict, image_cache_dir: str, timeout: int = 20) -> dict:
    """
    Download a single image.  Returns the manifest entry dict with 'ok' set.
    task keys: image_url, kb_url, title
    """
    img_url  = task["image_url"]
    out_path = os.path.join(image_cache_dir, url_to_filename(img_url))

    entry = {**task, "path": out_path, "ok": False}

    try:
        r = requests.get(img_url, headers=_HEADERS, timeout=timeout, stream=True)
        if r.status_code != 200:
            entry["error"] = f"HTTP {r.status_code}"
            return entry

        raw = r.content
        if not raw:
            entry["error"] = "empty response"
            return entry

        # Verify it's a valid image and save as JPEG
        img = Image.open(BytesIO(raw)).convert("RGB")
        img.save(out_path, format="JPEG", quality=90)
        entry["ok"] = True
    except Exception as e:
        entry["error"] = str(e)

    return entry


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download KB images needed for the CLIP index"
    )
    parser.add_argument("--kb_path",
        default="/mnt/data/lannth/mLAnR/M3-VQA/encyclopedic_kb_wiki.json")
    parser.add_argument("--questions",
        default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--image_cache_dir",
        default="/mnt/data/lannth/mLAnR/results/kb_images",
        help="Directory where downloaded images will be saved")
    parser.add_argument("--num_distractors", type=int, default=10_000,
        help="Number of random non-evidence KB entries to also download")
    parser.add_argument("--max_images_per_entry", type=int, default=1,
        help="Maximum images per KB entry (1 = first image only)")
    parser.add_argument("--num_workers", type=int, default=64,
        help="Parallel download threads")
    parser.add_argument("--timeout", type=int, default=20,
        help="HTTP timeout per request in seconds")
    parser.add_argument("--retry_failed", action="store_true",
        help="Re-attempt previously failed URLs (skips only successes by default)")
    args = parser.parse_args()

    os.makedirs(args.image_cache_dir, exist_ok=True)
    manifest_path = os.path.join(args.image_cache_dir, "download_manifest.json")

    # ── load existing manifest for resume ─────────────────────────────────────
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest: dict = json.load(f)
        print(f"Loaded existing manifest: {len(manifest)} entries "
              f"({sum(v['ok'] for v in manifest.values())} succeeded)")
    else:
        manifest: dict = {}

    skip_urls = {
        url for url, v in manifest.items()
        if v["ok"] or not args.retry_failed
    }

    # ── collect evidence URLs ─────────────────────────────────────────────────
    print("Collecting evidence URLs from questions …")
    evidence_urls = load_evidence_urls(args.questions)
    print(f"  {len(evidence_urls)} unique evidence URLs")

    # ── load KB ───────────────────────────────────────────────────────────────
    print(f"Loading KB (may take ~60 s) …")
    with open(args.kb_path) as f:
        kb = json.load(f)
    print(f"  {len(kb):,} KB entries")

    # ── partition and collect download tasks ──────────────────────────────────
    relevant_tasks, distractor_tasks = [], []
    for url, entry in kb.items():
        img_urls = entry.get("image_urls", [])[:args.max_images_per_entry]
        if not img_urls:
            continue
        tasks = [
            {"image_url": iu, "kb_url": url, "title": entry.get("title", "")}
            for iu in img_urls
        ]
        if normalize_url(url) in evidence_urls:
            relevant_tasks.extend(tasks)
        else:
            distractor_tasks.extend(tasks)

    random.seed(42)
    random.shuffle(distractor_tasks)
    all_tasks = relevant_tasks + distractor_tasks[:args.num_distractors]

    # filter out already-handled URLs
    pending = [t for t in all_tasks if t["image_url"] not in skip_urls]
    already_done = len(all_tasks) - len(pending)

    print(
        f"  {len(relevant_tasks)} relevant tasks"
        f" + {min(len(distractor_tasks), args.num_distractors)} distractor tasks"
        f" = {len(all_tasks)} total"
    )
    print(f"  {already_done} already in manifest → {len(pending)} to download\n")

    if not pending:
        n_ok = sum(1 for url in (t["image_url"] for t in all_tasks)
                   if manifest.get(url, {}).get("ok"))
        print(f"Nothing to do. {n_ok}/{len(all_tasks)} images successfully downloaded.")
        return

    # ── parallel download ─────────────────────────────────────────────────────
    n_ok = n_fail = 0

    def save_manifest():
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

    print(f"Downloading {len(pending)} images with {args.num_workers} workers …")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(download_one, t, args.image_cache_dir, args.timeout): t
            for t in pending
        }
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="Downloading", unit="img")
        for i, future in enumerate(pbar, 1):
            result = future.result()
            img_url = result["image_url"]
            manifest[img_url] = result
            if result["ok"]:
                n_ok += 1
            else:
                n_fail += 1
            pbar.set_postfix(ok=n_ok, failed=n_fail)

            # save manifest every 500 completions
            if i % 500 == 0:
                save_manifest()

    save_manifest()

    # ── summary ───────────────────────────────────────────────────────────────
    total_ok = sum(v["ok"] for v in manifest.values())
    print(
        f"\nDownload complete:"
        f"\n  This run : {n_ok} succeeded, {n_fail} failed"
        f"\n  Total OK : {total_ok} images in {args.image_cache_dir}"
        f"\n  Manifest : {manifest_path}"
    )
    if n_fail:
        print(
            f"\n  Re-run with --retry_failed to attempt the {n_fail} failures again."
        )


if __name__ == "__main__":
    main()
