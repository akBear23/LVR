"""
CLIP-box RAG Inference
======================
Two-stage pipeline:
  Stage 1  Retrieval   CLIP-box retrieval finds the top-K KB passages most
                        visually relevant to the image question.
  Stage 2  Generation  The retrieved passages are inserted as context; the
                        LVR model then generates a final answer.

Usage (from /mnt/data/lannth/mLAnR/lvr/):
    # Single box, top-5 passages
    python analysis/clip_box_rag_inference.py \\
        --model_path <ckpt> --steps 8 --top_k 5

    # Multi-entity retrieval (connected components)
    python analysis/clip_box_rag_inference.py \\
        --model_path <ckpt> --steps 8 \\
        --retrieval_mode clip_multibox --max_boxes 3 --box_fusion max --top_k 5

    # Per-step retrieval (NMS-deduped)
    python analysis/clip_box_rag_inference.py \\
        --model_path <ckpt> --steps 8 \\
        --retrieval_mode clip_stepbox  --max_boxes 3 --box_fusion max --top_k 5

    # Sanity-check on first 20 questions, save box images
    python analysis/clip_box_rag_inference.py \\
        --model_path <ckpt> --steps 8 --num_samples 20 --save_boxes
"""
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # project root → src.*
sys.path.insert(0, _HERE)                     # analysis/  → lvr_retrieval_eval

import re
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Importing lvr_retrieval_eval applies the GenerationConfig patch and all
# shared model / retrieval helpers automatically.
from lvr_retrieval_eval import (
    load_model,
    load_clip_model,
    build_pool,
    encode_passages_clip,
    get_box_and_clip_query,
    get_multi_box_clip_query,
    get_per_step_clip_query,
    retrieve,
    retrieve_multi_query,
    _save_box_viz,
)
from qwen_vl_utils import process_vision_info


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def format_rag_context(retrieved_passages, max_chars_per_passage=600):
    """
    Format the top-K retrieved passages as a numbered context block.
    Each passage is truncated to max_chars_per_passage characters.
    """
    blocks = []
    for i, p in enumerate(retrieved_passages, 1):
        text = p["text"]
        if len(text) > max_chars_per_passage:
            text = text[:max_chars_per_passage] + "…"
        blocks.append(f"[Document {i}]\n{text}")
    return "\n\n".join(blocks)


def build_rag_messages(question, context_text, img_path):
    """
    Build the message list for the RAG generation call.
    Image comes first so the model sees it before reading the documents.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant. "
                "Read the reference documents below and use them together with "
                "the image to answer the question concisely and accurately."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text",  "text": (
                    f"Reference Documents:\n{context_text}\n\n"
                    f"Question: {question}"
                )},
            ],
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# RAG generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_rag_answer(model, processor, messages,
                         steps, decoding_strategy,
                         max_new_tokens=256):
    """
    Run the LVR model with the RAG context prompt and return the decoded answer.
    LVR steps are retained so the model can visually ground while reading context.
    """
    text_fmt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_fmt], images=img_inputs, videos=vid_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            decoding_strategy=decoding_strategy,
            lvr_steps=[steps],
            return_dict_in_generate=True,
        )

    new_tokens = gen_out.sequences[0, input_len:]
    return processor.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    ).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, golds) -> bool:
    """True if the normalised prediction matches any normalised gold answer."""
    pred_norm = _normalize(pred)
    if isinstance(golds, list):
        return any(_normalize(str(g)) == pred_norm for g in golds)
    return _normalize(str(golds)) == pred_norm


def contain_match(pred: str, golds) -> bool:
    """
    True if any normalised gold answer appears as a substring of the
    normalised prediction (i.e. the answer is 'contained' in the output).
    Useful when the model generates a full sentence instead of a short phrase.
    """
    pred_norm = _normalize(pred)
    if isinstance(golds, list):
        return any(_normalize(str(g)) in pred_norm for g in golds)
    return _normalize(str(golds)) in pred_norm


def compute_aggregate_metrics(records):
    """Compute overall and per-hop EM and contain-EM from the records list."""
    em_by_hop      = {}
    contain_by_hop = {}
    for r in records:
        hop = r.get("hop", -1)
        if r["exact_match"] is not None:
            em_by_hop.setdefault(hop, []).append(r["exact_match"])
        if r["contain_match"] is not None:
            contain_by_hop.setdefault(hop, []).append(r["contain_match"])

    all_ems      = [v for vals in em_by_hop.values()      for v in vals]
    all_contains = [v for vals in contain_by_hop.values() for v in vals]

    by_hop = {}
    for hop in sorted(set(list(em_by_hop) + list(contain_by_hop))):
        entry = {"count": max(len(em_by_hop.get(hop, [])),
                              len(contain_by_hop.get(hop, [])))}
        if hop in em_by_hop:
            entry["em"]      = sum(em_by_hop[hop])      / len(em_by_hop[hop])
        if hop in contain_by_hop:
            entry["contain_em"] = sum(contain_by_hop[hop]) / len(contain_by_hop[hop])
        by_hop[str(hop)] = entry

    summary = {
        "overall_em":         sum(all_ems)      / len(all_ems)      if all_ems      else None,
        "overall_contain_em": sum(all_contains) / len(all_contains) if all_contains else None,
        "n_evaluated":        len(all_ems),
        "by_hop":             by_hop,
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_inference(
    model, processor,
    clip_model, clip_processor,
    questions, passages, passage_embeddings,
    image_dir,
    steps, decoding_strategy,
    retrieval_mode, top_k,
    box_threshold, box_padding,
    max_boxes, box_fusion, min_area, iou_threshold,
    max_new_tokens, max_chars_per_passage,
    save_boxes, viz_dir,
    out_path,
    debug_n=3,
):
    is_multi = retrieval_mode in ("clip_multibox", "clip_stepbox")
    records  = []

    for qi, q in enumerate(tqdm(questions, desc="RAG inference")):
        img_path = os.path.join(image_dir, q["image_id"])
        golds    = q.get("answers", q.get("answer", None))

        # ── Stage 1: retrieval ────────────────────────────────────────────────
        boxes                  = []
        retrieved_passages_meta = []

        if is_multi:
            if retrieval_mode == "clip_multibox":
                query_vecs, _, boxes = get_multi_box_clip_query(
                    model, processor, clip_model, clip_processor,
                    img_path, q["question"], steps=steps,
                    decoding_strategy=decoding_strategy,
                    box_threshold=box_threshold, box_padding=box_padding,
                    max_boxes=max_boxes, min_area=min_area,
                )
            else:   # clip_stepbox
                query_vecs, _, boxes = get_per_step_clip_query(
                    model, processor, clip_model, clip_processor,
                    img_path, q["question"], steps=steps,
                    decoding_strategy=decoding_strategy,
                    box_threshold=box_threshold, box_padding=box_padding,
                    max_boxes=max_boxes, iou_threshold=iou_threshold,
                )
            if query_vecs:
                top_idx, top_sims = retrieve_multi_query(
                    query_vecs, passage_embeddings, top_k=top_k, fusion=box_fusion)
                retrieved_passages_meta = [passages[i] for i in top_idx[:top_k]]
                top_sims_kept          = top_sims[:top_k].tolist()
            else:
                top_sims_kept = []

        else:   # clip_box
            clip_feat, _, box = get_box_and_clip_query(
                model, processor, clip_model, clip_processor,
                img_path, q["question"], steps=steps,
                decoding_strategy=decoding_strategy,
                box_threshold=box_threshold, box_padding=box_padding,
            )
            if box is not None:
                boxes = [box]
            if clip_feat is not None:
                top_idx, top_sims = retrieve(clip_feat, passage_embeddings, top_k=top_k)
                retrieved_passages_meta = [passages[i] for i in top_idx[:top_k]]
                top_sims_kept          = top_sims[:top_k].tolist()
            else:
                top_sims_kept = []

        # ── save box visualisation ────────────────────────────────────────────
        if save_boxes and boxes and viz_dir:
            safe_id = str(q.get("data_id", qi)).replace("/", "_")
            out_viz = os.path.join(viz_dir, retrieval_mode, f"q{qi:05d}_{safe_id}.jpg")
            _save_box_viz(img_path, boxes, out_viz,
                          question=q["question"], mode=retrieval_mode)

        # ── Stage 2: RAG generation ───────────────────────────────────────────
        if retrieved_passages_meta:
            context_text = format_rag_context(retrieved_passages_meta, max_chars_per_passage)
        else:
            context_text = "(No relevant documents retrieved.)"

        messages = build_rag_messages(q["question"], context_text, img_path)
        answer   = generate_rag_answer(
            model, processor, messages,
            steps=steps, decoding_strategy=decoding_strategy,
            max_new_tokens=max_new_tokens,
        )

        # ── metrics ───────────────────────────────────────────────────────────
        em      = exact_match(answer, golds)  if golds else None
        c_em    = contain_match(answer, golds) if golds else None

        record = {
            "data_id":          q.get("data_id"),
            "hop":              q.get("question_hop", -1),
            "entity_num":       q.get("entity_num"),
            "question":         q["question"],
            "gold_answers":     golds,
            "predicted_answer": answer,
            "exact_match":      em,
            "contain_match":    c_em,
            "rag_messages":     messages,
            "retrieval_mode":   retrieval_mode,
            "boxes":            boxes,
            "retrieved": [
                {
                    "rank":  rank,
                    "pid":   p["pid"],
                    "title": p["title"],
                    "sim":   float(top_sims_kept[rank - 1]) if rank <= len(top_sims_kept) else None,
                    "text_snippet": p["text"][:200],
                }
                for rank, p in enumerate(retrieved_passages_meta, 1)
            ],
        }
        records.append(record)

        # ── debug print ───────────────────────────────────────────────────────
        if qi < debug_n:
            print(f"\n{'─'*60}")
            print(f"[Q{qi}]  data_id={q.get('data_id')}  hop={q.get('question_hop')}")
            print(f"  question  : {q['question'][:120]}")
            print(f"  boxes     : {boxes}")
            print(f"  retrieved :")
            for r in record["retrieved"]:
                print(f"    [{r['rank']}] sim={r['sim']:.4f}  {r['title']}")
            print(f"  answer    : {answer[:200]}")
            if golds:
                print(f"  gold      : {golds}   EM={em}  contain={c_em}")

    # ── save results ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} records → {out_path}")

    summary = compute_aggregate_metrics(records)
    n = summary["n_evaluated"]
    if n:
        em_str  = f"{summary['overall_em']*100:.1f}%"      if summary["overall_em"]         is not None else "n/a"
        cem_str = f"{summary['overall_contain_em']*100:.1f}%" if summary["overall_contain_em"] is not None else "n/a"
        print(f"\nEM={em_str}  ContainEM={cem_str}  (n={n})")
        for hop, hm in summary["by_hop"].items():
            em_h  = f"{hm['em']*100:.1f}%"         if "em"         in hm else "n/a"
            cem_h = f"{hm['contain_em']*100:.1f}%"  if "contain_em" in hm else "n/a"
            print(f"  hop {hop}: EM={em_h}  ContainEM={cem_h}  (n={hm['count']})")

    summary_path = out_path.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    return records, summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CLIP-box RAG inference: retrieve → generate answer")

    # ── paths ──────────────────────────────────────────────────────────────────
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--kb_path",     default="/mnt/data/lannth/mLAnR/M3-VQA/encyclopedic_kb_wiki.json")
    parser.add_argument("--questions",   default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--image_dir",   default="/mnt/data/lannth/mLAnR/M3-VQA/images")
    parser.add_argument("--output_dir",  default="/mnt/data/lannth/mLAnR/results/rag_inference")

    # ── LVR generation settings ────────────────────────────────────────────────
    parser.add_argument("--steps",            type=int,   default=8)
    parser.add_argument("--decoding_strategy", default="steps",
                        choices=["steps", "latent"])
    parser.add_argument("--max_new_tokens",   type=int,   default=256,
                        help="Max tokens for the RAG answer generation")

    # ── retrieval settings ─────────────────────────────────────────────────────
    parser.add_argument("--retrieval_mode", default="clip_box",
                        choices=["clip_box", "clip_multibox", "clip_stepbox"])
    parser.add_argument("--clip_model",   default="openai/clip-vit-large-patch14")
    parser.add_argument("--clip_text_key", default="title", choices=["title", "text"],
                        help="Passage field encoded by CLIP text encoder")
    parser.add_argument("--top_k",         type=int,   default=5,
                        help="Number of passages to retrieve and put in context")
    parser.add_argument("--box_threshold", type=float, default=0.65)
    parser.add_argument("--box_padding",   type=int,   default=10)
    parser.add_argument("--num_distractors", type=int, default=50_000,
                        help="Distractor passages added to the retrieval pool")
    parser.add_argument("--encode_batch",  type=int,   default=256)

    # ── multi-box settings ─────────────────────────────────────────────────────
    parser.add_argument("--max_boxes",     type=int,   default=3)
    parser.add_argument("--box_fusion",    default="max", choices=["max", "mean", "sum"])
    parser.add_argument("--min_area",      type=int,   default=16)
    parser.add_argument("--iou_threshold", type=float, default=0.5)

    # ── context formatting ─────────────────────────────────────────────────────
    parser.add_argument("--max_chars_per_passage", type=int, default=600,
                        help="Truncate each retrieved passage to this many characters")

    # ── misc ───────────────────────────────────────────────────────────────────
    parser.add_argument("--num_samples",  type=int, default=None,
                        help="Evaluate on first N questions (None = all)")
    parser.add_argument("--save_boxes",   action="store_true",
                        help="Save annotated images showing the retrieved box regions")
    parser.add_argument("--debug_n",      type=int, default=3,
                        help="Print verbose output for the first N questions")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load questions ─────────────────────────────────────────────────────────
    print("Loading questions …")
    questions = [json.loads(l) for l in open(args.questions)]
    if args.num_samples:
        questions = questions[:args.num_samples]
    print(f"  {len(questions)} questions")

    # ── load KB and build passage pool ─────────────────────────────────────────
    print("Loading KB …")
    with open(args.kb_path) as f:
        kb = json.load(f)
    print(f"  {len(kb)} articles")

    print("Building passage pool …")
    all_questions = [json.loads(l) for l in open(args.questions)]
    passages, _ = build_pool(kb, all_questions, num_distractors=args.num_distractors)

    # ── load models ────────────────────────────────────────────────────────────
    print(f"\nLoading LVR model from {args.model_path} …")
    model, processor = load_model(args.model_path)
    print("LVR model loaded.\n")

    print(f"Loading CLIP model: {args.clip_model} …")
    clip_model, clip_processor = load_clip_model(args.clip_model)

    # ── encode passages with CLIP ──────────────────────────────────────────────
    cache_dir = os.path.join(
        args.output_dir, "passage_cache",
        f"clip_{args.clip_model.replace('/', '_')}_dist{args.num_distractors}",
    )
    print("Encoding passages with CLIP …")
    passage_embeddings = encode_passages_clip(
        clip_model, clip_processor, passages,
        batch_size=args.encode_batch,
        cache_dir=cache_dir,
        text_key=args.clip_text_key,
    )

    # ── output path ────────────────────────────────────────────────────────────
    mode_tag = args.retrieval_mode
    if args.retrieval_mode != "clip_box":
        mode_tag += f"_boxes{args.max_boxes}_{args.box_fusion}"
    out_name = (f"rag_{mode_tag}_{args.clip_text_key}_top{args.top_k}"
                f"_steps{args.steps:03d}_n{len(questions)}.json")
    out_path = os.path.join(args.output_dir, out_name)

    viz_dir = os.path.join(args.output_dir, "box_viz") if args.save_boxes else None

    # ── run inference ──────────────────────────────────────────────────────────
    run_rag_inference(
        model, processor,
        clip_model, clip_processor,
        questions, passages, passage_embeddings,
        image_dir=args.image_dir,
        steps=args.steps,
        decoding_strategy=args.decoding_strategy,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        box_threshold=args.box_threshold,
        box_padding=args.box_padding,
        max_boxes=args.max_boxes,
        box_fusion=args.box_fusion,
        min_area=args.min_area,
        iou_threshold=args.iou_threshold,
        max_new_tokens=args.max_new_tokens,
        max_chars_per_passage=args.max_chars_per_passage,
        save_boxes=args.save_boxes,
        viz_dir=viz_dir,
        out_path=out_path,
        debug_n=args.debug_n,
    )


if __name__ == "__main__":
    main()
