"""
Retrieval-augmented rollout generation for LVR+GRPO.

Generation loop (one sample):
  1. Run LVR model.generate() → capture h_t at each <|lvr_latent_end|> position.
  2. Use h_t + image patch embeddings → CLIP bbox query → retrieve top-k passages.
     OR: parse <search>query</search> text token for text-based retrieval (AutoRefine style).
  3. Format retrieved passages as <documents>...</documents>.
  4. Append docs to input and run a second generation pass for the final answer.
  5. Return the full completion string + metadata for reward computation.

The two-phase design keeps the first LVR generation under no_grad (for policy
rollout) and the second answer generation also under no_grad.  Gradients are
computed later by the GRPO trainer via teacher-forcing over the saved sequence.
"""

import re
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional

_GRPO_DIR = os.path.dirname(os.path.abspath(__file__))
_LVR_ROOT = os.path.abspath(os.path.join(_GRPO_DIR, ".."))
sys.path.insert(0, _LVR_ROOT)
sys.path.insert(1, os.path.join(_LVR_ROOT, "src"))
if _GRPO_DIR not in sys.path:
    sys.path.append(_GRPO_DIR)

from retrieval import PassageIndex


SPATIAL_MERGE = 2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_docs(passages: list[dict]) -> str:
    """Format retrieved passages as a <documents> block, like AutoRefine."""
    parts = []
    for p in passages:
        parts.append(f"[{p['title']}]\n{p['text'][:400]}")   # cap per-passage length
    return "<documents>\n" + "\n\n".join(parts) + "\n</documents>"


def _embed_urls_comment(passages: list[dict]) -> str:
    """Embed retrieved URLs as an HTML comment for retrieval_reward parsing."""
    urls = [p["url"] for p in passages]
    return f"<!-- retrieved: {' | '.join(urls)} -->"


def _extract_search_query(text: str) -> Optional[str]:
    """Parse <search>query</search> from generated text (AutoRefine-style)."""
    m = re.search(r"<search>(.*?)</search>", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Core rollout function
# ─────────────────────────────────────────────────────────────────────────────

def generate_with_retrieval(
    model,
    processor,
    passage_index: PassageIndex,
    image_path:    str,
    question:      str,
    lvr_steps:     int  = 8,
    top_k:         int  = 3,
    box_threshold: float = 0.65,
    box_padding:   int   = 10,
    max_answer_tokens: int = 256,
    decoding_strategy: str = "steps",
    device:            str = "cuda",
) -> dict:
    """
    Full retrieval-augmented rollout for one question.

    Returns:
        completion_text: str   — everything the model generated (for reward + loss)
        answer:          str   — extracted <answer>...</answer> content
        retrieved_passages: list[dict] — passages injected into context
        doc_span:        (int, int)    — (start, end) char offsets of <documents> block
                                         within completion_text; used for loss masking
    """
    image = Image.open(image_path).convert("RGB")

    # ── Phase 1: LVR generation ───────────────────────────────────────────────
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": question},
    ]}]
    text_fmt = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_fmt], images=img_inputs, videos=vid_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    input_len        = inputs.input_ids.shape[1]
    lvr_end_id       = processor.tokenizer.convert_tokens_to_ids("<|lvr_latent_end|>")
    lvr_start_id     = processor.tokenizer.convert_tokens_to_ids("<|lvr_start|>")
    search_end_str   = "</search>"

    # Capture h_t at each <|lvr_latent_end|>
    captured_h = []   # list of (D,) numpy arrays

    def _hook(module, inp, output):
        if (hasattr(output, "last_position_hidden_state")
                and output.last_position_hidden_state is not None):
            captured_h.append(
                output.last_position_hidden_state[0].float().detach().cpu().numpy()
            )

    handle = model.register_forward_hook(_hook)

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=512,
            decoding_strategy=decoding_strategy,
            lvr_steps=[lvr_steps],
            return_dict_in_generate=True,
        )
    handle.remove()

    generated_ids  = gen_out.sequences[0, input_len:]
    lvr_text       = processor.decode(generated_ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)

    # Check for text-search query (AutoRefine style)
    search_query = _extract_search_query(lvr_text)

    # ── Phase 2: Retrieval ────────────────────────────────────────────────────
    retrieved = []
    if search_query:
        # Text retrieval
        retrieved = passage_index.retrieve_by_text(search_query, top_k=top_k)
    elif captured_h:
        # Visual retrieval via LVR h_t → CLIP bbox
        with torch.no_grad():
            image_embeds_list = model.model.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
        image_embeds = torch.cat(image_embeds_list, dim=0).float().cpu().numpy()
        thw = tuple(inputs["image_grid_thw"][0].tolist())  # (T, H, W)

        retrieved = passage_index.retrieve_by_image(
            h_list        = captured_h,
            image_embeds  = image_embeds,
            thw           = thw,
            image         = image,
            top_k         = top_k,
            box_threshold = box_threshold,
            box_padding   = box_padding,
        )

    # ── Phase 3: Second pass — answer generation with retrieved docs ──────────
    if retrieved:
        doc_block = _format_docs(retrieved)
        url_comment = _embed_urls_comment(retrieved)
        # Inject docs after LVR latent block, before asking for answer
        doc_injection = f"\n{doc_block}\n{url_comment}\n"
    else:
        doc_injection = ""

    # Build phase-2 prompt: original prompt + LVR output + docs
    phase2_text = text_fmt + lvr_text + doc_injection
    inputs2 = processor(
        text=[phase2_text], return_tensors="pt", padding=True,
    ).to(device)

    with torch.no_grad():
        ans_out = model.generate(
            **inputs2,
            max_new_tokens=max_answer_tokens,
            do_sample=False,
            decoding_strategy="steps",
            lvr_steps=[0],
        )
    answer_ids  = ans_out[0, inputs2.input_ids.shape[1]:]
    answer_text = processor.decode(answer_ids, skip_special_tokens=True).strip()

    # Ensure answer is wrapped in <answer> tags
    if "<answer>" not in answer_text:
        answer_text = f"<answer>{answer_text}</answer>"

    # ── Assemble full completion ──────────────────────────────────────────────
    doc_start = len(lvr_text)
    completion_text = lvr_text + doc_injection + answer_text
    doc_end   = doc_start + len(doc_injection)

    # Extract final answer string
    m = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)
    answer = m.group(1).strip() if m else answer_text.strip()

    return {
        "completion_text":    completion_text,
        "answer":             answer,
        "retrieved_passages": retrieved,
        "doc_span_chars":     (doc_start, doc_end),   # char offsets in completion_text
        "lvr_text":           lvr_text,
        "answer_text":        answer_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Token-level doc mask (for loss computation)
# ─────────────────────────────────────────────────────────────────────────────

def build_doc_token_mask(
    completion_ids: torch.LongTensor,   # (C,) token IDs of completion
    doc_span_chars: tuple[int, int],    # char offsets returned by generate_with_retrieval
    completion_text: str,               # full completion string (for char → token mapping)
    processor,
) -> torch.BoolTensor:
    """
    Returns a boolean mask of shape (C,) where True means "model-generated token"
    and False means "injected document token" (should be excluded from policy loss).

    We use a simple heuristic: encode the doc text alone and count its tokens,
    then blank out that span in the completion.
    """
    doc_start_char, doc_end_char = doc_span_chars
    if doc_start_char == doc_end_char:
        return torch.ones(len(completion_ids), dtype=torch.bool)

    # How many tokens does the pre-doc text occupy?
    pre_doc  = completion_text[:doc_start_char]
    doc_text = completion_text[doc_start_char:doc_end_char]

    pre_ids = processor.tokenizer(pre_doc,  add_special_tokens=False).input_ids
    doc_ids = processor.tokenizer(doc_text, add_special_tokens=False).input_ids

    mask = torch.ones(len(completion_ids), dtype=torch.bool)
    start = len(pre_ids)
    end   = start + len(doc_ids)
    mask[start:end] = False
    return mask
