"""
GRPO trainer with retrieval-augmented rollout for LVR+M3-VQA.

Subclasses QwenGRPOTrainer and overrides:
  _generate_and_score_completions: uses generate_with_retrieval() instead of
      model.generate() to inject KB passages into each rollout.
  _compute_loss: extends the LVR token mask to also exclude document tokens
      from the policy gradient loss.

The GRPO advantage computation (group reward normalization) and PPO-clipped
surrogate are inherited unchanged from QwenGRPOTrainer.
"""

import sys
import os
import re
import json
import warnings
from typing import Any, Union, Optional

import torch
import numpy as np

_GRPO_DIR = os.path.dirname(os.path.abspath(__file__))
_LVR_ROOT = os.path.abspath(os.path.join(_GRPO_DIR, ".."))
sys.path.insert(0, _LVR_ROOT)
sys.path.insert(1, os.path.join(_LVR_ROOT, "src"))
if _GRPO_DIR not in sys.path:
    sys.path.append(_GRPO_DIR)

# Import grpo_trainer directly to avoid trainer/__init__.py side-effects
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "grpo_trainer", os.path.join(_LVR_ROOT, "src", "trainer", "grpo_trainer.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
QwenGRPOTrainer = _mod.QwenGRPOTrainer

from constants import MULTIMODAL_KEYWORDS

from generation import generate_with_retrieval, build_doc_token_mask
from retrieval import PassageIndex
from reward_funcs import accuracy_reward, format_reward, retrieval_reward

from accelerate.utils import gather
from trl.trainer.utils import selective_log_softmax
from trl.data_utils import is_conversational, maybe_apply_chat_template, apply_chat_template
from trl.extras.profiling import profiling_context
from qwen_vl_utils import process_vision_info


class RetrievalGRPOTrainer(QwenGRPOTrainer):
    """GRPO trainer that augments each rollout with KB retrieval."""

    def __init__(
        self,
        passage_index: PassageIndex,
        lvr_steps:     int   = 8,
        top_k:         int   = 3,
        box_threshold: float = 0.65,
        box_padding:   int   = 10,
        retrieval_reward_weight: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.passage_index  = passage_index
        self.lvr_steps      = lvr_steps
        self.top_k          = top_k
        self.box_threshold  = box_threshold
        self.box_padding    = box_padding
        self.retrieval_reward_weight = retrieval_reward_weight

    # ─────────────────────────────────────────────────────────────────────────
    # Override: rollout + scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode   = "eval" if self.control.should_evaluate else "train"

        prompts       = [x["prompt"]       for x in inputs]
        evidence_urls = [x.get("evidence_urls", []) for x in inputs]

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # ── Retrieve image paths from prompt for generate_with_retrieval ─────
        # The prompt is a list of message dicts; pull the image path from the
        # first image content block.
        image_paths = []
        questions   = []
        for prompt_msgs in prompts:
            img_path = None
            question = ""
            for msg in prompt_msgs:
                if msg["role"] != "user":
                    continue
                for block in (msg["content"] if isinstance(msg["content"], list) else []):
                    if isinstance(block, dict):
                        if block.get("type") == "image":
                            img_path = block.get("image", "")
                        elif block.get("type") == "text":
                            question = block.get("text", "")
            image_paths.append(img_path)
            questions.append(question)

        # ── Run retrieval-augmented rollouts ──────────────────────────────────
        # The ref_model is not needed during generation — move it to CPU to
        # free ~14 GB of GPU for the two model.generate() calls per sample.
        _ref_model_on_cpu = False
        if self.ref_model is not None:
            _ref_device = next(iter(self.ref_model.parameters())).device
            if _ref_device.type != "cpu":
                self.ref_model.cpu()
                torch.cuda.empty_cache()
                _ref_model_on_cpu = True

        rollout_results = []
        unwrapped = self.accelerator.unwrap_model(self.model_wrapped)

        # Gradient checkpointing is incompatible with use_cache during generation:
        # it forces past_key_values=None but _prepare_cache_for_generation still
        # inserts a DynamicCache object, causing prepare_inputs_for_generation to
        # slice input_ids to just the last token → image-token count mismatch crash.
        # Generation is inference-only (torch.no_grad), so GC gives no benefit here.
        _gc_was_enabled = getattr(unwrapped, "is_gradient_checkpointing", False)
        if _gc_was_enabled:
            unwrapped.gradient_checkpointing_disable()

        for img_path, question in zip(image_paths, questions):
            try:
                result = generate_with_retrieval(
                    model          = unwrapped,
                    processor      = self.processing_class,
                    passage_index  = self.passage_index,
                    image_path     = img_path,
                    question       = question,
                    lvr_steps      = self.lvr_steps,
                    top_k          = self.top_k,
                    box_threshold  = self.box_threshold,
                    box_padding    = self.box_padding,
                    max_answer_tokens = self.args.max_completion_length,
                    device         = str(device),
                )
            except Exception as e:
                warnings.warn(f"generate_with_retrieval failed: {e}")
                result = {
                    "completion_text":    "<answer>unknown</answer>",
                    "answer":             "unknown",
                    "retrieved_passages": [],
                    "doc_span_chars":     (0, 0),
                    "lvr_text":           "",
                    "answer_text":        "<answer>unknown</answer>",
                }
            rollout_results.append(result)
            torch.cuda.empty_cache()

        # Re-enable gradient checkpointing for the training forward/backward pass.
        if _gc_was_enabled:
            unwrapped.gradient_checkpointing_enable()

        # Restore ref_model to GPU for log-prob computation
        if _ref_model_on_cpu:
            self.ref_model.to(_ref_device)
            torch.cuda.empty_cache()

        # ── Tokenise full completions for log-prob computation ────────────────
        completion_texts = [r["completion_text"] for r in rollout_results]

        # Build prompt inputs (needed for LVR teacher-forcing if lvr_steps > 0)
        image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)
        prompt_inputs = self.processing_class(
            text=prompts_text, images=image_inputs, videos=video_inputs,
            padding=True, padding_side="left", return_tensors="pt", **video_kwargs,
        )
        prompt_inputs   = super()._prepare_inputs(prompt_inputs)
        prompt_ids  = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs["attention_mask"].to(device)

        if self.max_prompt_length is not None:
            prompt_ids  = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Tokenise completions separately, then concat
        completion_enc = self.processing_class.tokenizer(
            completion_texts, padding=True, padding_side="right",
            add_special_tokens=False, return_tensors="pt",
        )
        completion_ids  = completion_enc["input_ids"].to(device)
        completion_mask = completion_enc["attention_mask"].to(device)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # ── Build combined mask: LVR tokens + doc tokens excluded ─────────────
        lvr_start_id = unwrapped.config.lvr_start_id
        lvr_end_id   = unwrapped.config.lvr_end_id

        # LVR mask (existing logic from QwenGRPOTrainer)
        lvr_mask = torch.ones_like(prompt_completion_ids, dtype=torch.bool)
        for b in range(prompt_completion_ids.size(0)):
            active = False
            for t in range(prompt_completion_ids.size(1)):
                tok = prompt_completion_ids[b, t].item()
                if tok == lvr_start_id:
                    active = True
                elif tok == lvr_end_id:
                    active = False
                if active:
                    lvr_mask[b, t] = False

        # Doc mask (new: exclude retrieved document tokens)
        P = prompt_ids.size(1)
        for b, result in enumerate(rollout_results):
            doc_mask_b = build_doc_token_mask(
                completion_ids[b],
                result["doc_span_chars"],
                result["completion_text"],
                self.processing_class,
            )
            lvr_mask[b, P:P + doc_mask_b.size(0)] &= doc_mask_b.to(device)

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        final_mask     = attention_mask.bool() & lvr_mask

        logits_to_keep = completion_ids.size(1)
        batch_size     = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        # _prepare_inputs in grpo_trainer is a no-op, so move tensors to GPU explicitly here.
        multimodal_inputs = {
            k: (prompt_inputs[k].to(device) if isinstance(prompt_inputs[k], torch.Tensor) else prompt_inputs[k])
            for k in MULTIMODAL_KEYWORDS if k in prompt_inputs
        }

        # ── Log-probs (old / ref) under no_grad ──────────────────────────────
        with torch.no_grad():
            old_per_token_logps = None
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask,
                    logits_to_keep, batch_size, **multimodal_inputs,
                )
                old_per_token_logps = old_per_token_logps * final_mask[:, -logits_to_keep:]

            ref_per_token_logps = None
            if self.beta != 0.0 and self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask,
                    logits_to_keep, batch_size, **multimodal_inputs,
                )
                ref_per_token_logps = ref_per_token_logps * final_mask[:, -logits_to_keep:]

        # Move ref_model back to CPU now that its log-probs are stored as tensors.
        # This frees ~14 GB of GPU memory before _compute_loss runs the backward pass.
        if _ref_model_on_cpu and self.ref_model is not None:
            self.ref_model.cpu()
            torch.cuda.empty_cache()

        # ── Rewards ───────────────────────────────────────────────────────────
        # Wrap completions in the format expected by reward_funcs
        completions = [[{"role": "assistant", "content": r["completion_text"]}]
                       for r in rollout_results]

        # accuracy (F1)
        acc_rewards = accuracy_reward(completions, [x["assistant"] for x in inputs])
        # format
        fmt_rewards = format_reward(completions)
        # retrieval hit (optional bonus)
        ret_rewards = retrieval_reward(completions, evidence_urls=evidence_urls)

        rewards_tensor = torch.tensor(
            [a + 0.1 * f + self.retrieval_reward_weight * r
             for a, f, r in zip(acc_rewards, fmt_rewards, ret_rewards)],
            dtype=torch.float32, device=device,
        )
        rewards_tensor = gather(rewards_tensor)

        # ── GRPO advantage normalization ──────────────────────────────────────
        G = self.num_generations
        mean_r = rewards_tensor.view(-1, G).mean(dim=1).repeat_interleave(G)
        std_r  = rewards_tensor.view(-1, G).std(dim=1).repeat_interleave(G)
        advantages = rewards_tensor - mean_r
        if self.scale_rewards:
            advantages = advantages / (std_r + 1e-4)

        # ── Package outputs for _compute_loss ─────────────────────────────────
        return {
            "prompt_ids":           prompt_ids,
            "prompt_mask":          prompt_mask,
            "completion_ids":       completion_ids,
            "completion_mask":      completion_mask,
            "final_mask":           final_mask[:, -logits_to_keep:],
            "multimodal_inputs":    multimodal_inputs,
            "old_per_token_logps":  old_per_token_logps,
            "ref_per_token_logps":  ref_per_token_logps,
            "advantages":           advantages,
            # Store for logging
            "rewards":              rewards_tensor,
            "acc_rewards":          torch.tensor(acc_rewards, device=device),
            "ret_rewards":          torch.tensor(ret_rewards, device=device),
        }
