"""
Reward functions for M3-VQA GRPO training.

accuracy_reward: token-level F1 against all acceptable gold answers (like AutoRefine).
format_reward:   checks that the output has the expected LVR + <answer> structure.
retrieval_reward: bonus for retrieving the gold evidence article (optional).
"""

import os
import re
import json
import string
from collections import Counter
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def _f1(pred: str, gold: str) -> float:
    p_toks = Counter(_normalize(pred).split())
    g_toks = Counter(_normalize(gold).split())
    common = sum((p_toks & g_toks).values())
    if common == 0:
        return 0.0
    prec = common / sum(p_toks.values())
    rec  = common / sum(g_toks.values())
    return 2 * prec * rec / (prec + rec)


def _extract_answer(content: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    return m.group(1).strip() if m else content.strip()


def _gold_answers(asst_content) -> list[str]:
    """Parse gold answers from assistant content field.

    prepare_data.py stores them as a JSON list; also handles plain strings.
    """
    if isinstance(asst_content, list):
        return [str(a) for a in asst_content]
    try:
        parsed = json.loads(asst_content)
        if isinstance(parsed, list):
            return [str(a) for a in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [str(asst_content)]


# ─────────────────────────────────────────────────────────────────────────────
# Public reward functions
# ─────────────────────────────────────────────────────────────────────────────

def accuracy_reward(completions, assistant, **kwargs):
    """Max token-level F1 against all acceptable gold answers for M3-VQA.

    Gold answers come from `answer_evals` (flattened list stored as JSON in the
    assistant turn).  A prediction of "Caribbean" against golds
    ["Caribbean", "the Caribbean"] scores 1.0 on the first gold.
    """
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for completion, asst in zip(completions, assistant):
        content = completion[0]["content"]
        pred    = _extract_answer(content)
        golds   = _gold_answers(asst["content"])
        reward  = max(_f1(pred, g) for g in golds) if golds else 0.0
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "/tmp/grpo_retrieval_debug.log")
            with open(log_path, "a") as f:
                f.write(f"--- {current_time}  accuracy={reward:.3f} ---\n")
                f.write(f"  pred : {pred}\n")
                f.write(f"  golds: {golds}\n")
    return rewards


def format_reward(completions, **kwargs):
    """1.0 if completion has <|lvr_start|>...<|lvr_end|>...<answer>...</answer>.

    Allows for optional <documents>...</documents> between the LVR span and the
    answer (injected by the retrieval loop).
    """
    pattern = (
        r"<\|lvr_start\|>.*?<\|lvr_end\|>"   # LVR latent block
        r".*?"                                 # optional documents
        r"<answer>.*?</answer>"                # final answer
    )
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        rewards.append(1.0 if re.search(pattern, content, re.DOTALL) else 0.0)
    return rewards


def retrieval_reward(completions, evidence_urls=None, **kwargs):
    """Bonus reward if retrieved documents contain the gold evidence URL.

    The retrieval loop stores retrieved URLs in the completion under a
    <!-- retrieved: url1 | url2 --> HTML comment so this function can parse
    them without extra data structures.

    Args:
        evidence_urls: list of lists — per-sample list of gold evidence URLs.
    """
    if evidence_urls is None:
        return [0.0] * len(completions)

    rewards = []
    for completion, gold_urls in zip(completions, evidence_urls):
        content = completion[0]["content"]
        m = re.search(r"<!--\s*retrieved:\s*(.*?)\s*-->", content, re.DOTALL)
        if m is None:
            rewards.append(0.0)
            continue
        retrieved_urls = [u.strip() for u in m.group(1).split("|")]
        hit = any(g in retrieved_urls for g in gold_urls)
        rewards.append(1.0 if hit else 0.0)
    return rewards
