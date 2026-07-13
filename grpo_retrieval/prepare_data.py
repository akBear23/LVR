"""
Convert M3-VQA quesions.jsonl → GRPO training JSON.

Output format expected by GRPODataset:
    [
      {
        "image": "/abs/path/to/img.jpg",
        "conversations": [
          {"from": "human", "value": "<question text>"},
          {"from": "gpt",   "value": "<json list of acceptable answer strings>"}
        ],
        "evidence_urls": ["https://...", ...]   # kept for retrieval_reward
      },
      ...
    ]

Usage:
    python prepare_data.py \
        --questions /mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl \
        --image_dir /mnt/data/lannth/mLAnR/M3-VQA/images \
        --output_train /mnt/data/lannth/mLAnR/lvr/grpo_retrieval/m3vqa_train.json \
        --output_eval  /mnt/data/lannth/mLAnR/lvr/grpo_retrieval/m3vqa_eval.json \
        --eval_frac 0.05
"""

import argparse
import json
import os
import random


INSTRUCTION = (
    "Look at the image carefully and answer the question. "
    "Put your final answer inside <answer> </answer> tags."
)


def flatten_answer_evals(answer_evals: list) -> list[str]:
    """Flatten nested answer_evals into a deduplicated list of acceptable answers.

    answer_evals is a list of lists, e.g.:
        [["Ragdoll"], ["Maine Coon", "American longhair"]]
    Some inner elements may themselves be empty lists (malformed data) — skip those.
    """
    seen = set()
    flat = []
    for group in answer_evals:
        for ans in group:
            if not isinstance(ans, str):
                continue  # skip nested lists / non-strings
            if ans not in seen:
                seen.add(ans)
                flat.append(ans)
    return flat


def convert(questions_path: str, image_dir: str) -> list[dict]:
    with open(questions_path) as f:
        questions = [json.loads(l) for l in f]

    records = []
    for q in questions:
        image_path = os.path.join(image_dir, q["image_id"])
        if not os.path.exists(image_path):
            continue

        golds = flatten_answer_evals(q["answer_evals"])
        if not golds:
            golds = q["answers"]  # fallback

        record = {
            "image": image_path,
            "conversations": [
                {"from": "human", "value": f"{INSTRUCTION}\n\n{q['question']}"},
                {"from": "gpt",   "value": json.dumps(golds)},
            ],
            "evidence_urls": q.get("evidence_urls", []),
            "question_hop": q.get("question_hop", -1),
            "data_id": q["data_id"],
        }
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions",    default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--image_dir",   default="/mnt/data/lannth/mLAnR/M3-VQA/images")
    parser.add_argument("--output_train", default="/mnt/data/lannth/mLAnR/lvr/grpo_retrieval/m3vqa_train.json")
    parser.add_argument("--output_eval",  default="/mnt/data/lannth/mLAnR/lvr/grpo_retrieval/m3vqa_eval.json")
    parser.add_argument("--eval_frac",   type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print(f"Loading {args.questions} …")
    records = convert(args.questions, args.image_dir)
    print(f"  {len(records)} valid questions (images found)")

    random.seed(args.seed)
    random.shuffle(records)
    n_eval = max(1, int(len(records) * args.eval_frac))
    eval_records  = records[:n_eval]
    train_records = records[n_eval:]

    os.makedirs(os.path.dirname(args.output_train), exist_ok=True)

    with open(args.output_train, "w") as f:
        json.dump(train_records, f, indent=2)
    print(f"Train: {len(train_records)} → {args.output_train}")

    with open(args.output_eval, "w") as f:
        json.dump(eval_records, f, indent=2)
    print(f"Eval:  {len(eval_records)} → {args.output_eval}")


if __name__ == "__main__":
    main()
