"""
Evaluate LVR-7B on M3-VQA benchmark.

Usage:
    cd /mnt/data/lannth/mLAnR/lvr
    python evaluation/evaluate_m3vqa.py \
        --model_path /mnt/data/lannth/mLAnR/checkpoints/LVR-7B \
        --questions /mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl \
        --image_dir /mnt/data/lannth/mLAnR/M3-VQA/images \
        --output_dir /mnt/data/lannth/mLAnR/results/m3vqa \
        --steps 8
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig
from transformers.generation.configuration_utils import GenerationConfig

# Patch: some transformers versions expect nested config objects to have .to_dict(),
# but the model saves them as plain dicts. Wrap any such dict before instantiation.
_orig_from_model_config = GenerationConfig.from_model_config.__func__

def _patched_from_model_config(cls, model_config):
    for attr in ("decoder", "encoder"):
        val = getattr(model_config, attr, None)
        if isinstance(val, dict):
            d = val
            setattr(model_config, attr, type("_DictConfig", (), {"to_dict": lambda self, _d=d: _d})())
    return _orig_from_model_config(cls, model_config)

GenerationConfig.from_model_config = classmethod(_patched_from_model_config)

from src.model.qwen_lvr_model import QwenWithLVR
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from qwen_vl_utils import process_vision_info


def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True, lvr_head=config.lvr_head)

    model = QwenWithLVR.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def run_inference(model, processor, img_path, question, steps, decoding_strategy="steps"):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    text_formatted = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_formatted],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            decoding_strategy=decoding_strategy,
            lvr_steps=[steps],
        )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return output_text[0]


def normalize_answer(answer):
    """Lowercase and strip whitespace for comparison."""
    return answer.strip().lower()


def evaluate_m3vqa(model, processor, questions, image_dir, out_dir, steps, decoding_strategy="steps"):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{decoding_strategy}_steps{steps:03d}.json")

    if os.path.exists(out_file):
        print(f"Loading existing results from {out_file}")
        with open(out_file) as f:
            results = json.load(f)
    else:
        results = []
        for item in tqdm(questions, desc=f"Evaluating M3-VQA (steps={steps})"):
            img_path = os.path.join(image_dir, item["image_id"])
            question = item["question"]
            prediction = run_inference(model, processor, img_path, question, steps, decoding_strategy)
            results.append({
                "data_id": item["data_id"],
                "image_id": item["image_id"],
                "question": question,
                "prediction": prediction,
                "answers": item["answers"],
                "question_type": item.get("question_type", ""),
                "question_hop": item.get("question_hop", -1),
            })
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {out_file}")

    # Compute accuracy: prediction is correct if any gold answer appears in the output
    correct = 0
    total = len(results)
    hop_stats = {}

    for res in results:
        pred = normalize_answer(res["prediction"])
        gold_answers = [normalize_answer(a) for a in res["answers"]]
        is_correct = any(gold in pred for gold in gold_answers)
        if is_correct:
            correct += 1
        hop = res.get("question_hop", -1)
        if hop not in hop_stats:
            hop_stats[hop] = {"correct": 0, "total": 0}
        hop_stats[hop]["total"] += 1
        if is_correct:
            hop_stats[hop]["correct"] += 1

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nSteps={steps} | Overall Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print("\nBreakdown by question hop:")
    for hop in sorted(hop_stats.keys()):
        h = hop_stats[hop]
        hop_acc = h["correct"] / h["total"] * 100
        print(f"  Hop {hop}: {h['correct']}/{h['total']} = {hop_acc:.2f}%")

    summary_file = os.path.join(out_dir, f"{decoding_strategy}_steps{steps:03d}_summary.json")
    summary = {
        "steps": steps,
        "decoding_strategy": decoding_strategy,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "hop_breakdown": {
            str(k): {"correct": v["correct"], "total": v["total"], "accuracy": v["correct"] / v["total"] * 100}
            for k, v in hop_stats.items()
        },
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to LVR-7B checkpoint")
    parser.add_argument("--questions", default="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl")
    parser.add_argument("--image_dir", default="/mnt/data/lannth/mLAnR/M3-VQA/images")
    parser.add_argument("--output_dir", default="/mnt/data/lannth/mLAnR/results/m3vqa")
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16],
                        help="LVR revision steps to evaluate")
    parser.add_argument("--decoding_strategy", default="steps", choices=["steps", "latent"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    args = parser.parse_args()

    print(f"Loading questions from {args.questions}")
    questions = []
    with open(args.questions) as f:
        for line in f:
            questions.append(json.loads(line))

    if args.max_samples:
        questions = questions[:args.max_samples]
    print(f"Loaded {len(questions)} questions")

    print(f"Loading model from {args.model_path}")
    model, processor = load_model(args.model_path)
    print("Model loaded.")

    run_name = os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, run_name)

    for steps in args.steps:
        evaluate_m3vqa(
            model, processor, questions, args.image_dir,
            out_dir, steps, args.decoding_strategy
        )


if __name__ == "__main__":
    main()
