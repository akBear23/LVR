# M3-VQA Evaluation with Retrieval Augmentation

This directory contains two evaluation scripts for LVR-7B on the M3-VQA benchmark:

## 1. Basic Evaluation (`evaluate_m3vqa.py`)
- Direct inference on M3-VQA questions
- No retrieval augmentation
- Baseline performance

## 2. Retrieval-Augmented Evaluation (`evaluate_m3vqa_retrieval.py`)
- Augments questions with retrieved passages from encyclopedic KB
- Three retrieval modes:

### Retrieval Modes

#### `oracle` (Ground-truth Upper Bound)
- Uses `evidence_urls` and `evidence_section_ids` directly from annotations
- Shows performance when perfect retrieval is available
- **Expected to be highest accuracy**
- Simulates oracle knowledge of relevant passages

#### `entity` (Entity Name Lookup)
- Uses `img_entity_names` to look up KB entries by title matching
- Simulates having a visual entity recognizer that identifies objects in images
- **More realistic, no oracle information**
- Example: Image shows "Guava" → lookup "Guava" in KB title index

#### `hybrid` (Best-Effort Fallback)
- Uses `oracle` when evidence_urls available
- Falls back to `entity` lookup otherwise
- Useful for partial oracle scenarios

---

## Usage

### Quick Test (100 samples)
```bash
cd /mnt/data/lannth/mLAnR/lvr

# Without retrieval
python evaluation/evaluate_m3vqa.py \
    --model_path /mnt/data/lannth/mLAnR/checkpoints/LVR-7B \
    --max_samples 100 --steps 8

# With oracle retrieval
python evaluation/evaluate_m3vqa_retrieval.py \
    --model_path /mnt/data/lannth/mLAnR/checkpoints/LVR-7B \
    --retrieval_mode oracle \
    --max_samples 100 --steps 8

# With entity lookup retrieval
python evaluation/evaluate_m3vqa_retrieval.py \
    --model_path /mnt/data/lannth/mLAnR/checkpoints/LVR-7B \
    --retrieval_mode entity \
    --max_samples 100 --steps 8
```

### Full Evaluation (all modes, all steps)
```bash
sbatch scripts/run_m3vqa_retrieval_eval.sh
```

Or run sequentially:
```bash
for mode in oracle entity hybrid; do
  python evaluation/evaluate_m3vqa_retrieval.py \
    --model_path /mnt/data/lannth/mLAnR/checkpoints/LVR-7B \
    --retrieval_mode $mode \
    --steps 4 8 16
done
```

### All Arguments
```bash
python evaluation/evaluate_m3vqa_retrieval.py \
    --model_path PATH \
    --kb_path PATH \
    --questions PATH \
    --image_dir PATH \
    --output_dir PATH \
    --retrieval_mode {oracle|entity|hybrid} \
    --steps N [N ...] \
    --decoding_strategy {steps|latent} \
    --max_samples N
```

---

## Output

Results saved to:
```
results/m3vqa_retrieval/LVR-7B/
├── oracle_steps_008.json          # All predictions
├── oracle_steps_008_summary.json   # Accuracy metrics
├── entity_steps_008.json
├── entity_steps_008_summary.json
└── ...
```

Summary JSON includes:
- `accuracy`: Overall % accuracy
- `total`/`correct`: Counts
- `avg_passages_retrieved`: Avg # of KB passages per question
- `hop_breakdown`: Accuracy by question hop count (1-hop, 2-hop, etc.)

---

## Expected Results

Typical performance hierarchy:
```
oracle    > hybrid    > entity    > no-retrieval
(ground    (best-      (entity     (baseline)
 truth)    effort)     lookup)
```

Example (hypothetical):
- **No retrieval**:    45% accuracy
- **Entity lookup**:   55% accuracy (+10 pp)
- **Oracle**:          72% accuracy (+27 pp)
- **Hybrid**:          68% accuracy (+23 pp)

---

## KB Statistics

- **Total entries**: 2.0M Wikipedia pages
- **Indexed by**: URL + title
- **Entity lookup**: Matches `img_entity_names` to KB titles
- **Context format**: `[Title - Section]\n{passage}`

---

## Implementation Notes

1. **Passage retrieval**: 
   - Oracle: Direct lookup by URL + section ID
   - Entity: Title matching in KB (case-sensitive first, can be extended)

2. **Context augmentation**:
   - Prepends `[Retrieved Context]` section to question
   - Concatenates multiple passages with newlines

3. **Performance**:
   - Oracle retrieval: ~instant (direct lookup)
   - Entity retrieval: ~instant (title index lookup)
   - Full evaluation: ~2-4 hours depending on model size and GPU

4. **Requirements**:
   - KB loaded entirely in memory (~2GB)
   - GPU with sufficient VRAM for model + batch inference
