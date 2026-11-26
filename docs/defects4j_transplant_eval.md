# Defects4J-Transplant Evaluation (coinse/Defects4J-multifault)

This document describes how HyTri evaluates the transplantation benchmark from
[coinse/Defects4J-multifault](https://github.com/coinse/Defects4J-multifault) as an
**additional** dataset alongside MFCI-style Defects4J-Real, Bears-Real, and BugSwarm-Real.

## Setup

1. Clone the coinse artifact:

```bash
git clone https://github.com/coinse/Defects4J-multifault.git external/Defects4J-multifault
```

2. Convert to HyTri `synthetic_builds.jsonl` (requires existing `data/defects4j_full/`):

```bash
python scripts/create_defects4j_multifault_transplant.py \
  --coinse-dir external/Defects4J-multifault \
  --output-dir data/defects4j_multifault_transplant
```

Smoke conversion (one project, few builds):

```bash
python scripts/create_defects4j_multifault_transplant.py \
  --projects Chart --max-builds-per-project 5
```

3. Run evaluation (four datasets if transplant data exists):

```bash
python scripts/run_evaluation.py \
  --num-builds 150 \
  --seed 42 \
  --methods TraceSim,PWOverlap,ReBucket,BuildSheriff,GPTrace,HyTri \
  --output results/full_multifault_eval.csv
```

`OPENAI_API_KEY` is required only when GPTrace or HyTri are included.

Transplant-only run with resume after interrupt (same `--seed`, `--methods`, and `--output`):

```bash
export OPENAI_API_KEY=...   # needed for GPTrace/HyTri
python scripts/run_evaluation.py \
  --datasets Defects4J-Transplant \
  --num-builds 150 \
  --seed 42 \
  --resume \
  --output results/transplant_eval_150_seed42.csv
```

`--resume` keeps builds that already have all method rows, drops partial builds and
re-runs them, then appends the remainder. Without `--resume`, the output CSV is overwritten.

## Ground Truth

For each multi-fault version `M` in coinse `multiple_faults/{Project}.json`:

- Each failing test `t` has a **set** of bugs: `Y(t) = { N | t is in N's revealing-test list in version M }`.
- Stored per failure as `ground_truth_bugs` (list of bug IDs like `Chart-3`).
- `ground_truth_bug` (single ID) is kept only for legacy compatibility.

## Paper Metrics (Scientific Protocol)

### Primary (Defects4J-Transplant)

**Overlap-aware pairwise clustering metrics** per build:

- `same_gt(i,j) = 1` iff `Y(i) ∩ Y(j) ≠ ∅`
- `same_pred(i,j) = 1` iff failures `i,j` share a predicted cluster
- Report: pairwise precision, recall, F1 (mean over builds)

Use these as the **main** results for the transplantation benchmark.

### Secondary (comparability)

On failures with **exactly one** ground-truth bug (`|Y(t)| = 1`), compute standard
ARI, NMI, and B-cubed F1. Report as `ari_single_label`, `nmi_single_label`,
`bcubed_f1_single_label` in the results CSV.

### Not primary

Tuple membership (`predicted_bug ∈ Y(t)`) is not used as the main metric because it
ignores cluster structure.

### Statistics

After evaluation, see `results/transplant_eval_stats.txt` for:

- Bootstrap 95% CIs on pairwise F1 (per method)
- Paired Wilcoxon tests vs BuildSheriff

## Output Files

| File | Description |
|------|-------------|
| `data/defects4j_multifault_transplant/synthetic_builds.jsonl` | Converted builds |
| `data/defects4j_multifault_transplant/dataset_summary.json` | Conversion stats |
| `results/multifault_eval_defects4j_transplant.csv` | Per-dataset slice |
| `results/transplant_eval_stats.txt` | CIs and significance |

## Interpretation

- **MFCI (Defects4J-Real)** models independent co-occurring bugs with disjoint labels.
- **Transplantation** models coexisting/interacting faults; tests may map to multiple bugs.
- Do not compare raw ARI across MFCI and transplant without noting different label semantics.
- Report transplant results using **pairwise F1** as primary.

## Citation

If you use this dataset, cite the SSBSE 2021 paper (see coinse repository README).
