#!/usr/bin/env python3
"""Verify completeness and JSON-parse quality of llm_cache_qwen.json."""

import json
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
builds = json.loads((DATA_DIR / "pilot_45_builds.json").read_text(encoding="utf-8"))
cache = json.loads((DATA_DIR / "llm_cache_qwen.json").read_text(encoding="utf-8"))

print(f"Builds in dataset: {len(builds)}")
print(f"Builds in cache[hypotheses]: {len(cache['hypotheses'])}")
print(f"Unique normalizations: {len(cache['normalizations'])}")
print(f"Unique embeddings: {len(cache['embeddings'])}")

# 1. Check normalization fallback rate
norm_total = len(cache["normalizations"])
norm_fb = sum(
    1 for v in cache["normalizations"].values()
    if "@unknown ->" in (v.get("normalized_signature") or "")
)
print(f"\n[S_semantic] Normalization fallback: {norm_fb}/{norm_total} ({norm_fb/max(norm_total,1):.1%})")

# 2. Check hypothesis fallback & suspected_change hit rate
hyp_total = 0
hyp_fb = 0
hyp_has_file = 0
hyp_exact_match = 0

for b in builds:
    bid = b["build_id"]
    changed = set(b.get("combined_changed_files") or [])
    hyps = cache["hypotheses"].get(bid, [])
    for h in hyps:
        hyp_total += 1
        mech = h.get("mechanism") or ""
        is_fb = (
                h.get("category") == "unknown"
                and mech.startswith("Failure in ")
                and h.get("confidence") in (0.1, 0.3)
        )
        if is_fb:
            hyp_fb += 1
        sc = h.get("suspected_change")
        if sc and str(sc).lower() != "null":
            hyp_has_file += 1
            if sc in changed:
                hyp_exact_match += 1

print(f"[S_hyp] Total hypotheses: {hyp_total}")
print(f"[S_hyp] Fallback (parse fail / truncated): {hyp_fb}/{hyp_total} ({hyp_fb/max(hyp_total,1):.1%})")
print(f"[S_hyp] Non-null suspected_change: {hyp_has_file}/{hyp_total} ({hyp_has_file/max(hyp_total,1):.1%})")
print(f"[S_hyp] Exact match with changed_files: {hyp_exact_match}/{hyp_total} ({hyp_exact_match/max(hyp_total,1):.1%})")

# 3. Check timing stats
times = [v["llm_total_sec"] for v in cache["build_timings"].values()]
print(f"\n[Timing] Mean: {np.mean(times):.2f}s/build | Median: {np.median(times):.2f}s/build | Total: {np.sum(times)/60:.1f} min")

# 4. Print 3 sample outputs for manual sanity check
print("\n--- SAMPLE NORMALIZATIONS (3) ---")
for k, v in list(cache["normalizations"].items())[:3]:
    print(" ", v["category"], "|", v["normalized_signature"][:100])

print("\n--- SAMPLE HYPOTHESES (3) ---")
sample_b = builds[0]["build_id"]
for h in cache["hypotheses"].get(sample_b, [])[:3]:
    print(" ", h["failure_id"], "| file:", h["suspected_change"], "| conf:", h["confidence"], "| mech:", h["mechanism"][:90])