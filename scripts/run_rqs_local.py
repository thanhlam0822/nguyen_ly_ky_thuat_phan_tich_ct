#!/usr/bin/env python3
"""
Clean, non-overfitted CPU evaluation for RQ1, RQ2, RQ3 (~5 seconds).
- Disjoint split: 15 Tuning builds (seed=123) -> 30 Test builds (seed=42)
- Compact, principled tuning space: 24 configs (no 40k-config overfitting)
- Includes HyTri-Selective: static+dynamic on stack-rich builds, calibrated local LLM on stack-poor builds
"""

import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hytri.models import FailureInstance, StackFrame
from hytri.parsing.stack_trace import StackTraceParser
from hytri.parsing.signature import SignatureTokenizer
from hytri.ensemble.complexity_gate import ComplexityGate
from hytri.ensemble.hypothesis import Hypothesis
from hytri.ensemble.signals import MultiSignalComputer, SimilaritySignals
from hytri.ensemble.aggregator import AggregationEnsemble
from hytri.ensemble.threshold import AdaptiveThreshold, ThresholdConfig
from hytri.evaluation.stats_reporting import wilcoxon_signed_rank_pvalue

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_bcubed(true_labels, pred_labels):
    t, p = np.array(true_labels), np.array(pred_labels)
    n = len(t)
    if n == 0:
        return 0.0, 0.0, 0.0
    bp = sum((t[p == p[i]] == t[i]).sum() / (p == p[i]).sum() for i in range(n)) / n
    br = sum((p[t == t[i]] == p[i]).sum() / (t == t[i]).sum() for i in range(n)) / n
    bf1 = (2 * bp * br / (bp + br)) if (bp + br) > 0 else 0.0
    return float(bp), float(br), float(bf1)


def compute_metrics(true_labels, pred_labels):
    bp, br, bf1 = compute_bcubed(true_labels, pred_labels)
    return {
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "bcubed_precision": bp,
        "bcubed_recall": br,
        "bcubed_f1": bf1,
        "num_pred_clusters": len(set(pred_labels)),
    }


def get_ground_truth(failures_raw: list[dict]) -> list[int]:
    bug_map, labels = {}, []
    for f in failures_raw:
        if f.get("ground_truth_bug") is not None:
            bid = f["ground_truth_bug"]
        elif f.get("ground_truth"):
            gt = f["ground_truth"]
            bid = gt.get("root_cause_id", gt.get("original_bug_id", "unknown"))
        else:
            bid = f.get("bug_id", "unknown")
        if bid not in bug_map:
            bug_map[bid] = len(bug_map)
        labels.append(bug_map[bid])
    return labels


def to_instances(failures_raw: list[dict]) -> list[FailureInstance]:
    parser, tokenizer = StackTraceParser(), SignatureTokenizer()
    out = []
    for f in failures_raw:
        frames = [
            StackFrame(file=fr.get("file", ""), line=fr.get("line", 0), func=fr.get("func", ""))
            for fr in f.get("frames", [])[:20]
        ]
        inst = FailureInstance(
            build_id=f.get("build_id", ""),
            failure_id=f.get("failure_id", ""),
            test_name=f.get("test_name", ""),
            exception_type=f.get("exception_type", "Unknown"),
            exception_message=f.get("exception_message", ""),
            stack_trace=f.get("stack_trace", "")[:2000],
            frames=frames,
            project=f.get("project", ""),
        )
        if not inst.frames:
            inst.frames = parser.parse(inst.stack_trace).frames
        tokenizer.update_failure_signature(inst)
        out.append(inst)
    return out


def raw_cosine_matrix(texts: list[str], emb_cache: dict[str, list[float]]) -> np.ndarray:
    dim = len(next(iter(emb_cache.values())))
    mat = np.array([emb_cache.get(t, [0.0] * dim) for t in texts], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    cos = np.dot(mat / norms, (mat / norms).T).astype(np.float64)
    np.fill_diagonal(cos, 1.0)
    return np.clip(cos, -1.0, 1.0)


def calibrate_cosine(raw_cos: np.ndarray, mode: str) -> np.ndarray:
    if mode == "minmax":
        n = raw_cos.shape[0]
        if n <= 1:
            return np.ones_like(raw_cos)
        mask = ~np.eye(n, dtype=bool)
        lo, hi = float(np.min(raw_cos[mask])), float(np.max(raw_cos[mask]))
        sim = np.clip((raw_cos - lo) / (hi - lo), 0.0, 1.0) if (hi - lo) > 1e-6 else np.clip(raw_cos, 0.0, 1.0)
    else:
        sim = (raw_cos + 1.0) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim


def fallback_hypotheses(failures: list[FailureInstance], changed_files: list[str]) -> list[Hypothesis]:
    changed_base = {
        f.split("/")[-1].replace(".java", "").replace(".py", "").lower(): f
        for f in changed_files
    }
    out = []
    for fl in failures:
        suspected = None
        for fr in fl.frames:
            if fr.file:
                fb = fr.file.split("/")[-1].replace(".java", "").replace(".py", "").lower()
                for cb, orig in changed_base.items():
                    if cb and (cb in fb or fb in cb):
                        suspected = orig
                        break
            if suspected:
                break
        out.append(Hypothesis(
            failure_id=fl.failure_id,
            suspected_change=suspected,
            mechanism=f"Failure in {fl.exception_type or 'test'}"
                      + (f" likely related to change in {suspected}" if suspected else ""),
            confidence=0.3 if suspected else 0.1,
            reasoning="Fallback hypothesis",
        ))
    return out


def precompute_builds(builds: list[dict], cache: dict) -> list[dict]:
    gate = ComplexityGate(delta_exception=7, delta_assertion=7, max_files_for_single_cluster=3, require_file_overlap=True)
    sig_comp = MultiSignalComputer(openai_client=None, use_semantic=False)
    emb_cache, norm_cache, hyp_cache = cache["embeddings"], cache["normalizations"], cache["hypotheses"]
    timings = cache.get("build_timings", {})

    prepared = []
    for b in builds:
        t0 = time.perf_counter()
        failures_raw = b["failures"]
        failures = to_instances(failures_raw)
        gt = get_ground_truth(failures_raw)
        changed_files = b.get("combined_changed_files") or []
        file_diffs = b.get("combined_file_diffs") or {}

        gate_res = gate.analyze(changed_files, file_diffs) if changed_files else None
        short_circuit = bool(gate_res and gate_res.should_single_cluster)

        h_map = {h["failure_id"]: h for h in hyp_cache.get(b["build_id"], [])}
        qwen_hyps = [
            Hypothesis(
                failure_id=fl.failure_id,
                suspected_change=h_map.get(fl.failure_id, {}).get("suspected_change"),
                mechanism=h_map.get(fl.failure_id, {}).get("mechanism") or "Unknown mechanism",
                confidence=float(h_map.get(fl.failure_id, {}).get("confidence", 0.5)),
                reasoning="",
            )
            for fl in failures
        ]
        base_signals = sig_comp.compute(
            failures, changed_files=changed_files, file_diffs=file_diffs,
            hypotheses=qwen_hyps if changed_files else None,
        )

        sigs = []
        for fl_raw in failures_raw:
            nk = f"{fl_raw.get('exception_type')}|{fl_raw.get('exception_message','')[:300]}|{fl_raw.get('stack_trace','')[:800]}"
            norm = norm_cache.get(nk, {})
            sigs.append(
                norm.get("normalized_signature")
                or f"{fl_raw.get('exception_type','Unknown')}@unknown -> {(fl_raw.get('exception_message') or '')[:50]}"
            )
        raw_cos = raw_cosine_matrix(sigs, emb_cache)
        fb_hyps = fallback_hypotheses(failures, changed_files) if changed_files else []
        s_hyp_fb = sig_comp._compute_hypothesis_similarity(fb_hyps) if fb_hyps else np.array([])

        # Exact HyTri built-in stack quality check from threshold.py
        has_strong_stack = (
                base_signals.get_confidence("pdm") >= 0.3 or base_signals.get_confidence("overlap") >= 0.3
        )
        cpu_sec = time.perf_counter() - t0

        prepared.append({
            "build_id": b["build_id"],
            "source": b["source"],
            "split": b["split"],
            "num_failures": len(failures),
            "num_true_clusters": len(set(gt)),
            "failures": failures,
            "gt": gt,
            "short_circuit": short_circuit,
            "has_strong_stack": has_strong_stack,
            "base_signals": base_signals,
            "s_sem": {"none": calibrate_cosine(raw_cos, "none"), "minmax": calibrate_cosine(raw_cos, "minmax")},
            "s_hyp_qwen": base_signals.s_hypothesis.copy(),
            "s_hyp_fb": s_hyp_fb,
            "llm_sec": timings.get(b["build_id"], {}).get("llm_total_sec", 0.0),
            "cpu_sec": cpu_sec,
        })
    return prepared


def run_build(
        item: dict,
        weights: dict[str, float],
        use_sem: bool = True,
        hyp_mode: str = "qwen",   # "none", "fallback", "qwen"
        sem_cal: str = "none",
        disagreement_penalty: float = 0.2,
        dt_stack: float = 0.57,
        dt_message: float = 0.73,
        selective: bool = False,
) -> tuple[dict, float]:
    n = item["num_failures"]
    if n <= 1 or item["short_circuit"]:
        return compute_metrics(item["gt"], [0] * n), item["cpu_sec"]

    # Selective Hybrid: skip local LLM on stack-rich builds, invoke only on stack-poor builds
    effective_use_sem = use_sem
    effective_hyp = hyp_mode
    latency = item["cpu_sec"]
    if selective and item["has_strong_stack"]:
        effective_use_sem = False
        effective_hyp = "none"
    elif effective_use_sem or effective_hyp == "qwen":
        latency += item["llm_sec"]

    b = item["base_signals"]
    sig = SimilaritySignals(
        s_pdm=b.s_pdm,
        s_overlap=b.s_overlap,
        s_message=b.s_message,
        s_test_name=b.s_test_name,
        s_diff_line=b.s_diff_line,
        s_semantic=item["s_sem"][sem_cal] if effective_use_sem else np.array([]),
        s_hypothesis=(
            item["s_hyp_qwen"] if effective_hyp == "qwen"
            else item["s_hyp_fb"] if effective_hyp == "fallback"
            else np.array([])
        ),
        confidences=b.confidences.copy(),
        n_failures=b.n_failures,
    )
    sig.confidences["semantic"] = 0.7 if effective_use_sem else 0.0
    sig.confidences["hypothesis"] = 0.8 if (effective_hyp != "none" and sig.s_hypothesis.size > 0) else 0.0

    agg = AggregationEnsemble(
        signal_weights=weights,
        use_confidence_weighting=True,
        disagreement_penalty=disagreement_penalty,
        min_cluster_size=1,
        use_type_adaptation=True,
    )
    th = AdaptiveThreshold(ThresholdConfig(dt_stack=dt_stack, dt_message=dt_message)).select_threshold(sig, failure_type="mixed")
    res = agg.cluster(item["failures"], sig, distance_threshold=th)
    fid_to_label = {fid: cid for cid, c in enumerate(res.clusters) for fid in c.failure_ids}
    pred = [fid_to_label.get(f.failure_id, 0) for f in item["failures"]]
    return compute_metrics(item["gt"], pred), latency


def eval_set(items: list[dict], **cfg):
    rows, lats = [], []
    for it in items:
        m, lat = run_build(it, **cfg)
        rows.append({"build_id": it["build_id"], "source": it["source"], "latency_sec": round(lat, 3), **m})
        lats.append(lat)
    summ = {
        "ari": float(np.mean([r["ari"] for r in rows])),
        "nmi": float(np.mean([r["nmi"] for r in rows])),
        "bp": float(np.mean([r["bcubed_precision"] for r in rows])),
        "br": float(np.mean([r["bcubed_recall"] for r in rows])),
        "f1": float(np.mean([r["bcubed_f1"] for r in rows])),
        "med_lat": float(np.median(lats)),
        "mean_lat": float(np.mean(lats)),
    }
    for src in ["Defects4J-Real", "Bears-Real", "BugSwarm-Real"]:
        sub = [r for r in rows if r["source"] == src]
        summ[src] = float(np.mean([r["ari"] for r in sub])) if sub else 0.0
    return rows, summ


def safe_wilcoxon(a: list[float], b: list[float]) -> float:
    if np.allclose(a, b):
        return 1.0
    try:
        return float(wilcoxon_signed_rank_pvalue(a, b))
    except Exception:
        return 1.0


# 3 principled weight profiles (small, interpretable)
CODE_DEFAULT = {"pdm": 0.25, "overlap": 0.25, "diff_line": 0.10, "message": 0.15, "test_name": 0.10, "semantic": 0.15, "hypothesis": 0.10}
PAPER_FIXED_DIFF = {"pdm": 0.20, "overlap": 0.20, "diff_line": 0.18, "message": 0.12, "test_name": 0.08, "semantic": 0.12, "hypothesis": 0.10}
STATIC_BALANCED = {"pdm": 0.22, "overlap": 0.22, "diff_line": 0.20, "message": 0.12, "test_name": 0.08, "semantic": 0.11, "hypothesis": 0.05}
PROFILES = {"code_default": CODE_DEFAULT, "paper_fixed_diff": PAPER_FIXED_DIFF, "static_balanced": STATIC_BALANCED}


def main():
    builds = json.loads((DATA_DIR / "pilot_45_builds.json").read_text(encoding="utf-8"))
    cache = json.loads((DATA_DIR / "llm_cache_qwen.json").read_text(encoding="utf-8"))
    prepared = precompute_builds(builds, cache)
    test_items = [x for x in prepared if x["split"] == "test"]
    tune_items = [x for x in prepared if x["split"] == "tuning"]

    # RQ1
    print("=" * 94)
    print("RQ1: CHẨN ĐOÁN HYTRI + QWEN2.5-CODER-7B (MẶC ĐỊNH) TRÊN 30 TEST BUILDS")
    print("=" * 94)
    rq1_cfgs = {
        "HyTri-NoLLM (Static+Dynamic)": dict(weights=CODE_DEFAULT, use_sem=False, hyp_mode="none"),
        "HyTri+Qwen (Unpatched _call_openai)": dict(weights=CODE_DEFAULT, use_sem=True, hyp_mode="fallback"),
        "HyTri+Qwen (Only S_semantic)": dict(weights=CODE_DEFAULT, use_sem=True, hyp_mode="none"),
        "HyTri+Qwen (Only S_hyp)": dict(weights=CODE_DEFAULT, use_sem=False, hyp_mode="qwen"),
        "HyTri+Qwen (Default Full)": dict(weights=CODE_DEFAULT, use_sem=True, hyp_mode="qwen"),
    }
    rq1_map = {}
    print(f"{'Variant':<38} {'ARI':>7} {'B3 F1':>7} | {'D4J':>6} {'Bears':>6} {'BugSw':>6}")
    print("-" * 94)
    for name, cfg in rq1_cfgs.items():
        rows, s = eval_set(test_items, **cfg)
        rq1_map[name] = {r["build_id"]: r for r in rows}
        print(f"{name:<38} {s['ari']:>7.3f} {s['f1']:>7.3f} | {s['Defects4J-Real']:>6.3f} {s['Bears-Real']:>6.3f} {s['BugSwarm-Real']:>6.3f}")

    # RQ2: Compact 24-config tuning strictly on 15 Tuning builds
    print("\n" + "=" * 94)
    print("RQ2: HIỆU CHỈNH GỌN (24 CẤU HÌNH) TRÊN 15 TUNING BUILDS -> ĐÁNH GIÁ TRÊN 30 TEST BUILDS")
    print("=" * 94)
    _, tune_def = eval_set(tune_items, weights=CODE_DEFAULT, use_sem=True, hyp_mode="qwen", sem_cal="none", disagreement_penalty=0.2, dt_message=0.73)

    best = None
    for cal in ["none", "minmax"]:
        for pname, w in PROFILES.items():
            for lam in [0.1, 0.2]:
                for dtm in [0.65, 0.73]:
                    _, s = eval_set(tune_items, weights=w, use_sem=True, hyp_mode="qwen", sem_cal=cal, disagreement_penalty=lam, dt_stack=0.57, dt_message=dtm)
                    cand = (round(s["ari"], 6), round(s["f1"], 6))
                    if best is None or cand > best["key"]:
                        best = {"key": cand, "tune_ari": s["ari"], "sem_cal": cal, "profile": pname, "weights": w, "lam": lam, "dt_message": dtm}

    print(f"Tuning set (N=15): Default ARI = {tune_def['ari']:.3f} -> Selected ARI = {best['tune_ari']:.3f} ({best['tune_ari']-tune_def['ari']:+.3f})")
    print(f"Selected config (24 checked): sem_cal={best['sem_cal']}, weights={best['profile']}, lambda={best['lam']}, dt_stack=0.57 (fixed), dt_message={best['dt_message']}\n")

    rq2_steps = {
        "Step 0: HyTri+Qwen (Default)": dict(weights=CODE_DEFAULT, use_sem=True, hyp_mode="qwen", sem_cal="none", disagreement_penalty=0.2, dt_stack=0.57, dt_message=0.73),
        "Step 1: + Min-Max Calibration": dict(weights=CODE_DEFAULT, use_sem=True, hyp_mode="qwen", sem_cal=best["sem_cal"], disagreement_penalty=0.2, dt_stack=0.57, dt_message=0.73),
        "Step 2: + Calibrated Fusion (Full)": dict(weights=best["weights"], use_sem=True, hyp_mode="qwen", sem_cal=best["sem_cal"], disagreement_penalty=best["lam"], dt_stack=0.57, dt_message=best["dt_message"]),
        "Step 3: HyTri-Selective (Stack-Gated)": dict(weights=best["weights"], use_sem=True, hyp_mode="qwen", sem_cal=best["sem_cal"], disagreement_penalty=best["lam"], dt_stack=0.57, dt_message=best["dt_message"], selective=True),
    }
    rq2_map, rq2_summ = {}, {}
    print(f"{'RQ2 Configuration (30 Test Builds)':<38} {'ARI':>7} {'B3 F1':>7} | {'D4J':>6} {'Bears':>6} {'BugSw':>6} | {'Med(s)':>6} {'Mean(s)':>7}")
    print("-" * 94)
    for name, cfg in rq2_steps.items():
        rows, s = eval_set(test_items, **cfg)
        rq2_map[name] = {r["build_id"]: r for r in rows}
        rq2_summ[name] = s
        print(f"{name:<38} {s['ari']:>7.3f} {s['f1']:>7.3f} | {s['Defects4J-Real']:>6.3f} {s['Bears-Real']:>6.3f} {s['BugSwarm-Real']:>6.3f} | {s['med_lat']:>6.2f} {s['mean_lat']:>7.2f}")

    # RQ3
    print("\n" + "=" * 94)
    print("RQ3: SO SÁNH TỔNG THỂ TRÊN 30 TEST BUILDS & KIỂM ĐỊNH WILCOXON")
    print("=" * 94)
    by_method = defaultdict(dict)
    base_csv = RESULTS_DIR / "test_baselines_30.csv"
    if base_csv.exists():
        with open(base_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                by_method[r["method"]][r["build_id"]] = {
                    "build_id": r["build_id"], "source": r["source"],
                    "ari": float(r["ari"]), "nmi": float(r["nmi"]),
                    "bcubed_precision": float(r["bcubed_precision"]),
                    "bcubed_recall": float(r["bcubed_recall"]),
                    "bcubed_f1": float(r["bcubed_f1"]),
                    "latency_sec": 0.05,
                }

    by_method["HyTri-NoLLM"] = rq1_map["HyTri-NoLLM (Static+Dynamic)"]
    by_method["HyTri+Qwen (Default)"] = rq2_map["Step 0: HyTri+Qwen (Default)"]
    by_method["HyTri+Qwen (Calibrated)"] = rq2_map["Step 2: + Calibrated Fusion (Full)"]
    by_method["HyTri-Selective (Proposed)"] = rq2_map["Step 3: HyTri-Selective (Stack-Gated)"]

    order = [
        "TraceSim", "PWOverlap", "ReBucket", "BuildSheriff",
        "HyTri-NoLLM", "HyTri+Qwen (Default)", "HyTri+Qwen (Calibrated)", "HyTri-Selective (Proposed)",
    ]
    test_bids = [it["build_id"] for it in test_items]

    print(f"{'Method':<27} {'ARI':>6} {'NMI':>6} {'B3 F1':>6} | {'D4J':>6} {'Bears':>6} {'BugSw':>6} | {'Mean(s)':>7}")
    print("-" * 94)
    summary_rows, per_build_rows = [], []
    for m in order:
        if m not in by_method or not by_method[m]:
            continue
        rows = [by_method[m][bid] for bid in test_bids if bid in by_method[m]]
        for r in rows:
            per_build_rows.append({"method": m, **r})
        rs = {
            "method": m,
            "ari": round(float(np.mean([r["ari"] for r in rows])), 4),
            "nmi": round(float(np.mean([r["nmi"] for r in rows])), 4),
            "bcubed_f1": round(float(np.mean([r["bcubed_f1"] for r in rows])), 4),
            "d4j_ari": round(float(np.mean([r["ari"] for r in rows if r["source"] == "Defects4J-Real"])), 4),
            "bears_ari": round(float(np.mean([r["ari"] for r in rows if r["source"] == "Bears-Real"])), 4),
            "bugswarm_ari": round(float(np.mean([r["ari"] for r in rows if r["source"] == "BugSwarm-Real"])), 4),
            "mean_latency_sec": round(float(np.mean([r.get("latency_sec", 0.0) for r in rows])), 2),
        }
        summary_rows.append(rs)
        print(f"{m:<27} {rs['ari']:>6.3f} {rs['nmi']:>6.3f} {rs['bcubed_f1']:>6.3f} | {rs['d4j_ari']:>6.3f} {rs['bears_ari']:>6.3f} {rs['bugswarm_ari']:>6.3f} | {rs['mean_latency_sec']:>7.2f}")

    print("\n--- WILCOXON SIGNED-RANK TESTS (N = 30 Test Builds, Metric = ARI) ---")
    for ma, mb in [
        ("HyTri+Qwen (Calibrated)", "HyTri+Qwen (Default)"),
        ("HyTri-Selective (Proposed)", "HyTri+Qwen (Default)"),
        ("HyTri-Selective (Proposed)", "BuildSheriff"),
        ("HyTri-Selective (Proposed)", "HyTri-NoLLM"),
        ("HyTri+Qwen (Calibrated)", "BuildSheriff"),
    ]:
        va = [by_method[ma][bid]["ari"] for bid in test_bids]
        vb = [by_method[mb][bid]["ari"] for bid in test_bids]
        p = safe_wilcoxon(va, vb)
        w = sum(1 for a, b in zip(va, vb) if a > b + 1e-6)
        t = sum(1 for a, b in zip(va, vb) if abs(a - b) <= 1e-6)
        l = len(va) - w - t
        print(f"  {ma:<26} vs {mb:<24} | Delta={np.mean(va)-np.mean(vb):+.3f} | W/T/L={w}/{t}/{l} | p={p:.4f}")

    # Save clean CSVs (fixed extrasaction='ignore')
    with open(RESULTS_DIR / "clean_rq3_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    fields = ["method", "build_id", "source", "ari", "nmi", "bcubed_precision", "bcubed_recall", "bcubed_f1", "latency_sec"]
    with open(RESULTS_DIR / "clean_rq3_per_build.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_build_rows)

    print("\nSaved clean_rq3_summary.csv and clean_rq3_per_build.csv successfully.")


if __name__ == "__main__":
    main()