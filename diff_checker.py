# 3-way JSON diff (single-load version)
# Compares:
#   A. single-agent  vs multi-agent run1
#   B. single-agent  vs multi-agent run2
#   C. multi-agent run1 vs run2
# Loads each JSON only once to avoid float parse noise.

import json, numpy as np
from pathlib import Path
from typing import Dict, Any

RUN_DIF = Path("checkpoints/rllm-agent/4b-frozenlake_agent/batch_snapshots")
RUN_DIR = Path("checkpoints/rllm-coe/frozenlake-prod/batch_snapshots")

BASENAME  = "batch_multi_agent_judge_step_{step}.json"
BASENAME2 = "batch_multi_agent_judge_step_{step}_2ndrun.json"

RTOL = 1e-6
ATOL = 1e-6

def load_json_strip_meta(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    data.pop("meta_info", None)
    return data

def np_array_safe(x):
    try:
        arr = np.array(x, dtype=np.float64)
        if arr.dtype == object:
            return None
        return arr
    except Exception:
        return None

def diff_tensor(a_arr: np.ndarray, b_arr: np.ndarray):
    if a_arr is None or b_arr is None:
        return False, [], 0, {}
    if a_arr.shape != b_arr.shape:
        return True, [], -1, {"shape_mismatch": (a_arr.shape, b_arr.shape)}
    if a_arr.size == 0:
        return False, [], 0, {"empty": True}

    mask = ~np.isclose(a_arr, b_arr, rtol=RTOL, atol=ATOL)
    changed = bool(mask.any())
    if not changed:
        return False, [], 0, {"equal": True}

    rows_changed = np.where(mask.any(axis=1))[0] if mask.ndim >= 2 else np.where(mask)[0]
    diff_vals = np.abs((a_arr - b_arr)[mask])
    stats = {
        "count": int(diff_vals.size),
        "min": float(diff_vals.min()),
        "max": float(diff_vals.max()),
        "mean": float(diff_vals.mean()),
    }
    if a_arr.ndim >= 2 and rows_changed.size:
        bs = a_arr.shape[0]
        k = 0
        for i in range(bs):
            if i in rows_changed: k += 1
            else: break
        stats["prefix_rows_changed"] = int(k)
        stats["batch_rows"] = int(bs)
        stats["prefix_only"] = (k == len(rows_changed) and k > 0)
    return True, rows_changed.tolist(), int(stats["count"]), stats

def compare(label: str, A: Dict[str, Any], B: Dict[str, Any]):
    print(f"\n---- {label} ----")
    ta, tb = A.get("batch_tensors", {}), B.get("batch_tensors", {})
    common = sorted(set(ta.keys()) & set(tb.keys()))
    any_diff, prefix_hits = False, []
    for k in common:
        a_arr, b_arr = np_array_safe(ta[k]), np_array_safe(tb[k])
        changed, rows, cnt, stats = diff_tensor(a_arr, b_arr)
        if changed:
            any_diff = True
            if "shape_mismatch" in stats:
                print(f"{k:20s} SHAPE MISMATCH: {stats['shape_mismatch'][0]} vs {stats['shape_mismatch'][1]}")
                continue
            print(f"{k:20s} CHANGED: rows={len(rows)} changes={cnt} "
                  f"Δ[min={stats['min']:.6g} max={stats['max']:.6g} mean={stats['mean']:.6g}]")
            if stats.get("prefix_only"):
                prefix_hits.append((k, stats["prefix_rows_changed"], stats["batch_rows"]))
        else:
            print(f"{k:20s} OK (identical)")
    if not any_diff:
        print("ALL IDENTICAL (within tolerance)")
    if prefix_hits:
        rows_set = {h[1] for h in prefix_hits}
        bs_set = {h[2] for h in prefix_hits}
        if len(rows_set)==1 and len(bs_set)==1:
            k = list(rows_set)[0]; bs = list(bs_set)[0]
            print(f"→ Prefix-only: first {k} of {bs} rows differ for {[h[0] for h in prefix_hits]}")

def summarize_step(step: int):
    print(f"\n===== STEP {step} =====")
    f_sa  = RUN_DIF / f"batch_single_agent_step_{step}.json"
    f_ma1 = RUN_DIR / BASENAME.format(step=step)
    f_ma2 = RUN_DIR / BASENAME2.format(step=step)

    cache = {}
    for p in [f_sa, f_ma1, f_ma2]:
        if p.exists():
            cache[p] = load_json_strip_meta(p)
        else:
            print(f"Missing: {p}")

    if f_sa in cache and f_ma1 in cache:
        compare("A. single-agent vs multi-agent run1", cache[f_sa], cache[f_ma1])
    if f_sa in cache and f_ma2 in cache:
        compare("B. single-agent vs multi-agent run2", cache[f_sa], cache[f_ma2])
    if f_ma1 in cache and f_ma2 in cache:
        compare("C. multi-agent run1 vs run2", cache[f_ma1], cache[f_ma2])

for i in range(1, 11):
    summarize_step(i)
