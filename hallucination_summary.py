#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan <root>/<task>/<model>/tables/overall_summary.json, group tasks, and compute:
- Per task-group × model: mean, SD, and 95% CI for
    * support_rate
    * conservative_hallucination_rate
    * strict_hallucination_rate
    * bm25_topk_mean (if available in overall_summary.json)
    * bm25_top1_mean (computed from rows.csv by aligning study × question_id)
- Overall per-model summary across groups
- Pair-wise tests across models per metric (Welch t-test + Holm correction; permutation fallback)
- Plots:
    * Bar charts of per-model means with 95% CI (rates + BM25; BM25 shows top-k if available else top-1)
    * Box plots per metric (by model) with the mean + 95% CI annotated above each box
    * Radar plot: support rate per domain grouped by model (requires per_domain.csv per run)
"""

import os
import glob
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional CSV export & table shaping
try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False

# Optional SciPy for Welch t; otherwise permutation fallback
try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ------------------- CONFIG -------------------
root_path = "/Users/liangz2/Documents/gpt_reviewer/hallucination"

# Canonical models to include (order preserved)
MODELS = ["gpt5", "gpt35", "o3_mini"]
PRETTY = {"gpt5": "gpt-5", "gpt35": "gpt-3.5", "o3_mini": "OpenAI o3-mini"}
PRETTY_ORDER = [PRETTY[m] for m in MODELS]

# Domains to show on radar (clockwise order)
RADAR_DOMAINS = ["Domain1", "Domain2", "Domain3", "Domain4", "Domain5", "OverallConclusion"]

# Task-group mapping (folder names on the right → merged group on the left)
TASK_GROUPS = {
    "Adverse events": {"Adverse_event"},
    "Anxiety symptoms": {"Anxiety0_1", "Anxiety1_6", "Anxiety7_24"},
    "Depressive symptoms": {"Depressive0_1", "Depressive1_6", "Depressive7_24"},
    "Diagnosis of mental disorders": {"Diagnosis0_1", "Diagnosis1_6", "Diagnosis7_24"},
    "Distress/PTSD symptoms": {"PTSD0_1", "PTSD1_6", "PTSD7_24"},
    "Psychological functioning and impairment": {"Psycho0_1", "Psycho1_6", "Psycho7_24"},
    "Quality of life": {"QOL0_1", "QOL1_6", "QOL7_24"},
    "Social outcomes": {"Social0_1", "Social1_6", "Social7_24"},
}

# Outputs
PRINT_PER_TASK_TABLE = True
SAVE_TASK_CSV = True
TASK_CSV_PATH = "/Users/liangz2/Documents/gpt_reviewer/hall_result/per_taskgroup_model_summary.csv"

SAVE_PAIRWISE_CSV = True
PAIRWISE_CSV_PATH = "/Users/liangz2/Documents/gpt_reviewer/hall_result/pairwise_tests.csv"

PLOT_PER_MODEL_BARS = True
PLOT_BARS_OUTFILE = "/Users/liangz2/Documents/gpt_reviewer/hall_result/hallucination_rates_by_model.png"

# BM25 per-model bar plot (top-k if available, else top-1)
PLOT_BM25_BARS = True
BM25_BARS_OUTFILE = "/Users/liangz2/Documents/gpt_reviewer/hall_result/bm25_score_by_model.png"

# Radar plot output
RADAR_SUPPORT_OUTFILE = "/Users/liangz2/Documents/gpt_reviewer/hall_result/radar_support_rate_by_domain.png"

PLOT_BOXPLOTS = True
BOXPLOT_OUTDIR = "/Users/liangz2/Documents/gpt_reviewer/hall_result/plots_box"
# ----------------------------------------------

METRICS = [
    ("support_rate", "Support rate"),
    ("conservative_hallucination_rate", "Conservative hallucination rate"),
    ("strict_hallucination_rate", "Strict hallucination rate"),
]

# ------------------- helpers -------------------
def mean(xs):
    xs = [x for x in xs if (x is not None and not (isinstance(x, float) and math.isnan(x)))]
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs):
    xs = [x for x in xs if (x is not None and not (isinstance(x, float) and math.isnan(x)))]
    if len(xs) < 2:
        return 0.0 if len(xs) == 1 else float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

def ci95_bounds(mu, sd, n):
    """95% CI for mean by normal approx. If n==1 -> (mu, mu)."""
    if not n or np.isnan(mu) or np.isnan(sd):
        return (float("nan"), float("nan"))
    if n == 1:
        return (mu, mu)
    half = 1.96 * sd / (n ** 0.5)
    return (mu - half, mu + half)

def as_rate(x):
    """Normalize to [0,1]; accept 0–1 or 0–100."""
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0:
        return None
    if v <= 1.0:
        return v
    if v <= 100.0:
        return v / 100.0
    return None

def canonical_model_label(raw: str | None) -> str | None:
    if not raw:
        return None
    return MODEL_SYNONYMS.get(raw.strip().lower())

MODEL_SYNONYMS = {
    "gpt-5": "gpt5", "gpt5": "gpt5", "gpt_5": "gpt5",
    "gpt-3.5": "gpt35", "gpt35": "gpt35", "gpt-4": "gpt35", "gpt4": "gpt35", "gpt-4o": "gpt35", "gpt-4o-mini": "gpt35",
    "openai o3-mini": "o3_mini", "o3_mini": "o3_mini", "o3-mini": "o3_mini", "gpt-o3-mini": "o3_mini",
}

# Build reverse lookup: atomic folder name -> group label
ATOMIC_TO_GROUP = {}
for g, atoms in TASK_GROUPS.items():
    for a in atoms:
        ATOMIC_TO_GROUP[a] = g

def group_task(task_folder_name: str) -> str:
    return ATOMIC_TO_GROUP.get(task_folder_name, task_folder_name)

# Welch test & helpers
def hedges_g(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if sp == 0 or np.isnan(sp):
        return float("nan")
    d = (np.mean(a) - np.mean(b)) / sp
    J = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    return float(J * d)

def welch_test(a, b, alpha=0.05):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    diff = m1 - m2
    se = np.sqrt((s1 / n1) + (s2 / n2))
    if HAVE_SCIPY and n1 > 1 and n2 > 1:
        res = stats.ttest_ind(a, b, equal_var=False)
        df = (s1 / n1 + s2 / n2) ** 2 / ((s1**2) / (n1**2 * (n1 - 1)) + (s2**2) / (n2**2 * (n2 - 1)))
        tcrit = stats.t.ppf(1 - alpha / 2.0, df)
        lo, hi = diff - tcrit * se, diff + tcrit * se
        return diff, se, float(df), float(res.statistic), float(res.pvalue), float(lo), float(hi)
    # permutation fallback
    rng = np.random.default_rng(42)
    comb = np.concatenate([a, b])
    obs = diff
    n_perm = 20000 if (n1 + n2) < 200 else 10000
    count = 0
    for _ in range(n_perm):
        rng.shuffle(comb)
        x = comb[:n1]
        y = comb[n1:]
        if abs(np.mean(x) - np.mean(y)) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    # bootstrap CI
    n_boot = 5000
    boots = []
    for _ in range(n_boot):
        x = np.random.choice(a, size=n1, replace=True)
        y = np.random.choice(b, size=n2, replace=True)
        boots.append(np.mean(x) - np.mean(y))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return diff, se, float("nan"), float("nan"), float(p), float(lo), float(hi)

def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    cmax = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        if adj < cmax:
            adj = cmax
        cmax = max(cmax, adj)
        adjusted[idx] = min(1.0, adj)
    return adjusted

# ---------- BM25 helpers ----------
def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None

def extract_bm25_mean_topk_from_blob(blob: dict) -> float | None:
    """
    Try to get a per-run BM25 'mean of top-k scores' from overall_summary.json.
    """
    key_candidates = [
        "bm25_mean_topk", "bm25_mean_top_k", "mean_bm25_topk", "bm25_topk_mean"
    ]
    for k in key_candidates:
        v = _coerce_float(blob.get(k))
        if v is not None:
            return v

    # Fallback via bm25_by_verdict (weighted by counts)
    bmv = blob.get("bm25_by_verdict")
    if isinstance(bmv, dict) and bmv:
        # prefer 'mean_topk_score'
        num, den = 0.0, 0.0
        for ver in bmv.values():
            c = _coerce_float(ver.get("count"))
            mu = _coerce_float(ver.get("mean_topk_score"))
            if c is not None and mu is not None:
                num += c * mu; den += c
        if den > 0:
            return num / den
        # fallback to top-1
        num, den = 0.0, 0.0
        for ver in bmv.values():
            c = _coerce_float(ver.get("count"))
            mu = _coerce_float(ver.get("mean_top_score"))
            if c is not None and mu is not None:
                num += c * mu; den += c
        if den > 0:
            return num / den
    return None

def read_rows_top1_mean(tables_dir: str):
    """
    From <tables_dir>/rows.csv, compute mean ± SD ± 95% CI of bm25_top_score,
    aligned by unique (study, question_id). If multiple duplicates appear, average them first.
    Returns dict with keys: mean, sd, n, ci_low, ci_high (or all NaN if file missing).
    """
    out = dict(mean=float("nan"), sd=float("nan"), n=0, ci_low=float("nan"), ci_high=float("nan"))
    if not HAVE_PANDAS:
        return out
    csv_path = os.path.join(tables_dir, "rows.csv")
    if not os.path.isfile(csv_path):
        return out
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out

    # normalize columns (accept case/underscore variants)
    cols = {c.lower(): c for c in df.columns}
    need = {"study", "question_id", "bm25_top_score"}
    if not need.issubset(set(cols.keys())):
        # try some common synonyms
        if "questionid" in cols and "question_id" not in cols:
            cols["question_id"] = cols["questionid"]
        if "bm25_topscore" in cols and "bm25_top_score" not in cols:
            cols["bm25_top_score"] = cols["bm25_topscore"]
        if not need.issubset(set(cols.keys())):
            return out

    d = df[[cols["study"], cols["question_id"], cols["bm25_top_score"]]].copy()
    d.columns = ["study", "question_id", "bm25_top_score"]
    # coerce numeric
    d["bm25_top_score"] = pd.to_numeric(d["bm25_top_score"], errors="coerce")
    d = d.dropna(subset=["study", "question_id", "bm25_top_score"])

    if d.empty:
        return out

    # align: unique per (study, question_id). If duplicates, average them.
    d = d.groupby(["study", "question_id"], as_index=False)["bm25_top_score"].mean()

    vals = d["bm25_top_score"].to_numpy(dtype=float)
    n = len(vals)
    mu = float(np.mean(vals)) if n > 0 else float("nan")
    sd = float(np.std(vals, ddof=1)) if n > 1 else (0.0 if n == 1 else float("nan"))
    lo, hi = ci95_bounds(mu, sd if not np.isnan(sd) else 0.0, n) if n > 0 else (float("nan"), float("nan"))
    return dict(mean=mu, sd=sd, n=n, ci_low=lo, ci_high=hi)

# ------------------- load overall_summary.json -------------------
pattern = os.path.join(root_path, "*", "*", "tables", "overall_summary.json")
files = sorted(glob.glob(pattern))
if not files:
    print(f"[INFO] No files found under: {pattern}")
    raise SystemExit(0)

records = []  # one per run: {task_group, task_atomic, model, path, support_rate, ..., bm25_mean_topk?, bm25_top1_mean?}
for fpath in files:
    rel = os.path.relpath(fpath, root_path)
    parts = rel.split(os.sep)
    # expected: [task_atomic, model, "tables", "overall_summary.json"]
    if len(parts) < 4:
        print(f"[WARN] Unexpected path shape: {fpath} (skipping)")
        continue
    task_atomic, model_folder, _, _ = parts
    task_group = group_task(task_atomic)
    tables_dir = os.path.dirname(fpath)

    # infer model from folder; allow JSON override if present
    model = canonical_model_label(model_folder)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except Exception as e:
        print(f"[WARN] unreadable JSON {fpath}: {e}")
        continue

    if isinstance(blob, dict):
        for k in ("model", "model_name", "llm", "llm_model"):
            if isinstance(blob.get(k), str):
                m2 = canonical_model_label(blob[k])
                if m2:
                    model = m2
                    break

    if model not in MODELS:
        continue

    sr  = as_rate(blob.get("support_rate"))
    chr_ = as_rate(blob.get("conservative_hallucination_rate"))
    shr = as_rate(blob.get("strict_hallucination_rate"))
    if any(v is None for v in (sr, chr_, shr)):
        print(f"[WARN] missing rates in {fpath}; support={sr}, cons={chr_}, strict={shr}. Skipping.")
        continue

    # BM25 top-k mean (from overall_summary.json or its fallbacks)
    bm25_mtk = extract_bm25_mean_topk_from_blob(blob)

    # NEW: BM25 top-1 mean from rows.csv (aligned via study × question_id)
    bm25_top1_stats = read_rows_top1_mean(tables_dir)
    bm25_top1_mean = bm25_top1_stats["mean"]

    rec = {
        "task_group": task_group,
        "task_atomic": task_atomic,
        "model": model,
        "path": fpath,
        "support_rate": sr,
        "conservative_hallucination_rate": chr_,
        "strict_hallucination_rate": shr,
        "bm25_mean_topk": bm25_mtk,        # may be None
        "bm25_top1_mean_rows": bm25_top1_mean,  # NaN if rows.csv missing
    }
    # keep per-run CI/stats for optional later use
    rec.update({f"bm25_top1_{k}_rows": v for k, v in bm25_top1_stats.items()})
    records.append(rec)

if not records:
    print("[INFO] No usable overall_summary.json files after filtering.")
    raise SystemExit(0)

# ------------------- overall per-model summary -------------------
by_model = {m: [] for m in MODELS}
for r in records:
    by_model[r["model"]].append(r)

print("\n=== Aggregated summary per model (across merged task groups) ===")
overall_summary = []
for m in MODELS:
    rows = by_model.get(m, [])
    srs  = [r["support_rate"] for r in rows]
    chrs = [r["conservative_hallucination_rate"] for r in rows]
    shrs = [r["strict_hallucination_rate"] for r in rows]
    # top-k means (across runs) where available
    bm_k  = [r["bm25_mean_topk"] for r in rows if r.get("bm25_mean_topk") is not None]
    # top-1 means from rows.csv (across runs)
    bm_1  = [r["bm25_top1_mean_rows"] for r in rows if not (r.get("bm25_top1_mean_rows") is None or (isinstance(r.get("bm25_top1_mean_rows"), float) and math.isnan(r.get("bm25_top1_mean_rows"))))]

    n = len(rows)
    support_mu, support_sd = mean(srs), std(srs)
    cons_mu, cons_sd = mean(chrs), std(chrs)
    strict_mu, strict_sd = mean(shrs), std(shrs)

    lo_s, hi_s = ci95_bounds(support_mu, support_sd, n)
    lo_c, hi_c = ci95_bounds(cons_mu, cons_sd, n)
    lo_t, hi_t = ci95_bounds(strict_mu, strict_sd, n)

    # top-k (if present)
    bm_k_mu, bm_k_sd, lo_bk, hi_bk, n_bk = float("nan"), float("nan"), float("nan"), float("nan"), 0
    if bm_k:
        n_bk = len(bm_k)
        bm_k_mu, bm_k_sd = mean(bm_k), std(bm_k)
        lo_bk, hi_bk = ci95_bounds(bm_k_mu, bm_k_sd, n_bk)

    # top-1 (rows.csv)
    bm_1_mu, bm_1_sd, lo_b1, hi_b1, n_b1 = float("nan"), float("nan"), float("nan"), float("nan"), 0
    if bm_1:
        n_b1 = len(bm_1)
        bm_1_mu, bm_1_sd = mean(bm_1), std(bm_1)
        lo_b1, hi_b1 = ci95_bounds(bm_1_mu, bm_1_sd, n_b1)

    overall_summary.append({
        "model": m, "n": n,
        "support_mean": support_mu, "support_std": support_sd, "support_ci_low": lo_s, "support_ci_high": hi_s,
        "cons_hall_mean": cons_mu, "cons_hall_std": cons_sd, "cons_hall_ci_low": lo_c, "cons_hall_ci_high": hi_c,
        "strict_hall_mean": strict_mu, "strict_hall_std": strict_sd, "strict_hall_ci_low": lo_t, "strict_hall_ci_high": hi_t,
        "bm25_topk_mean": bm_k_mu, "bm25_topk_std": bm_k_sd, "bm25_topk_ci_low": lo_bk, "bm25_topk_ci_high": hi_bk, "n_bm25_topk": n_bk,
        "bm25_top1_mean": bm_1_mu, "bm25_top1_std": bm_1_sd, "bm25_top1_ci_low": lo_b1, "bm25_top1_ci_high": hi_b1, "n_bm25_top1": n_b1,
    })

    msg_k = f"BM25 top-k mean: {bm_k_mu:.3f} (sd {bm_k_sd:.3f}, 95%CI {lo_bk:.3f}–{hi_bk:.3f}, n={n_bk})" if n_bk else "BM25 top-k mean: n=0"
    msg_1 = f"BM25 top-1 mean: {bm_1_mu:.3f} (sd {bm_1_sd:.3f}, 95%CI {lo_b1:.3f}–{hi_b1:.3f}, n={n_b1})" if n_b1 else "BM25 top-1 mean: n=0"
    print(
        f"{m:8s}  n={n:3d}  "
        f"support: {support_mu*100:5.1f}% (sd {support_sd*100:4.1f}, 95%CI {lo_s*100:5.1f}–{hi_s*100:5.1f})  "
        f"cons-hall: {cons_mu*100:5.1f}% (sd {cons_sd*100:4.1f}, 95%CI {lo_c*100:5.1f}–{hi_c*100:5.1f})  "
        f"strict-hall: {strict_mu*100:5.1f}% (sd {strict_sd*100:4.1f}, 95%CI {lo_t*100:5.1f}–{hi_t*100:5.1f})\n  {msg_k}\n  {msg_1}"
    )

# ------------------- per task-group × model stats -------------------
by_group_model = {}  # (task_group, model) -> list of rows
for r in records:
    key = (r["task_group"], r["model"])
    by_group_model.setdefault(key, []).append(r)

task_groups_order = list(TASK_GROUPS.keys())
extras = sorted({r["task_group"] for r in records if r["task_group"] not in TASK_GROUPS})
task_groups_order += [tg for tg in extras if tg not in task_groups_order]

rows_out = []
print("\n=== Per task-group × model (mean, SD, 95% CI) ===")
for tg in task_groups_order:
    print(f"\n[{tg}]")
    for m in MODELS:
        rows = by_group_model.get((tg, m), [])
        n = len(rows)
        if n == 0:
            print(f"  {m:8s}: n= 0  (no files)")
            rows_out.append({
                "task_group": tg, "model": m, "n": 0,
                "support_mean": math.nan, "support_std": math.nan, "support_ci_low": math.nan, "support_ci_high": math.nan,
                "cons_hall_mean": math.nan, "cons_hall_std": math.nan, "cons_hall_ci_low": math.nan, "cons_hall_ci_high": math.nan,
                "strict_hall_mean": math.nan, "strict_hall_std": math.nan, "strict_hall_ci_low": math.nan, "strict_hall_ci_high": math.nan,
                "bm25_topk_mean": math.nan, "bm25_topk_std": math.nan, "bm25_topk_ci_low": math.nan, "bm25_topk_ci_high": math.nan, "n_bm25_topk": 0,
                "bm25_top1_mean": math.nan, "bm25_top1_std": math.nan, "bm25_top1_ci_low": math.nan, "bm25_top1_ci_high": math.nan, "n_bm25_top1": 0,
            })
            continue

        srs  = [r["support_rate"] for r in rows]
        chrs = [r["conservative_hallucination_rate"] for r in rows]
        shrs = [r["strict_hallucination_rate"] for r in rows]
        # top-k across runs
        bk  = [r["bm25_mean_topk"] for r in rows if r.get("bm25_mean_topk") is not None]
        # top-1 across runs
        b1  = [r["bm25_top1_mean_rows"] for r in rows if not (r.get("bm25_top1_mean_rows") is None or (isinstance(r.get("bm25_top1_mean_rows"), float) and math.isnan(r.get("bm25_top1_mean_rows"))))]

        s_mu, s_sd = mean(srs), std(srs); s_lo, s_hi = ci95_bounds(s_mu, s_sd, n)
        c_mu, c_sd = mean(chrs), std(chrs); c_lo, c_hi = ci95_bounds(c_mu, c_sd, n)
        t_mu, t_sd = mean(shrs), std(shrs); t_lo, t_hi = ci95_bounds(t_mu, t_sd, n)

        if bk:
            bk_mu, bk_sd = mean(bk), std(bk); bk_lo, bk_hi = ci95_bounds(bk_mu, bk_sd, len(bk))
            msg_k = f"bm25 top-k mean {bk_mu:.3f} (sd {bk_sd:.3f}, 95%CI {bk_lo:.3f}–{bk_hi:.3f}, n={len(bk)})"
        else:
            bk_mu = bk_sd = bk_lo = bk_hi = math.nan
            msg_k = "bm25 top-k mean n=0"

        if b1:
            b1_mu, b1_sd = mean(b1), std(b1); b1_lo, b1_hi = ci95_bounds(b1_mu, b1_sd, len(b1))
            msg_1 = f"bm25 top-1 mean {b1_mu:.3f} (sd {b1_sd:.3f}, 95%CI {b1_lo:.3f}–{b1_hi:.3f}, n={len(b1)})"
        else:
            b1_mu = b1_sd = b1_lo = b1_hi = math.nan
            msg_1 = "bm25 top-1 mean n=0"

        print(
            f"  {m:8s}: n={n:2d}  "
            f"support {s_mu*100:5.1f}% (sd {s_sd*100:4.1f}, 95%CI {s_lo*100:5.1f}–{s_hi*100:5.1f})  "
            f"cons {c_mu*100:5.1f}% (sd {c_sd*100:4.1f}, 95%CI {c_lo*100:5.1f}–{c_hi*100:5.1f})  "
            f"strict {t_mu*100:5.1f}% (sd {t_sd*100:4.1f}, 95%CI {t_lo*100:5.1f}–{t_hi*100:5.1f})  |  {msg_k}  |  {msg_1}"
        )

        rows_out.append({
            "task_group": tg, "model": m, "n": n,
            "support_mean": s_mu, "support_std": s_sd, "support_ci_low": s_lo, "support_ci_high": s_hi,
            "cons_hall_mean": c_mu, "cons_hall_std": c_sd, "cons_hall_ci_low": c_lo, "cons_hall_ci_high": c_hi,
            "strict_hall_mean": t_mu, "strict_hall_std": t_sd, "strict_hall_ci_low": t_lo, "strict_hall_ci_high": t_hi,
            "bm25_topk_mean": bk_mu, "bm25_topk_std": bk_sd, "bm25_topk_ci_low": bk_lo, "bm25_topk_ci_high": bk_hi, "n_bm25_topk": len(bk),
            "bm25_top1_mean": b1_mu, "bm25_top1_std": b1_sd, "bm25_top1_ci_low": b1_lo, "bm25_top1_ci_high": b1_hi, "n_bm25_top1": len(b1),
        })

if SAVE_TASK_CSV:
    if not HAVE_PANDAS:
        print("[WARN] pandas not installed; cannot save CSV. `pip install pandas` to enable.")
    else:
        pd.DataFrame(rows_out).to_csv(TASK_CSV_PATH, index=False)
        print(f"\n[Saved] Per task-group × model summary → {TASK_CSV_PATH}")

# ------------------- pair-wise tests across models (rates only) -------------------
from itertools import combinations

def run_pairwise_tests(metric_key, label):
    print(f"\n-- {label} --")
    vecs = {m: np.array([r[metric_key] for r in by_model.get(m, [])], dtype=float) for m in MODELS}
    pairs = list(combinations(MODELS, 2))
    raw_p, tmp = [], []
    for a, b in pairs:
        va, vb = vecs[a], vecs[b]
        diff, se, df, tstat, p, lo, hi = welch_test(va, vb)
        g = hedges_g(va, vb)
        tmp.append((a, b, len(va), len(vb), float(np.mean(va)), float(np.mean(vb)),
                    diff, se, df, tstat, p, lo, hi, g))
        raw_p.append(p)
    adj = holm_bonferroni(np.array(raw_p, dtype=float))
    rows = []
    for (row, p_adj) in zip(tmp, adj):
        a, b, n1, n2, m1, m2, diff, se, df, tstat, p, lo, hi, g = row
        print(
            f"{a} vs {b}: Δ={diff:.4f}  95%CI [{lo:.4f}, {hi:.4f}]  "
            f"{('t=' + f'{tstat:.2f}, df=' + f'{df:.1f}  ') if HAVE_SCIPY else ''}"
            f"p={p:.4g} (Holm p={float(p_adj):.4g})  g={g:.3f}  (n={n1} vs {n2})"
        )
        rows.append({
            "metric": label,
            "group_a": a, "group_b": b,
            "n_a": n1, "n_b": n2,
            "mean_a": m1, "mean_b": m2,
            "diff_a_minus_b": diff,
            "se_diff": se,
            "df_welch": df,
            "t_stat": tstat,
            "p_value": p,
            "p_value_holm": float(p_adj),
            "ci_low": lo, "ci_high": hi,
            "hedges_g": g,
        })
    return rows

print("\n=== Pair-wise tests (across all groups; Welch two-sided where available) ===")
pairwise_rows = []
pairwise_rows += run_pairwise_tests("support_rate", "Support rate")
pairwise_rows += run_pairwise_tests("conservative_hallucination_rate", "Conservative hallucination rate")
pairwise_rows += run_pairwise_tests("strict_hallucination_rate", "Strict hallucination rate")

if SAVE_PAIRWISE_CSV and pairwise_rows:
    if not HAVE_PANDAS:
        print("[WARN] pandas not installed; cannot save pairwise CSV.")
    else:
        pd.DataFrame(pairwise_rows).to_csv(PAIRWISE_CSV_PATH, index=False)
        print(f"\n[Saved] Pair-wise test table → {PAIRWISE_CSV_PATH}")

# ------------------- bar plots: per-model means with 95% CI (rates) -------------------
if PLOT_PER_MODEL_BARS:
    sns.set_theme(style="whitegrid")
    LABEL_MAP = {"gpt5": "gpt-5", "gpt35": "gpt-3.5", "o3_mini": "OpenAI o3-mini"}

    plot_defs = [
        ("support_mean", "support_ci_low", "support_ci_high", "support_std",
         "Support rate", "support_by_model.png"),
        ("cons_hall_mean", "cons_hall_ci_low", "cons_hall_ci_high", "cons_hall_std",
         "Conservative hallucination rate", "cons_hall_by_model.png"),
        ("strict_hall_mean", "strict_hall_ci_low", "strict_hall_ci_high", "strict_hall_std",
         "Strict hallucination rate", "strict_hall_by_model.png"),
    ]
    for mean_col, lo_col, hi_col, std_col, title, fname in plot_defs:
        xs = [row["model"] for row in overall_summary]
        ys = [row[mean_col] for row in overall_summary]
        stds = [row[std_col] for row in overall_summary]
        yerr_low, yerr_up = [], []
        for row in overall_summary:
            mu = row[mean_col]; lo, hi = row[lo_col], row[hi_col]
            yerr_low.append(0.0 if math.isnan(lo) else max(0.0, mu - lo))
            yerr_up.append(0.0 if math.isnan(hi) else max(0.0, hi - mu))
        plt.figure(figsize=(6.8, 4.2))
        try:
            sns.barplot(x=[LABEL_MAP.get(x, x) for x in xs], y=ys, estimator=np.mean, errorbar=None)
        except TypeError:
            sns.barplot(x=[LABEL_MAP.get(x, x) for x in xs], y=ys, estimator=np.mean, ci=None)
        xpos = np.arange(len(xs))
        plt.errorbar(xpos, ys, yerr=[yerr_low, yerr_up], fmt="none", capsize=4, linewidth=1.2, color="black")
        top = max((ys[i] + yerr_up[i]) for i in range(len(ys))) if ys else 1.0
        plt.ylim(0, max(1.0, top + 0.08))
        for i, (mu, sd) in enumerate(zip(ys, stds)):
            sd_disp = 0.0 if (sd is None or (isinstance(sd, float) and math.isnan(sd))) else sd
            y_text = ys[i] + yerr_up[i] + 0.02
            plt.text(i, y_text, f"{mu*100:.1f}% (sd {sd_disp*100:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="semibold")
        plt.xlabel("")
        plt.ylabel(f"{title} (0–1.0)")
        plt.title(f"{title} by model (mean ± 95% CI)")
        plt.tight_layout()
        plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()

# ------------------- optional BM25 per-model bar plot -------------------
if PLOT_BM25_BARS:
    # Prefer top-k; if not available for any model, fall back to top-1
    have_topk = any(row.get("n_bm25_topk", 0) > 0 for row in overall_summary)
    metric = (
        ("bm25_topk_mean", "bm25_topk_ci_low", "bm25_topk_ci_high", "BM25 mean top-k score")
        if have_topk else
        ("bm25_top1_mean", "bm25_top1_ci_low", "bm25_top1_ci_high", "BM25 mean top-1 score")
    )
    mean_col, lo_col, hi_col, title = metric

    LABEL_MAP = {"gpt5": "gpt-5", "gpt35": "gpt-3.5", "o3_mini": "OpenAI o3-mini"}
    xs = [row["model"] for row in overall_summary]
    ys = [row[mean_col] for row in overall_summary]
    yerr_low, yerr_up = [], []
    for row in overall_summary:
        mu = row[mean_col]
        lo, hi = row[lo_col], row[hi_col]
        yerr_low.append(0.0 if (lo is None or math.isnan(lo)) else max(0.0, mu - lo))
        yerr_up.append(0.0 if (hi is None or math.isnan(hi)) else max(0.0, hi - mu))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6.8, 4.2))
    try:
        sns.barplot(x=[LABEL_MAP.get(x, x) for x in xs], y=ys, estimator=np.mean, errorbar=None)
    except TypeError:
        sns.barplot(x=[LABEL_MAP.get(x, x) for x in xs], y=ys, estimator=np.mean, ci=None)
    xpos = np.arange(len(xs))
    plt.errorbar(xpos, ys, yerr=[yerr_low, yerr_up], fmt="none", capsize=4, linewidth=1.2, color="black")
    
    # space for labels
    top = max((ys[i] + yerr_up[i]) for i in range(len(ys))) if ys else 1.0
    plt.ylim(0, top + 0.10 * max(1.0, top))

    # --- NEW: label = mean and 95% CI ---
    for i, row in enumerate(overall_summary):
        mu = row[mean_col]
        lo = row[lo_col]
        hi = row[hi_col]
        if (lo is None or math.isnan(lo)) or (hi is None or math.isnan(hi)):
            label = f"mean {mu:.3f}\n95% CI [n/a]"
        else:
            label = f"mean {mu:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]"
        y_text = ys[i] + yerr_up[i] + 0.02 * max(1.0, top)
        plt.text(i, y_text, label, ha="center", va="bottom", fontsize=9, fontweight="semibold")

    plt.xlabel("")
    plt.ylabel(title)
    plt.title(f"{title} by model (mean ± 95% CI)")
    plt.tight_layout()
    plt.savefig(BM25_BARS_OUTFILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {BM25_BARS_OUTFILE}")


# ------------------- box plots with mean + 95% CI annotations -------------------
if PLOT_BOXPLOTS:
    if not HAVE_PANDAS:
        print("[WARN] pandas not installed; box plots require pandas. `pip install pandas` to enable.")
    else:
        os.makedirs(BOXPLOT_OUTDIR, exist_ok=True)
        obs_df = pd.DataFrame([{
            "task_group": r["task_group"],
            "model": r["model"],
            "support_rate": r["support_rate"],
            "conservative_hallucination_rate": r["conservative_hallucination_rate"],
            "strict_hallucination_rate": r["strict_hallucination_rate"],
        } for r in records])

        def annotate_box_means(ax, df_metric, x_col, y_col, order):
            stats = []
            for i, grp in enumerate(order):
                vals = df_metric.loc[df_metric[x_col] == grp, y_col].dropna().values
                if len(vals) == 0:
                    stats.append((i, grp, np.nan, np.nan, np.nan)); continue
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                lo, hi = ci95_bounds(mu, sd, len(vals))
                stats.append((i, grp, mu, lo, hi))
            current_max = float(df_metric[y_col].max()) if len(df_metric) else 1.0
            ax.set_ylim(0.0, max(1.0, current_max + 0.08))
            for i, grp, mu, lo, hi in stats:
                if np.isnan(mu): continue
                grp_max = float(df_metric.loc[df_metric[x_col] == grp, y_col].max())
                y_text = min(grp_max + 0.03, ax.get_ylim()[1] - 0.01)
                label = f"μ={mu*100:.1f}%\n95% CI [{lo*100:.1f}%, {hi*100:.1f}%]"
                ax.text(i, y_text, label, ha="center", va="bottom", fontsize=9, weight="semibold")

        def plot_box_for_metric(metric_key, pretty_name, outfile):
            d = obs_df[["model", metric_key]].dropna().copy()
            d = d[d["model"].isin(MODELS)]
            sns.set_theme(style="whitegrid")
            plt.figure(figsize=(7.2, 4.8))
            ax = sns.boxplot(data=d, x="model", y=metric_key, order=MODELS, showfliers=False)
            ax.set_xlabel("Model"); ax.set_ylabel(f"{pretty_name} (0–1)")
            ax.set_title(f"{pretty_name} by model — distribution across task runs")
            annotate_box_means(ax, d.rename(columns={metric_key: "value"}), x_col="model", y_col="value", order=MODELS)
            plt.tight_layout(); plt.savefig(os.path.join(BOXPLOT_OUTDIR, outfile), dpi=300, bbox_inches="tight"); plt.close()

        plot_box_for_metric("support_rate", "Support rate", "box_support_rate_by_model.png")
        plot_box_for_metric("conservative_hallucination_rate", "Conservative hallucination rate", "box_conservative_hallucination_by_model.png")
        plot_box_for_metric("strict_hallucination_rate", "Strict hallucination rate", "box_strict_hallucination_by_model.png")

# ---------- RADAR: support rate by domain (grouped by model) ----------
def _norm_domain(d: str) -> str:
    s = (d or "").strip().replace(" ", "_").lower()
    if s in {"domain1", "1", "d1"}: return "Domain1"
    if s in {"domain2", "2", "d2"}: return "Domain2"
    if s in {"domain3", "3", "d3"}: return "Domain3"
    if s in {"domain4", "4", "d4"}: return "Domain4"
    if s in {"domain5", "5", "d5"}: return "Domain5"
    if s in {"overallconclusion", "overall_conclusion", "overall", "overall_bias", "overall_risk_of_bias"}:
        return "OverallConclusion"
    return d  # pass through unknowns

def _plot_radar(series_dict, axes_labels, title, outfile, legend_fontsize=8):
    N = len(axes_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = plt.subplot(111, polar=True)
    for label, vals in series_dict.items():
        vals = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v) for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, marker="o", label=label)
        ax.fill(angles, vals, alpha=0.10)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8]); ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10), fontsize=legend_fontsize)
    plt.tight_layout(); plt.savefig(outfile, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[Saved] {outfile}")

if HAVE_PANDAS:
    domain_pattern = os.path.join(root_path, "*", "*", "tables", "per_domain.csv")
    domain_files = sorted(glob.glob(domain_pattern))
    per_domain_rows = []
    for fpath in domain_files:
        rel = os.path.relpath(fpath, root_path)
        parts = rel.split(os.sep)
        if len(parts) < 4:
            continue
        task_atomic, model_folder = parts[0], parts[1]
        model = canonical_model_label(model_folder)
        if model not in MODELS: continue
        try:
            df_dom = pd.read_csv(fpath)
        except Exception as e:
            print(f"[WARN] cannot read {fpath}: {e}"); continue
        if "domain" not in df_dom.columns or "support_rate" not in df_dom.columns:
            cols = {c.lower(): c for c in df_dom.columns}
            dcol = cols.get("domain"); scol = cols.get("support_rate")
            if not dcol or not scol:
                print(f"[WARN] {fpath} lacks needed columns; skipping."); continue
            df_dom = df_dom.rename(columns={dcol: "domain", scol: "support_rate"})
        for _, r in df_dom.iterrows():
            dname = _norm_domain(str(r["domain"]))
            if dname not in RADAR_DOMAINS: continue
            rate = as_rate(r["support_rate"])
            if rate is None: continue
            per_domain_rows.append({"model": model, "domain": dname, "support_rate": rate})
    if per_domain_rows:
        df_pd = pd.DataFrame(per_domain_rows)
        agg = df_pd.groupby(["model", "domain"])["support_rate"].agg(["count", "mean", "std"]).reset_index()
        lo, hi = [], []
        for _, row in agg.iterrows():
            mu, sd, n = row["mean"], row["std"], int(row["count"])
            ci_lo, ci_hi = ci95_bounds(mu, sd if not np.isnan(sd) else 0.0, n)
            lo.append(ci_lo); hi.append(ci_hi)
        agg["ci_low"] = lo; agg["ci_high"] = hi
        out_csv = "/Users/liangz2/Documents/gpt_reviewer/hall_result/support_rate_by_domain_and_model.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        agg.to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")
        series_dict = {}
        for m in MODELS:
            pretty = PRETTY[m]
            vals = []
            for d in RADAR_DOMAINS:
                sub = agg[(agg["model"] == m) & (agg["domain"] == d)]
                vals.append(float(sub["mean"].iloc[0]) if len(sub) else np.nan)
            series_dict[pretty] = vals
        _plot_radar(series_dict, RADAR_DOMAINS, "Mean of support rate of RoB domains", RADAR_SUPPORT_OUTFILE, legend_fontsize=8)
    else:
        print("[INFO] No per_domain.csv files found — skipping radar plot.")
else:
    print("[WARN] pandas not installed; cannot build radar plot from per-domain CSVs.")
