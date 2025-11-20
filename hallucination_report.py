#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hallucination_report.py

Parse verdict JSONs (one file per study, each a list of items) and compute:
- Overall counts/rates per verdict class
- Hallucination rates:
    * strict:  (Contradicted + Not Found + Out of Scope) / N
    * conservative: (Contradicted + Not Found) / N
- Per-study and per-domain breakdowns
- (Optional) agreement-aware rates (machine vs human label equality)
- (Optional) retrieval stats if present (BM25 top score)

Usage:
  python hallucination_report.py --input_dir /path/to/verdict_jsons --save_csv /path/to/outdir
"""

import os
import json
import argparse
from collections import Counter

try:
    import pandas as pd
    _HAS_PD = True
except Exception:
    _HAS_PD = False


# ---------------------------
# Helpers
# ---------------------------
def _norm(s):
    # robust stringify + strip
    if isinstance(s, str):
        return s.strip()
    if s is None:
        return ""
    # last resort
    return str(s).strip()

def _norm_lower(s):
    return _norm(s).lower()

def _verdict_normalize(vtext):
    """
    Normalize verdict label into one of:
    Supported, Contradicted, Not Found, Out of Scope, Unknown
    """
    v = _norm_lower(vtext)
    if v == "supported":
        return "Supported"
    if v == "contradicted":
        return "Contradicted"
    if v in {"not found", "not_found", "notfound"}:
        return "Not Found"
    if v in {"out of scope", "out_of_scope", "outofscope"}:
        return "Out of Scope"
    return "Unknown" if not v else v.title()

def _extract_verdict_text(verdict_value):
    """
    Verdict can be:
      - string: "Supported"
      - dict: {"verdict":"Supported", "quote":"...", "rationale":"..."}
      - list/tuple of any of the above (be forgiving)
    Return a string best-effort.
    """
    if isinstance(verdict_value, str):
        return verdict_value
    if isinstance(verdict_value, dict):
        for k in ("verdict", "label", "decision", "result"):
            if k in verdict_value and isinstance(verdict_value[k], str):
                return verdict_value[k]
        # fall back to stringified dict
        return str(verdict_value)
    if isinstance(verdict_value, (list, tuple)):
        for el in verdict_value:
            s = _extract_verdict_text(el)
            if s:
                return s
        return ""
    # fallback
    return str(verdict_value) if verdict_value is not None else ""

def _domain_for_qid(qid: str) -> str:
    q = _norm_lower(qid).replace(" ", "_")
    if q.startswith("1.") or q.startswith("domain_1"):
        return "Domain1"
    if q.startswith("2.") or q.startswith("domain_2"):
        return "Domain2"
    if q.startswith("3.") or q.startswith("domain_3"):
        return "Domain3"
    if q.startswith("4.") or q.startswith("domain_4"):
        return "Domain4"
    if q.startswith("5.") or q.startswith("domain_5"):
        return "Domain5"
    if q in {"overall_risk_of_bias", "overall", "overall_bias"}:
        return "OverallConclusion"
    return "Other"

def _to_str(x):
    return x if isinstance(x, str) else _norm(x)

def _bm25_top_score(item):
    """
    Robustly pull a 'top score' from several possible layouts:
      - bm25_top_score: float
      - bm25_topk_scores: [floats]
      - bm25_hits: [{"score": ...}, ...]
      - retrieved: [{"score": ...}, ...]
    Return float or None.
    """
    try:
        if "bm25_top_score" in item:
            return float(item["bm25_top_score"])
    except Exception:
        pass

    try:
        if "bm25_topk_scores" in item and isinstance(item["bm25_topk_scores"], list):
            vals = [float(x) for x in item["bm25_topk_scores"]]
            return max(vals) if vals else None
    except Exception:
        pass

    for key in ("bm25_hits", "retrieved"):
        try:
            hits = item.get(key, None)
            if isinstance(hits, list):
                vals = []
                for h in hits:
                    if isinstance(h, dict) and "score" in h:
                        try:
                            vals.append(float(h["score"]))
                        except Exception:
                            pass
                if vals:
                    return max(vals)
        except Exception:
            pass

    return None


# ---------------------------
# Core loader
# ---------------------------
def load_rows(path: str):
    """
    Returns a list of 'rows' (dict) with normalized fields:
      study, question_id, domain, model_response, human_response, verdict, is_strict_hallucination, is_conservative_hallucination, bm25_top_score
    """
    files = []
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.lower().endswith(".json"):
                files.append(os.path.join(path, f))
    else:
        files.append(path)

    rows = []
    for fp in sorted(files):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {fp}: {e}")
            continue

        # If a file contains a single dict, wrap it for consistency
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            print(f"[WARN] Unexpected JSON structure in {fp}; expected list.")
            continue

        for item in data:
            # study naming: accept study_name or study, else from file stem
            study = item.get("study_name") or item.get("study") or os.path.splitext(os.path.basename(fp))[0]
            qid = _to_str(item.get("question_id", ""))
            dom = _domain_for_qid(qid)

            # machine/human label can sometimes be oddly typed; coerce to str
            mresp = _to_str(item.get("model_response", item.get("machine_response", "")))
            hresp = _to_str(item.get("human_response", ""))

            # verdict might be string or dict
            verdict_text = _extract_verdict_text(item.get("verdict", ""))
            vrd = _verdict_normalize(verdict_text)

            is_strict = int(vrd in {"Contradicted", "Not Found", "Out of Scope"})
            is_cons = int(vrd in {"Contradicted", "Not Found"})

            rows.append({
                "study": study,
                "file": os.path.basename(fp),
                "question_id": qid,
                "domain": dom,
                "model_response": mresp,
                "human_response": hresp,
                "verdict": vrd,
                "is_strict_hallucination": is_strict,
                "is_conservative_hallucination": is_cons,
                "agree_with_human": int(_norm_lower(mresp) == _norm_lower(hresp) and _norm_lower(hresp) != ""),
                "bm25_top_score": _bm25_top_score(item),
            })
    return rows


# ---------------------------
# Aggregations
# ---------------------------
def _summarize_block(rows):
    N = len(rows)
    vc = Counter(r["verdict"] for r in rows)
    strict_h = sum(r["is_strict_hallucination"] for r in rows)
    cons_h = sum(r["is_conservative_hallucination"] for r in rows)
    support = vc.get("Supported", 0)

    def rate(x): 
        return (x / N) if N else 0.0

    return {
        "n_items": N,
        "supported": support,
        "contradicted": vc.get("Contradicted", 0),
        "not_found": vc.get("Not Found", 0),
        "out_of_scope": vc.get("Out of Scope", 0),
        "strict_hallucinations": strict_h,
        "conservative_hallucinations": cons_h,
        "support_rate": round(rate(support), 4),
        "conservative_hallucination_rate": round(rate(cons_h), 4),
        "strict_hallucination_rate": round(rate(strict_h), 4),
    }

def summarize(rows):
    overall = _summarize_block(rows)

    # per-study
    per_study = {}
    for s in sorted(set(r["study"] for r in rows)):
        rows_s = [r for r in rows if r["study"] == s]
        per_study[s] = {"overall": _summarize_block(rows_s)}

    # per-domain
    per_domain = {}
    for d in sorted(set(r["domain"] for r in rows)):
        rows_d = [r for r in rows if r["domain"] == d]
        per_domain[d] = {"overall": _summarize_block(rows_d)}

    # agreement-aware
    agree = [r for r in rows if r["agree_with_human"] == 1]
    disagree = [r for r in rows if r["agree_with_human"] == 0]
    agree_sum = _summarize_block(agree) if agree else None
    disagree_sum = _summarize_block(disagree) if disagree else None

    # verdict-wise BM25 (if present)
    bm25_by_verdict = {}
    for v in ["Supported", "Contradicted", "Not Found", "Out of Scope"]:
        vals = [r["bm25_top_score"] for r in rows if r["verdict"] == v and r.get("bm25_top_score") is not None]
        if vals:
            bm25_by_verdict[v] = {
                "count": len(vals),
                "mean_top_score": sum(vals)/len(vals),
                "min_top_score": min(vals),
                "max_top_score": max(vals),
            }

    return {
        "overall": overall,
        "per_study": per_study,
        "per_domain": per_domain,
        "agreement_split": {
            "agree_with_human": agree_sum,
            "disagree_with_human": disagree_sum
        },
        "bm25_by_verdict": bm25_by_verdict
    }


def to_dataframe(rows):
    if not _HAS_PD:
        return None
    return pd.DataFrame(rows)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Quantify hallucination from verdict JSONs.")
    ap.add_argument("--input_dir", required=True, help="Folder of verdict JSONs (or a single JSON file).")
    ap.add_argument("--save_csv", default=None, help="If set, write rows.csv, per_study.csv, per_domain.csv here.")
    return ap.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.input_dir)
    if not rows:
        print("[ERROR] No rows loaded. Check --input_dir.")
        return

    # Print headline
    summary = summarize(rows)
    ov = summary["overall"]
    print("\n=== OVERALL ===")
    print(f"N items: {ov['n_items']}")
    print(f"Supported: {ov['supported']}  "
          f"Contradicted: {ov['contradicted']}  "
          f"Not Found: {ov['not_found']}  "
          f"Out of Scope: {ov['out_of_scope']}")
    print(f"Support rate: {ov['support_rate']:.3f}")
    print(f"Conservative hallucination rate: {ov['conservative_hallucination_rate']:.3f}")
    print(f"Strict hallucination rate:       {ov['strict_hallucination_rate']:.3f}")

    # By study
    print("\n=== BY STUDY (strict hallucination rate) ===")
    for s, sm in sorted(summary["per_study"].items(), key=lambda kv: kv[0]):
        print(f"{s:40s}  n={sm['overall']['n_items']:3d}  "
              f"strict={sm['overall']['strict_hallucination_rate']:.3f}  "
              f"support={sm['overall']['support_rate']:.3f}")

    # By domain
    print("\n=== BY DOMAIN (strict hallucination rate) ===")
    for d, sm in sorted(summary["per_domain"].items(), key=lambda kv: kv[0]):
        print(f"{d:18s}  n={sm['overall']['n_items']:3d}  "
              f"strict={sm['overall']['strict_hallucination_rate']:.3f}  "
              f"support={sm['overall']['support_rate']:.3f}")

    # Agreement-aware
    print("\n=== AGREEMENT SPLIT ===")
    ag = summary["agreement_split"]["agree_with_human"]
    dg = summary["agreement_split"]["disagree_with_human"]

    if ag:
        print(f"Agree w/ human:     n={ag['n_items']:3d}  "
              f"strict={ag['strict_hallucination_rate']:.3f}  "
              f"support={ag['support_rate']:.3f}")
    else:
        print("Agree w/ human:     n=0")

    if dg:
        print(f"Disagree w/ human:  n={dg['n_items']:3d}  "
              f"strict={dg['strict_hallucination_rate']:.3f}  "
              f"support={dg['support_rate']:.3f}")
    else:
        print("Disagree w/ human:  n=0")

    # BM25 by verdict (if present)
    if summary["bm25_by_verdict"]:
        print("\n=== BM25 TOP-SCORE BY VERDICT (if available) ===")
        for v, stats in summary["bm25_by_verdict"].items():
            print(f"{v:14s} count={stats['count']:3d}  "
                  f"mean_top={stats['mean_top_score']:.3f}  "
                  f"min={stats['min_top_score']:.3f}  "
                  f"max={stats['max_top_score']:.3f}")

    # Optional CSVs
    if args.save_csv:
        os.makedirs(args.save_csv, exist_ok=True)
        if _HAS_PD:
            df = to_dataframe(rows)
            df.to_csv(os.path.join(args.save_csv, "rows.csv"), index=False)

            roll_cols = ["n_items", "supported", "contradicted", "not_found", "out_of_scope",
                         "support_rate", "conservative_hallucination_rate", "strict_hallucination_rate"]

            # Per study
            per_study_rows = []
            for s, sm in summary["per_study"].items():
                row = {"study": s}
                row.update(sm["overall"])
                per_study_rows.append(row)
            pd.DataFrame(per_study_rows)[["study"] + roll_cols].to_csv(
                os.path.join(args.save_csv, "per_study.csv"), index=False
            )

            # Per domain
            per_domain_rows = []
            for d, sm in summary["per_domain"].items():
                row = {"domain": d}
                row.update(sm["overall"])
                per_domain_rows.append(row)
            pd.DataFrame(per_domain_rows)[["domain"] + roll_cols].to_csv(
                os.path.join(args.save_csv, "per_domain.csv"), index=False
            )

            with open(os.path.join(args.save_csv, "overall_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary["overall"], f, indent=2)
            print(f"\n[WROTE] CSVs to {args.save_csv}")
        else:
            print("[WARN] pandas not installed; skipping CSV export.")


if __name__ == "__main__":
    main()
