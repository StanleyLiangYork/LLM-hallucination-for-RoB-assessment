#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized hallucination QA+Evidence:
- For each study (subfolder of --json), read its machine-response JSON once.
- Read the corresponding PDF once from --pdf_root/<study_name>/*.pdf
- Build a single BM25 index over the PDF text (chunked with overlap).
- Loop over all items in the JSON, query BM25, and ask GPT-5 for a verdict.

NEW:
- Before processing a study, check --out_json for an existing output JSON with results.
  If found (and non-empty), print and skip this study.

Output: one JSON file per study saved in --out_json, same base filename as input.
"""

import os
import re
import json
import glob
import argparse
import time
from typing import List, Dict, Any, Tuple

import pdfplumber
from openai import OpenAI, BadRequestError

# ---------- BM25 ----------
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception as _e:
    _HAS_BM25 = False
    raise ImportError(
        "rank_bm25 is required. Install with: pip install rank-bm25"
    )

# ---------------------------
# Model temperature handling
# ---------------------------
MODELS_WITH_TEMPERATURE = {
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    # add other non-reasoning chat models here if needed
}
def _supports_temperature(model: str) -> bool:
    base = (model or "").lower().strip()
    return base in MODELS_WITH_TEMPERATURE

# ---------------------------
# Question-id → question text mapping (normalized keys)
# ---------------------------
def norm_qid(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")

QID_TO_QUESTION: Dict[str, str] = {
    # Domain 1
    "1.1": "Was the allocation sequence random?",
    "1.2": "Was the allocation sequence concealed until participants were enrolled and assigned to interventions?",
    "1.3": "Did baseline differences between intervention groups suggest a problem with the randomization process?",
    "domain_1_conclusion": "Risk-of-bias judgement, based on 1.1–1.3, for bias arising from the randomization process",

    # Domain 2
    "2.1": "Were participants aware of their assigned intervention during the trial?",
    "2.2": "Were carers and people delivering the interventions aware of participants' assigned intervention during the trial?",
    "2.3": "If 2.1 or 2.2 is Yes/Probably Yes/No information: Were there deviations from the intended intervention that arose because of the trial context?",
    "2.4": "If 2.3 is Yes/Probably Yes: Were these deviations likely to have affected the outcome?",
    "2.5": "If 2.4 is Yes/Probably Yes/No information: Were these deviations from intended intervention balanced between groups?",
    "2.6": "Was an appropriate analysis used to estimate the effect of assignment to intervention?",
    "2.7": "If 2.6 is No/Probably No/No information: Was there potential for a substantial impact of failure to analyse participants in their randomized groups?",
    "domain_2_conclusion": "Risk-of-bias judgement, based on 2.1–2.7, for deviations from intended interventions (effect of assignment)",

    # Domain 3
    "3.1": "Were data for this outcome available for all, or nearly all, participants randomized?",
    "3.2": "If 3.1 is No/Probably No/No information: Is there evidence that the result was not biased by missing outcome data?",
    "3.3": "If 3.2 is No/Probably No: Could missingness in the outcome depend on its true value?",
    "3.4": "If 3.3 is Yes/Probably Yes/No information: Is it likely that missingness in the outcome depended on its true value?",
    "domain_3_conclusion": "Risk-of-bias judgement, based on 3.1–3.4, for missing outcome data",

    # Domain 4
    "4.1": "Was the method of measuring the outcome inappropriate?",
    "4.2": "Could measurement or ascertainment of the outcome have differed between intervention groups?",
    "4.3": "If 4.1 and 4.2 are No/Probably No/No information: Were outcome assessors aware of the intervention received?",
    "4.4": "If 4.3 is Yes/Probably Yes/No information: Could assessment of the outcome have been influenced by knowledge of intervention received?",
    "4.5": "If 4.4 is Yes/Probably Yes/No information: Is it likely that assessment of the outcome was influenced by knowledge of intervention received?",
    "domain_4_conclusion": "Risk-of-bias judgement, based on 4.1–4.5, for measurement of the outcome",

    # Domain 5
    "5.1": "Were the data that produced this result analysed in accordance with a pre-specified analysis plan finalized before unblinded outcome data were available?",
    "5.2": "Is the numerical result likely to have been selected from multiple eligible outcome measurements (scales/definitions/time points) within the domain?",
    "5.3": "Is the numerical result likely to have been selected from multiple eligible analyses of the data?",
    "domain_5_conclusion": "Risk-of-bias judgement, based on 5.1–5.3, for selection of the reported result",

    # Overall
    "overall_risk_of_bias": "Based on all domains, assess the overall risk of bias",
}
QID_TO_QUESTION[norm_qid("Overall risk of bias")] = QID_TO_QUESTION["overall_risk_of_bias"]

def question_text_for_qid(qid: str) -> str:
    return QID_TO_QUESTION.get(norm_qid(qid), "")

# ---------------------------
# Text utils
# ---------------------------
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

def normalize_ws(s: str) -> str:
    return _WS_RE.sub(" ", s or "").strip()

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]

def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
        i += max(1, chunk_size - overlap)
    return chunks

# ---------------------------
# PDF reading
# ---------------------------
def read_pdf_text(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)

# ---------------------------
# BM25 build & search
# ---------------------------
def build_bm25_index(chunks: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [tokenize(c) for c in chunks]
    index = BM25Okapi(tokenized)
    return index, tokenized

def bm25_retrieve(index: BM25Okapi, tokenized_corpus: List[List[str]],
                  query_text: str, corpus_chunks: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    q_tokens = tokenize(query_text)
    scores = index.get_scores(q_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for rank, i in enumerate(ranked, start=1):
        results.append({
            "rank": rank,
            "score": float(scores[i]),
            "chunk_id": int(i),
            "snippet": corpus_chunks[i]
        })
    return results

# ---------------------------
# GPT call
# ---------------------------
def call_gpt(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Calls Chat Completions. If model rejects 'temperature', we omit it and retry once.
    Expects a JSON object in the response; otherwise returns a default.
    """
    kwargs = {"model": model, "messages": messages}
    if _supports_temperature(model):
        kwargs["temperature"] = 0

    try:
        resp = client.chat.completions.create(**kwargs)
    except BadRequestError as e:
        msg = getattr(e, "message", "") or str(e)
        if "temperature" in msg.lower():
            kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            raise

    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {
            "verdict": "Not Found",
            "quote": "",
            "rationale": "Model returned non-JSON; defaulted to Not Found.",
            "raw": content,
        }

# ---------------------------
# Verdict prompt builder
# ---------------------------
SYSTEM_PROMPT = """You are a reviewer of medical literature. 
Given a machine response (label), a human response (gold), the exact question asked, and retrieved paper excerpts, 
decide whether the machine's claim is Supported, Contradicted, Not Found, or Out of Scope.
Return strict JSON only."""

def make_user_prompt(study: str,
                     question_id: str,
                     question_text: str,
                     machine_resp: str,
                     human_resp: str,
                     explanation_text: str,
                     retrieved_chunks: List[Dict[str, Any]]) -> str:
    ctx_blocks = []
    for r in retrieved_chunks:
        ctx_blocks.append(
            f"[Chunk {r['rank']} | BM25 score={r['score']:.4f}]\n{r['snippet']}"
        )
    ctx_text = "\n\n".join(ctx_blocks)

    prompt = f"""
Study: {study}
Question ID: {question_id}
Question: {question_text or '(unknown question text)'} 

Machine response (label): {machine_resp}
Human (gold) response: {human_resp}

Machine explanation (comment+reasoning):
\"\"\"{explanation_text.strip()}\"\"\"

Paper excerpts (no pages; chunked):
{ctx_text}

Task: Considering the specific question above, judge the machine response relative to the study text.
Answer with this JSON schema only:
{{
  "verdict": "Supported | Contradicted | Not Found | Out of Scope",
  "quote": "A minimal quote supporting your verdict (empty if Not Found/Out of Scope)",
  "rationale": "One sentence explaining why"
}}
"""
    return prompt.strip()

# ---------------------------
# Study I/O helpers
# ---------------------------
def find_pdf_for_study(pdf_root: str, study_name: str) -> str:
    study_dir = os.path.join(pdf_root, study_name)
    if not os.path.isdir(study_dir):
        return ""
    pdfs = sorted(glob.glob(os.path.join(study_dir, "*.pdf")))
    return pdfs[0] if pdfs else ""

def find_json_in_folder(folder: str) -> str:
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    return files[0] if files else ""

def expected_out_path(out_dir: str, in_json_path: str) -> str:
    return os.path.join(out_dir, os.path.basename(in_json_path)) if in_json_path else ""

def _existing_result_count(path: str) -> int:
    """
    Return number of items if `path` exists and is a JSON list; else 0.
    """
    try:
        if not (path and os.path.isfile(path) and os.path.getsize(path) > 0):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        return 0
    except Exception:
        return 0

# ---------------------------
# Study processing
# ---------------------------
def process_study(study_dir: str,
                  pdf_root: str,
                  client: OpenAI,
                  model: str,
                  out_dir: str,
                  top_k: int = 5,
                  chunk_size: int = 1400,
                  overlap: int = 200) -> None:
    t0 = time.time()

    study_name = os.path.basename(study_dir.rstrip("/"))
    in_json_path = find_json_in_folder(study_dir)
    if not in_json_path:
        print(f"[WARN] No JSON in {study_dir}; skipping.")
        return

    # --- NEW: skip if already evaluated ---
    out_path = expected_out_path(out_dir, in_json_path)
    n_exist = _existing_result_count(out_path)
    if n_exist > 0:
        print(f"[SKIP] Study={study_name} already evaluated ({n_exist} items): {out_path}")
        return

    pdf_path = find_pdf_for_study(pdf_root, study_name)
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[WARN] No PDF for {study_name} at {pdf_root}/{study_name}; skipping.")
        return

    with open(in_json_path, "r", encoding="utf-8") as f:
        try:
            response_items = json.load(f)
        except Exception as e:
            print(f"[WARN] Cannot parse JSON: {in_json_path} ({e}); skipping.")
            return
    if not isinstance(response_items, list):
        print(f"[WARN] JSON not a list: {in_json_path}; skipping.")
        return

    # --- READ PDF ONCE ---
    print(f"[RUN] Study={study_name}  JSON={os.path.basename(in_json_path)}  PDF=OK")
    paper_text = read_pdf_text(pdf_path)
    corpus_chunks = chunk_text(paper_text, chunk_size=chunk_size, overlap=overlap)
    if not corpus_chunks:
        print(f"[WARN] Empty text extracted from {pdf_path}; skipping.")
        return
    t_pdf = time.time()

    # --- BUILD BM25 ONCE ---
    bm25_idx, tokenized_corpus = build_bm25_index(corpus_chunks)
    print(f"[INFO] BM25 built: {len(corpus_chunks)} chunks.")
    t_bm25 = time.time()

    # --- LOOP ALL ITEMS USING SAME INDEX ---
    results = []
    gpt_time_accum = 0.0
    for item in response_items:
        qid = str(item.get("question_id", ""))
        qtext = question_text_for_qid(qid)

        resp_label = str(item.get("response", ""))
        topic = str(item.get("topic", ""))  # may be blank in some files
        gold = item.get("gold", {}) or {}
        gold_label = str(gold.get("response", ""))

        # combine comment + reasoning only (skip any 'gold' comments)
        comment = normalize_ws(item.get("comment", ""))
        reasoning = normalize_ws(item.get("reasoning", "")) or ""
        expl = normalize_ws(f"{comment} {reasoning}".strip())

        # Build a stronger BM25 query: question text + machine label + explanation
        query_bits = [qtext, resp_label, expl]
        query_text = normalize_ws(" ".join([q for q in query_bits if q]))

        retrieved = bm25_retrieve(bm25_idx, tokenized_corpus, query_text, corpus_chunks, top_k=top_k)

        # Simple retrieval metrics
        bm25_scores = [r["score"] for r in retrieved]
        bm25_top = float(bm25_scores[0]) if bm25_scores else 0.0
        bm25_mean = float(sum(bm25_scores) / len(bm25_scores)) if bm25_scores else 0.0

        # Compose GPT prompt once per item
        user_prompt = make_user_prompt(
            study=study_name,
            question_id=qid,
            question_text=qtext,
            machine_resp=resp_label,
            human_resp=gold_label,
            explanation_text=expl,
            retrieved_chunks=retrieved
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        tg0 = time.time()
        verdict = call_gpt(client, model, messages)
        gpt_time_accum += (time.time() - tg0)

        results.append({
            "study": study_name,
            "json_file": os.path.basename(in_json_path),
            "pdf_file": os.path.basename(pdf_path),
            "question_id": qid,
            "question_text": qtext,
            "machine_response": resp_label,
            "human_response": gold_label,
            "topic": topic,
            "explanation_used": expl,
            "bm25_topk_scores": bm25_scores,
            "bm25_top_score": bm25_top,
            "bm25_mean_topk": bm25_mean,
            "retrieved": retrieved,
            "verdict": verdict
        })

    # Save one output per study, reusing the input JSON base name
    os.makedirs(out_dir, exist_ok=True)
    out_path = expected_out_path(out_dir, in_json_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] → {out_path}")

    # --- TIMING REPORT ---
    t_end = time.time()
    total = t_end - t0
    t_pdf_chunk = t_pdf - t0
    t_bm_only = t_bm25 - t_pdf
    t_gpt = gpt_time_accum
    print(f"[TIME] Study={study_name} total={total:.3f}s "
          f"(pdf+chunk={t_pdf_chunk:.3f}s, bm25={t_bm_only:.3f}s, gpt_calls={t_gpt:.3f}s, n_items={len(results)})")

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Optimized hallucination QA with shared BM25 per study.")
    ap.add_argument("--json", required=True,
                    help="Root folder containing per-study subfolders with machine-response JSON inside.")
    ap.add_argument("--pdf_root", required=True,
                    help="Root folder containing per-study subfolders with the PDF(s).")
    ap.add_argument("--api_key", required=True,
                    help="Path to a text file containing the OpenAI API key.")
    ap.add_argument("--model", default="gpt-5",
                    help="Model name (e.g., gpt-5, gpt-4o, gpt-3.5-turbo).")
    ap.add_argument("--out_json", required=True,
                    help="Output folder for verdict JSONs.")
    ap.add_argument("--top_k", type=int, default=5, help="BM25 top-k chunks to pass to GPT.")
    ap.add_argument("--chunk_size", type=int, default=1400, help="Chunk size (characters).")
    ap.add_argument("--overlap", type=int, default=200, help="Chunk overlap (characters).")
    return ap.parse_args()

def main():
    args = parse_args()

    with open(args.api_key, "r", encoding="utf-8") as f:
        key = f.read().strip()
    client = OpenAI(api_key=key)

    # Ensure output directory exists (for existence checks & saves)
    os.makedirs(args.out_json, exist_ok=True)

    # Walk study subfolders within --json
    study_dirs = [p for p in glob.glob(os.path.join(args.json, "*")) if os.path.isdir(p)]
    if not study_dirs:
        print(f"[WARN] No subfolders under {args.json}")
        return

    for study_dir in study_dirs:
        # quick pre-check to avoid entering process_study at all
        in_json_path = find_json_in_folder(study_dir)
        if in_json_path:
            out_path = expected_out_path(args.out_json, in_json_path)
            n_exist = _existing_result_count(out_path)
            if n_exist > 0:
                study_name = os.path.basename(study_dir.rstrip("/"))
                print(f"[SKIP] Study={study_name} already evaluated ({n_exist} items): {out_path}")
                continue

        process_study(
            study_dir=study_dir,
            pdf_root=args.pdf_root,
            client=client,
            model=args.model,
            out_dir=args.out_json,
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )

if __name__ == "__main__":
    main()
