# Retrieve-Then-Verify for Evaluating Evidence Support and Hallucination in Large Language Model–Generated Medical Information: Empirical Study

**Authors:** Zhaohui Liang<sup>1</sup>, PhD; Cynthia Sheffield<sup>2*</sup>, MBA; Gisela Butera<sup>2</sup>, M.Ed; Sameer Antani<sup>1*</sup>, PhD

**Affiliations**

1. Division of Intramural Research, National Library of Medicine, National Institutes of Health (NIH), Bethesda, MD, USA
2. Division of Library Services, Office of Research Services, NIH, Bethesda, MD, USA

\* Corresponding authors

## Abstract

**Background:** Despite high reported accuracy on clinical and evidence appraisal tasks, AI-generated medical information may lack explicit support from source documents. This creates challenges for digital health practitioners regarding transparency, auditability, and trust when AI systems are used for evidence synthesis, guideline development, and clinical knowledge management. Large language models (LLMs) can generate fluent and seemingly correct outputs, but existing evaluations often rely on agreement with human judgments and do not directly assess whether AI-generated content is grounded in underlying evidence.

**Objective:** This study measures evidence support and hallucination in AI-generated medical information by assessing the extent to which LLM-generated risk-of-bias assessments are supported by source clinical trial reports.

**Methods:** We evaluated 3 LLMs (GPT-5, OpenAI o3-mini, and GPT-3.5) on risk-of-bias (RoB 2) assessment using all 97 randomized controlled trials for which the source Cochrane systematic review provided complete human RoB 2 annotations and full-text reports were accessible, constituting the complete reference set. No train-validation split was applied; all 97 studies were used for evaluation. Model outputs were constrained to structured RoB 2 signaling questions and domain-level judgments. For each generated claim, relevant text passages were retrieved from trial reports using the Okapi BM25 (Best Matching 25) algorithm. A verification step assigned evidence verdicts (supported, contradicted, not found, or out of scope) with verbatim quotations. We quantified evidence support rates and conservative and strict hallucination rates. Task performance was evaluated using exact and binary accuracy, sensitivity, specificity, F1-score, Youden J, and agreement with human reviewers using Cohen kappa and Fleiss kappa.

**Results:** Binary accuracy of AI-generated risk-of-bias judgments was high across domains (90%-98%), whereas exact accuracy was substantially lower (42%-71%), reflecting frequent disagreements in severity classification despite correct directional classification. GPT-5 achieved the strongest overall performance, including perfect binary accuracy for overall risk-of-bias conclusions and the highest agreement with human reviewers (quadratic kappa up to 0.81). However, evidence support rates across models ranged from only 60% to 65%, with conservative hallucination rates of 34%-37%. GPT-5 showed the highest mean evidence support (64.3%) and the lowest strict hallucination rate (35.7%). Mean top-1 BM25 retrieval scores were similar across models (approximately 30-31), suggesting that differences in hallucination were not primarily attributable to differences in retrieval strength.

**Conclusions:** AI-generated medical information can achieve high decision-level accuracy while still lacking documentary support in a substantial proportion of outputs. Measuring evidence support and hallucination reveals important limitations that are not captured by agreement metrics alone. Retrieval-based evidence verification provides a reproducible and transparent approach for evaluating the reliability of AI-generated medical information, with direct relevance to digital health practice, evidence-based medicine, and medical informatics.

**Keywords:** AI; large language models; digital health; medical informatics; evidence-based medicine; information quality; information retrieval; hallucinations; systematic review; risk-of-bias

## Overview

This repository implements the paper's retrieve-then-verify pipeline:

1. Generate structured Cochrane RoB 2 assessments from trial-report PDFs with a user-selected OpenAI model.
2. Extract the human RoB 2 reference annotations from the Cochrane data-row CSV without an additional model call.
3. Retrieve the top 5 report passages for each generated claim using Okapi BM25.
4. Use a configurable OpenAI judge to assign Supported, Contradicted, Not Found, or Out of Scope.
5. Export item-level data and aggregate evidence-support and hallucination reports.

The implementation is based on and supersedes the initial scripts in the [antani-lab reference repository](https://github.com/antani-lab/LLM-hallucination-for-RoB-assessment). It fixes batch topic handling, preserves all top-k evidence and scores, validates response labels and numerical mappings, supports resumable runs, and separates deterministic data parsing from model-based verification.

## Repository layout

```text
.
├── get_robv2.py              # Generate model RoB 2 assessments
├── extract_human_rob.py      # Parse Cochrane human RoB 2 annotations
├── hallucination_qe.py       # BM25 retrieval and item-level evidence verdicts
├── hallucination_report.py   # CSV/JSON aggregate reports
├── rob_pipeline/             # Shared schemas and utilities
├── tests/                    # Offline unit tests
└── data/
    ├── CD014722-data-rows.csv
    ├── human_review_structured/   # Deterministically extracted human annotations
    └── pdfs/<Study>/<Author_Year>.pdf
```

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the OpenAI key in the environment. A key may instead be read from an untracked file with `--api-key-file`.

```bash
export OPENAI_API_KEY="your-key"
```

The selected model is passed unchanged to the OpenAI API. GPT-5, OpenAI o3-mini, and GPT-4o support Structured Outputs; the client falls back to JSON mode or prompt-constrained JSON for models such as legacy GPT-3.5 Turbo that do not support the same schema feature. The API model identifier for OpenAI o3-mini is `o3-mini`. See the official [GPT-5](https://developers.openai.com/api/docs/models/gpt-5), [o3-mini](https://developers.openai.com/api/docs/models/o3-mini), and [GPT-3.5 Turbo](https://developers.openai.com/api/docs/models/gpt-3.5-turbo) documentation.

## 1. Validate the inputs

The source folder should contain one subfolder per study. Folder IDs use `First author + year`; PDFs use underscores in place of spaces.

```bash
python get_robv2.py \
  --csv data/CD014722-data-rows.csv \
  --pdf-root data/pdfs \
  --output outputs/rob \
  --model gpt-5 \
  --dry-run
```

The 355 Cochrane data rows collapse to 246 unique study-topic assessments after equivalent subgroup and prevention-population rows are merged. The script verifies that merged rows have identical RoB annotations and stops if a conflict is found.

## 2. Generate model RoB 2 assessments

```bash
python get_robv2.py \
  --csv data/CD014722-data-rows.csv \
  --pdf-root data/pdfs \
  --output outputs/rob \
  --model gpt-5 \
  --reasoning-effort medium
```

Replace `gpt-5` with another available model ID, such as `o3-mini`, `gpt-4o`, `gpt-4o-mini`, or `gpt-3.5-turbo`. OpenAI o3-mini is configured as a reasoning model with a 200,000-token context window. Omit `--reasoning-effort` for models that do not use it.

For the OpenAI o3-mini condition used in the study:

```bash
python get_robv2.py \
  --csv data/CD014722-data-rows.csv \
  --pdf-root data/pdfs \
  --output outputs/rob \
  --model o3-mini \
  --reasoning-effort medium
```

For a single report:

```bash
python get_robv2.py \
  --pdf "data/pdfs/Bernardi 2020/Bernardi_2020.pdf" \
  --study "Bernardi 2020" \
  --topic "Anxiety symptoms at 0-1 months" \
  --output outputs/rob \
  --model gpt-4o
```

Outputs retain the study/topic/model hierarchy:

```text
outputs/rob/Anxiety0_1/gpt-5/Bernardi 2020/
├── Bernardi 2020.json
└── metadata.json
```

The assessment JSON is a list of 28 signaling-question, domain-conclusion, and overall-conclusion objects. Each object contains `question_id`, `question_text`, `domain`, `response`, `numerical`, `comment`, `reasoning`, `study`, and topic fields.

## 3. Extract human RoB 2 annotations

```bash
python extract_human_rob.py \
  --csv data/CD014722-data-rows.csv \
  --output data/human_review_structured
```

The repository includes the resulting 246 grouped JSON files in `data/human_review_structured`. The extractor pairs each support field with the judgement field having the same domain name. It converts HTML paragraphs and line breaks to text and splits signaling responses at IDs such as `1.1`, `1a.1`, `2.1a`, and `4.3b`. Cluster-specific and component IDs are preserved rather than silently collapsed into non-equivalent questions. `canonical_question_id` is populated only when a comparison relationship is unambiguous. Malformed or omitted response codes remain null with their original source text retained; the extractor does not guess a label.

## 4. Retrieve evidence and assign hallucination verdicts

```bash
python hallucination_qe.py \
  --responses outputs/rob \
  --pdf-root data/pdfs \
  --human-root data/human_review_structured \
  --output outputs/verdicts \
  --judge-model gpt-5 \
  --reasoning-effort medium \
  --top-k 5 \
  --chunk-size 1400 \
  --overlap 200
```

The script recursively discovers model response files, preserves their relative folder structure, and writes progress after every item so an interrupted run can resume. Each item stores all top-5 BM25 scores, snippets, chunk IDs, page ranges, the judge verdict, a minimal quotation, and whether that quotation occurs verbatim in the retrieved excerpts.

Verdict definitions:

- **Supported:** the report excerpts directly support the response and its material explanation.
- **Contradicted:** the excerpts directly conflict with the response or a material explanatory claim.
- **Not Found:** the claim is in scope, but the retrieved report evidence establishes neither support nor contradiction.
- **Out of Scope:** the item cannot be evaluated for that question, including genuinely inapplicable conditional items.

When `--human-root` is supplied, the matching human response is retained in each output item for accuracy analysis. The human label is never shown to the evidence judge; verdicts are based only on the generated claim and retrieved report excerpts. Ambiguous component-specific human responses are left blank rather than collapsed into a potentially incorrect gold label.

## 5. Generate reports

```bash
python hallucination_report.py \
  --input outputs/verdicts \
  --output outputs/reports
```

Generated files include:

- `items.csv`: one row per judged claim
- `overall.csv`: pooled evidence-support and hallucination metrics
- `by_study.csv`, `by_topic.csv`, `by_model.csv`, `by_topic_model.csv`, and `by_domain.csv`
- `bm25_by_verdict.csv`: retrieval-score summaries by verdict
- `summary.json`: machine-readable aggregate results

Metrics are defined as:

- Support rate = Supported / all judged items
- Conservative hallucination rate = (Contradicted + Not Found) / all judged items
- Strict hallucination rate = (Contradicted + Not Found + Out of Scope) / all judged items

## Testing

Tests do not call the OpenAI API.

```bash
python -m pip install -r requirements-dev.txt
pytest
```

Before a paid run, use `--dry-run`, `--limit`, `--limit-files`, or `--limit-items` to validate paths and a small subset.

## Data and security

- Do not commit API keys. `.env`, `api_key.txt`, and `apikey.txt` are ignored.
- Trial-report PDFs and the Cochrane review PDF are third-party materials and are ignored by Git. Obtain them from their original sources.
- The Cochrane data-row CSV is not ignored and may be added to the repository after the authors confirm that its redistribution is permitted.
- Generated responses and reports are ignored by default because they may be large and may contain OpenAI response identifiers.

## Reproducibility notes

- Pin model snapshots when exact reruns are required; model aliases can change over time.
- The metadata sidecar records the selected model, reasoning effort, PDF SHA-256 hash, page count, context limit, and whether PDF text was truncated.
- For short-context models, the script retains the beginning and end of the report and records truncation in metadata. This should be considered when comparing results across model families.
- PDF extraction quality affects retrieval. Scanned PDFs without usable OCR should be repaired or excluded before evaluation.
