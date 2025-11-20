#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import pandas as pd
import pdfplumber
from openai import OpenAI

# Optional: pip install tiktoken
import tiktoken

# ---------------------------
# Model context limits (tokens)
# ---------------------------
MODEL_CTX_LIMIT = {
    "gpt-3.5-turbo": 16385,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "o3-mini": 200000,      # adjust if your provider differs
    "o4-mini": 128000,
    "o3": 128000,
    "gpt-5": 400000,
}
DEFAULT_CTX_LIMIT = 16385  # safe default


def _encoding_for(model_name: str):
    """Best-effort tiktoken encoding for a given model."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        # universal fallback for GPT-3.5/4 families
        return tiktoken.get_encoding("cl100k_base")


def _truncate_to_budget(text: str, enc, budget_tokens: int) -> tuple[str, int]:
    """Truncate `text` to at most `budget_tokens` (by tokens). Returns (truncated_text, used_tokens)."""
    toks = enc.encode(text)
    if len(toks) <= budget_tokens:
        return text, len(toks)
    toks = toks[:max(0, budget_tokens)]
    return enc.decode(toks), min(len(toks), budget_tokens)


def get_rob_response(prompt, criteria, context, client, model_name="gpt-3.5-turbo",
                     output_reserve_tokens: int = 2000, verbose: bool = True):
    """
    Build messages, truncate `context` to fit the model context window, then call the API.
    Returns the text response.
    """
    # --------------------------
    # Prompts (unchanged content)
    # --------------------------
    robv2_instruct = """
    You are a literature reviewer. Your task is to review the given paper and retrieve required information listed as \
    questions with brief explanations provided by the user. Your reply should be formatted as a list responding to \
    the questions. \
    In the list of questions, each question is separated by an empty line: \
    The question id is the numeric number in front of the question, and is end with a ':' \
    The response should be follow the list of '– response should be one the following'\
    Then you need to give a numerical value of the response based on the following mappings: \
    { "Yes": 5, "Probably Yes": 4, "Probably": 3, "No": 2, "No information": 1, "Not Applicable": 0 } \
    { "High risk": 3, "Some concerns": 2, "Low risk": 1 } \
    You should response to each question both with a text response and a numerical value. \
    After that, an instruction asks you to provide a short comment to explain why you choose your response. \
    There are six question domains in this set of questions.\
    After answering all the questions to a domain, you should provide a conclusion of this domain \
    The domain conclusion question id is in the format of "Domain_X_Conclusion" \
    The after the conclusion, you should continue to respond to the next domain questions. \
    Your response should be formatted to JSON format. \
    Each response to each question should be formatted as dictionary with the following format: \
        {\
            "question_id": "question id", \
            "response": "response to the question", \
            "numerical": "numerical value of the response based on the mappings", \
            "comment": "explanation of your response", \
        } \
        """

    robv2_prompt = f"""
    Domain 1: Risk of bias arising from the randomization process \
    \n
    1.1: Was the allocation sequence random? \
        – response should be one the following: Yes / Probably Yes / Probably / No / No information \
        Instruction: Give comments on your answer. \
    \n
    1.2: Was the allocation sequence concealed until participants were enrolled and assigned to interventions? \
        – response should be one the following: Yes / Probably Yes / Probably / No / No information \
        Instruction: Give comments on your answer. \
    \n
    1.3: Did baseline differences between intervention groups suggest a problem with the randomization process? \
        – response should be one the following: Yes / Probably Yes / Probably / No / No information \
        Instruction: Give comments on your answer. \
    \n
    Domain 1 Conclusion: Risk-of-bias judgement, based on the responses for question 1.1, 1.2, 1.3, assess risk of bias arising from the randomization process \
        – response should be one the following: Low risk / High risk / Some concerns \
        Instruction: Give comments on your answer. \
    \n
    Domain 2: Risk of bias due to deviations from the intended interventions (effect of assignment to intervention) \
    \n
    2.1: Were participants aware of their assigned intervention during the trial? \
        – response should be one the following: Yes / Probably Yes / Probably / No / No information \
            Instruction: Give comments on your answer. \
    \n
    2.2: Were carers and people delivering the interventions aware of participants' assigned intervention during the trial? \
        – response should be one the following: Yes / Probably Yes / Probably / No / No information \
        Instruction: Give comments on your answer. \
    \n
    2.3: If one of “Yes" or  "Probably Yes" or "No information” is the answer to question 2.1 or to question 2.2, answer this question: Were there deviations from the intended intervention that arose because of the trial context? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably / No / No information \
        Instruction: Give comments on your answer. \
    \n
    2.4: If “Yes" or "Probably Yes” is the answer to question 2.3, answer this question: Were these deviations likely to have affected the outcome? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably / No / No information \
        Instruction: Give comments on your answer. \
    \n
    2.5: If “Yes" or "Probably Yes" or "No Information” is the answer to question 2.4, answer this question: Were these deviations from intended intervention balanced between groups? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably / No / No information \
        Instruction: Give comments on your answer. \
    \n
    2.6: Was an appropriate analysis used to estimate the effect of assignment to intervention? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    2.7: If “No" or "Probably No" or "No Information” is the answer to question 2.6, answer this question: Was there potential for a substantial impact (on the result) of the failure to analyse participants in the group to which they were randomized? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    Domain 2 Conclusion: Risk-of-bias judgement, based on the responses for question 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, assess risk of bias due to deviations from the intended interventions (effect of assignment to intervention) \
        – response should be one the following: Low risk / High risk / Some concerns \
        Instruction: Give comments on your answer.
    \n
    Domain 3: Missing outcome data \
    \n
    3.1: Were data for this outcome available for all, or nearly all, participants randomized? \
        - response should be one the following: Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    3.2: If “No" or "Probably No" or "No Information” is the answer to question 3.1, answer this question: Is there evidence that the result was not biased by missing outcome data? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably No / No \
        Instruction: Give comments on your answer. \
    \n
    3.3: If “No" or "Probably No" is the answer to question 3.2, answer this question: Could missingness in the outcome depend on its true value? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably No / No / No information\
        Instruction: Give comments on your answer. \
    \n
    3.4: If “Yes" or "Probably Yes" or "No information" is the answer to question 3.3, answer this question: Is it likely that missingness in the outcome depended on its true value? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    Domain 3 Conclusion: Risk-of-bias judgement, based on the responses for question 3.1, 3.2, 3.3, 3.4, assess risk of bias due to missing outcome data \
        – response should be one the following: Low risk / High risk / Some concerns \
        Instruction: Give comments on your answer. \
    \n
    Domain 4: Risk of bias in measurement of the outcome \
    \n
    4.1: Was the method of measuring the outcome inappropriate? \
        – response should be one the following: Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    4.2: Could measurement or ascertainment of the outcome have differed between intervention groups? \
        – response should be one the following: Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    4.3: If “No" or "Probably No", or "No information" is the answer to question 4.1 and 4.2, answer this question: Were outcome assessors aware of the intervention received by study participants? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    4.4: If “Yes" or "Probably Yes" or "No information" is the answer to question 4.3, answer this question: Could assessment of the outcome have been influenced by knowledge of intervention received? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    4.5: If “Yes" or "Probably Yes" or "No information" is the answer to question 4.4, answer this question: Is it likely that assessment of the outcome was influenced by knowledge of intervention received? \
        – response should be one the following: Not Applicable / Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    Domain 4 Conclusion: Risk-of-bias judgement, based on the responses for question 4.1, 4.2, 4.3, 4.4, 4.5, assess risk of bias in measurement of the outcome \
        – response should be one the following: Low risk / High risk / Some concerns \
        Instruction: Give comments on your answer. \
    \n
    Domain 5: Risk of bias in selection of the reported result \
    \n
    5.1: Were the data that produced this result analysed in accordance with a pre-specified analysis plan that was finalized before unblinded outcome data were available for analysis? \
        – response should be one the following: Yes / Probably Yes / Probably No / No / No information \
        Instruction: Give comments on your answer. \
    \n
    5.2: Is the numerical result being assessed likely to have been selected, on the basis of the results, from multiple eligible outcome measurements (e.g. scales, definitions, time points) within the outcome domain? \
        – response should be one the following: Yes / Probably Yes / Probably No / No / No Information \
        Instruction: Give comments on your answer. \
    \n
    5.3: Is the numerical result being assessed likely to have been selected, on the basis of the results, from multiple eligible analyses of the data? \
        – response should be one the following: Yes / Probably Yes / Probably No / No / No Information \
        Instruction: Give comments on your answer. \
    \n
    Domain 5 Conclusion: Risk-of-bias judgement, based on the responses for question 5.1, 5.2, 5.3, assess risk of bias in selection of the reported result \
        – response should be one the following: Low risk / High risk / Some concerns \
        Instruction: Give comments on your answer. \
    \n
    Overall risk of bias: based on the responses for all the domains, assess the overall risk of bias \
        – response should be one the following: Low risk / High risk / Some concerns \
        Instruction: Give comments on your answer. \
    """

    output_instruct = """
    You must return output as **valid JSON**. \
    Do not include any free text, markdown, explanation, or formatting outside the JSON block. \
    Make sure all property names are enclosed in double quotes and there are no trailing commas. \
    The response must be parsable by json.loads(). Strictly follow this rule. \
    Each response to each question should be formatted as dictionary with the following format, remember to response both text and numerical values: \
    { \
        "question_id": "question id", \
        "response": "response to the question", \
        "numerical": "numerical value of the response based on the mappings", \
        "comment": "explanation of your response", \
        "reasoning": "summary of the reasoning process for the response", \
    } \
    """

    # --------------------------
    # Token-budgeted truncation
    # --------------------------
    ctx_limit = MODEL_CTX_LIMIT.get(model_name, DEFAULT_CTX_LIMIT)
    enc = _encoding_for(model_name)

    fixed_text = "\n".join([
        robv2_instruct,
        f"the topic to be analyzed: {prompt}",
        # context will be inserted separately
        f"list of questions to response: {robv2_prompt}",
        f"Requirements for rigor in the evaluation criteria throughout the review: {criteria}",
        f"output instruction: {output_instruct}",
    ])
    fixed_tokens = len(enc.encode(fixed_text))

    # Budget left for the (large) PDF context
    budget_for_context = max(0, ctx_limit - output_reserve_tokens - fixed_tokens)
    truncated_context, used_ctx_tokens = _truncate_to_budget(context, enc, budget_for_context)

    if verbose:
        print(f"[Token budgeting] model={model_name} ctx_limit={ctx_limit} "
              f"fixed={fixed_tokens} context_used={used_ctx_tokens} "
              f"reserve={output_reserve_tokens} total≈{fixed_tokens+used_ctx_tokens} "
              f"(<= {ctx_limit}? {'YES' if fixed_tokens+used_ctx_tokens <= ctx_limit else 'NO'})")

    # --------------------------
    # Build messages with truncated context
    # --------------------------
    messages = [
        {'role': "system", 'content': robv2_instruct},
        {'role': "user", 'content': f"the topic to be analyzed: {prompt}"},
        {'role': "user", 'content': f"the paper to be analyzed: {truncated_context}"},
        {'role': "user", 'content': f"list of questions to response: {robv2_prompt}"},
        {'role': "user", 'content': f"Requirements for rigor in the evaluation criteria throughout the review: {criteria}"},
        {'role': "user", 'content': f"output instruction: {output_instruct}"},
    ]

    # --------------------------
    # API call
    # --------------------------
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    answer = response.choices[0].message.content
    return answer


def extract_json_from_text(text):
    """Extract first JSON object from text (simple brace matcher)."""
    start = text.find('{')
    if start == -1:
        raise ValueError("No opening brace '{' found in the response.")
    brace = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '{':
            brace += 1
        elif c == '}':
            brace -= 1
            if brace == 0:
                block = text[start:i+1]
                return json.loads(block)
    raise ValueError("No matching closing brace found for JSON object.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Use GPT to evaluate research bias with token-aware truncation.")
    parser.add_argument("--pdf", type=str, default=None, help="Path to the PDF file to analyze.")
    parser.add_argument("--api_key", type=str, required=True, help="Path to the text file containing the OpenAI API key.")
    parser.add_argument("--csv", type=str, default=None, help="CSV file with columns: Study,path,Analysis name")
    parser.add_argument("--topic", type=str, default=None, help="Review topic to be used for the analysis.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name (e.g., gpt-3.5-turbo, gpt-4o, gpt-4o-mini, o3-mini).")
    return parser.parse_args(input_args) if input_args is not None else parser.parse_args()


def main(args):
    if not args.pdf and not args.csv:
        raise ValueError("Either --pdf or --csv must be provided.")

    # Load API key and init client
    with open(args.api_key, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    client = OpenAI(api_key=api_key)

    os.makedirs(args.output, exist_ok=True)

    # Single PDF
    if args.pdf:
        if not args.topic:
            raise ValueError("When using --pdf, you must also provide --topic.")
        pdf_path = args.pdf
        study = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Processing PDF: {pdf_path}")

        with pdfplumber.open(pdf_path) as pdf:
            paper_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        response_text = get_rob_response(args.topic, _CRITERIA_TEXT, paper_text, client, model_name=args.model)
        # Save
        study_dir = os.path.join(args.output, study)
        os.makedirs(study_dir, exist_ok=True)
        out_json = os.path.join(study_dir, f"{study}.json")
        try:
            parsed = json.loads(response_text)
        except Exception:
            parsed = {"raw_text": response_text}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        print(f"Saved evaluation → {out_json}")

    # Batch CSV
    if args.csv:
        df = pd.read_csv(args.csv, encoding="latin1")
        if "Study" not in df.columns or "path" not in df.columns:
            raise ValueError("CSV must include columns: Study, path (and optionally Analysis name)")
        analysis_topic = df["Analysis name"].iloc[0] if "Analysis name" in df.columns and pd.notna(df["Analysis name"].iloc[0]) else args.topic
        if not analysis_topic:
            raise ValueError("Provide --topic or an 'Analysis name' column in the CSV.")

        for _, row in df.iterrows():
            study = str(row["Study"])
            pdf_path = str(row["path"])
            print(f"\nProcessing study: {study} ({pdf_path})")

            # If output exists, skip (you can remove this if you want re-runs)
            study_dir = os.path.join(args.output, study)
            out_json = os.path.join(study_dir, f"{study}.json")
            if os.path.exists(out_json):
                print(f"Output already exists → {out_json}. Skipping.")
                continue

            with pdfplumber.open(pdf_path) as pdf:
                paper_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

            response_text = get_rob_response(analysis_topic, _CRITERIA_TEXT, paper_text, client, model_name=args.model)
            os.makedirs(study_dir, exist_ok=True)
            try:
                parsed = json.loads(response_text)
            except Exception:
                parsed = {"raw_text": response_text}
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            print(f"Saved evaluation → {out_json}")


# Criteria text
_CRITERIA_TEXT = """
Adopt an extraordinarily exacting and comprehensive evaluation standard that scrutinizes every detail of the review process.
Base your assessment exclusively on the responses to the questions, and deliver a crystal-clear, concise summary of your findings.
"""


if __name__ == "__main__":
    args = parse_args()
    main(args)
