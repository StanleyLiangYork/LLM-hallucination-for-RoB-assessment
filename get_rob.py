import argparse
import pandas as pd
import pdffile
import pdfplumber
import re
import tiktoken


def get_rob_response(prompt, criteria, context, client, model_name="o3-mini"):
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

    messages = []
    messages.append({'role': "system", 'content': robv2_instruct})
    messages.append({'role': "user", 'content': f"the topic to be analyzed: {prompt}"})
    messages.append({'role': "user", 'content': f"the paper to be analyzed: {context}"})
    messages.append({'role': "user", 'content': f"list of questions to response: {robv2_prompt}"})
    messages.append({'role': "user", 'content': f"Requirements for rigor in the evaluation criteria throughout the review: {criteria}"})
    messages.append({'role': "user", 'content': f"output instruction: {output_instruct}"})
    
    response = client.chat.completions.create(
        model="o3-mini",  # gpt-4o, o4-mini, o3
        messages=messages,
    )
    answer = response.choices[0].message.content

    return answer


# --- helper: truncate by tokens (uses tiktoken if available) ---
def truncate_to_tokens(text: str, max_tokens: int = 15_000, model_name: str = "gpt-4o-mini"):
    """
    Returns (truncated_text, token_count_used).
    If tiktoken isn't available or the model isn't known, falls back to a ~4 chars/token heuristic.
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")  # solid default for GPT-3.5/4/4o families
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text, len(toks)
        return enc.decode(toks[:max_tokens]), max_tokens
    except Exception:
        approx_chars_per_token = 4  # conservative heuristic
        max_chars = max_tokens * approx_chars_per_token
        truncated = text if len(text) <= max_chars else text[:max_chars]
        est_tokens = min(max_tokens, max(1, len(truncated) // approx_chars_per_token))
        return truncated, est_tokens



def extract_json_from_text(text):
    """
    Extract the first valid JSON block from the text using a simple bracket matching strategy.
    """
    start = text.find('{')
    if start == -1:
        raise ValueError("No opening brace '{' found in the response.")

    # Scan forward and count braces to find the matching closing brace
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                try:
                    json_block = text[start:i + 1]
                    return json.loads(json_block)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse JSON block: {e}")

    raise ValueError("No matching closing brace found for JSON object.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Use GPT to evaluate research bias.")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        required=False,
        help="Path to the PDF file to analyze.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        required=True,
        help="Path to the text file of the API key for OpenAI to access the GPT model.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="read the csv file containing paths of the PDF for review.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="The review topic to be used for the analysis.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        required=True,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="None",
        help="The name of the GPT model: o3-mini, o4-mini, o3, gpt-4o.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    batch_mode = False
    if args.api_key is None:
        raise ValueError("API key is required. Please provide the path to the API key file.")

    if args.output is None:
        raise ValueError("Output folder path is required. Please provide the path to the output folder.")
    
    # if args.topic is None:
    #     raise ValueError("Topic is required. Please provide the topic for the analysis.")
    # else:
    #     topic = args.topic

    if args.pdf is None and args.csv is None:
        raise ValueError("Either --pdf or --csv must be provided.")
    elif args.pdf:
        pdf_path = args.pdf
        context = pdffile.read_pdf(pdf_path)
        batch_mode = False
    elif args.csv:
        csv_path = args.csv
        batch_mode = True

    # Load API key
    with open(args.api_key, "r") as f:
        key = f.read().strip()

    # Initialize OpenAI client
    client = OpenAI(api_key=key)

    # Process PDF or CSV
    if args.pdf and batch_mode==False:
        if args.topic:
            topic = args.topic
        else:
            raise ValueError("Topic is required. Please provide the topic for the analysis.")
        pdf_path = args.pdf
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        with pdfplumber.open(pdf_path) as pdf:
            paper_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        # paper_text = pdffile.read_pdf(pdf_path)
    elif args.csv and batch_mode==True:
        csv_path = args.csv
    else:
        raise ValueError("Either --pdf or --csv must be provided.")
    
    if args.model:
        model_name = args.model
    else:
        model_name = "o3-mini"

    criteria = """
    Adopt an extraordinarily exacting and comprehensive evaluation standard that scrutinizes every detail of the review process. \
    Base your assessment exclusively on the responses to the questions, and deliver a crystal-clear, concise summary of your findings.
    """
    save_folder = args.output
    if not batch_mode:
        response_text = get_rob_response(topic, criteria, context, client, model_name=model_name)
        # print(f"Response: {response_text}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        research_name = os.path.basename(os.path.dirname(pdf_path))
        os.makedirs(os.path.join(save_folder, research_name), exist_ok=True)
        save_file = os.path.join(save_folder, research_name, research_name + ".json")
        parsed_response = json.loads(response_text)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(parsed_response, f, indent=2)

        print(f"Saved evaluation of {research_name}")

    if batch_mode:
        data_df = pd.read_csv(csv_path, encoding="latin1")
        analysis_topic = data_df["Analysis name"].iloc[0]
        for idx, row in data_df.iterrows():
            study_id = row["Study"]
            print(f"Processing study: {study_id}")
            pdf_path = row["path"]
            print(f"processing {pdf_path}")
            research_name = row["Study"]

            save_path = os.path.join(save_folder, research_name)
            if os.path.exists(save_path):
                print(f"Response has already been saved for {research_name}. Skipping.")
                continue
            with pdfplumber.open(pdf_path) as pdf:
                context = "\n".join(page.extract_text() or "" for page in pdf.pages)
            # context = pdffile.read_pdf(pdf_path)
            context, token_count = truncate_to_tokens(context, max_tokens=13_000, model_name=model_name)
            print(f"paper_text truncated to <15k tokens for {model_name}; tokens≈{token_count}")
            # response_text = get_rob_response(analysis_topic, criteria, context, client, model_name=model_name)
            save_folder = args.output
            os.makedirs(os.path.join(save_folder, research_name), exist_ok=True)
            save_file = os.path.join(save_folder, research_name, research_name + ".json")
            parsed_response = json.loads(response_text)
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(parsed_response, f, indent=2)
            print(f"Saved evaluation of {research_name}")

        

if __name__ == "__main__":
    args = parse_args()
    main(args)