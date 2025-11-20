import argparse
import json
import os
from openai import OpenAI
import pandas as pd


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Use GPT to evaluate research bias.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        required=False,
        help="Path to the CSV file to analyze.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        required=True,
        help="Path to the text file of the API key for OpenAI to access the GPT model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        required=True,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="None",
        help="The path to the list of columns to include in the analysis.",
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


def csv_parse(csv_path, column_list):
    df = pd.read_csv(csv_path,encoding="latin1")
    study_name = column_list[0]
    domain_1_result = column_list[1]
    domain_1_comment = column_list[2]
    domain_2_result = column_list[3]
    domain_2_comment = column_list[4]
    domain_3_result = column_list[5]
    domain_3_comment = column_list[6]
    domain_4_result = column_list[7]
    domain_4_comment = column_list[8]
    domain_5_result = column_list[9]
    domain_5_comment = column_list[10]
    overall_result = column_list[11]
    overall_comment = column_list[12]

    result_dict = {}
    result_dict["Study"] = df[study_name].tolist()
    result_dict["Domain 1 Result"] = df[domain_1_result].tolist()
    result_dict["Domain 1 Comment"] = df[domain_1_comment].tolist()
    result_dict["Domain 2 Result"] = df[domain_2_result].tolist()
    result_dict["Domain 2 Comment"] = df[domain_2_comment].tolist()
    result_dict["Domain 3 Result"] = df[domain_3_result].tolist()
    result_dict["Domain 3 Comment"] = df[domain_3_comment].tolist()
    result_dict["Domain 4 Result"] = df[domain_4_result].tolist()
    result_dict["Domain 4 Comment"] = df[domain_4_comment].tolist()
    result_dict["Domain 5 Result"] = df[domain_5_result].tolist()
    result_dict["Domain 5 Comment"] = df[domain_5_comment].tolist()
    result_dict["Overall Result"] = df[overall_result].tolist()
    result_dict["Overall Comment"] = df[overall_comment].tolist()

    return result_dict


def get_rob_response(context_dict, client, model_name="o3-mini"):
    """
    -- context_dict: A dictionary extracting the human ROB responses from a single row.
    -- client: The OpenAI client to access the GPT model.
    -- model_name: The name of the GPT model to use (default is "o3-mini").
    """


    robv2_instruct = """
    You are a literature reviewer. Your task is to extract the response to each question from the Risk of Bias (ROB) assessment from the concise information \
    by human review. Your reply should be formatted as a list responding to the questions. \
    Your response should be formatted to JSON format. \
    Then you need to give a numerical value of the response based on the following mappings: \
    { "Yes": 5, "Probably Yes": 4, "Probably": 3, "No": 2, "No information": 1, "Not Applicable": 0 } \
    { "High risk": 3, "Some concerns": 2, "Low risk": 1 } \
    You should response to each question both with a text response and a numerical value. \
    Each response to each question should be formatted as dictionary with the following format: \
        {\
            "question_id": "question id", \
            "response": "response to the question", \
            "numerical": "numerical value of the response based on the mappings", \
            "comment": "explanation of your response", \
        } \
        """
    
    robv2_prompt_domain_1 = f"""
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
    """

    robv2_prompt_domain_2 = f"""
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
    """

    robv2_prompt_domain_3 = f"""
    Domain 3: Missing outcome data
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
    \n"""

    robv2_prompt_domain_4 = f"""
    Domain 4: Risk of bias in measurement of the outcome
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
        – response should be one the following: NA / Yes / Probably Yes / Probably No / No / No information \
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
    """
    
    robv2_prompt_domain_5 = f"""
    Domain 5: Risk of bias in selection of the reported result
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
    """

    robv2_prompt_overall = f"""
    Overall risk of bias: based on the responses for all the domains, assess the overall risk of bias \
        – response should be one the following: Low risk / High risk / Some concerns \
        Instruction: Give comments on your answer. \
    """

    output_instruct = """
    Your response should be formatted to the JSON format. \
    Each response to each question should be formatted as dictionary with the following format, remember to response both text and numerical values: \
    { \
        "question_id": "The question id from the ROB question", \
        "response": "the response extracted from the human response", \
        "numerical": "numerical value of the response based on the mappings", \
        "comment": "the explanation extracted from the human response", \
    } \
    """

    topic = context_dict["Study"]
    domain_1 = context_dict["Domain 1 Result"]
    domain_1_comment = context_dict["Domain 1 Comment"]
    domain_2 = context_dict["Domain 2 Result"]
    domain_2_comment = context_dict["Domain 2 Comment"]
    domain_3 = context_dict["Domain 3 Result"]
    domain_3_comment = context_dict["Domain 3 Comment"]
    domain_4 = context_dict["Domain 4 Result"]
    domain_4_comment = context_dict["Domain 4 Comment"]
    domain_5 = context_dict["Domain 5 Result"]
    domain_5_comment = context_dict["Domain 5 Comment"]
    overall_result = context_dict["Overall Result"]
    overall_comment = context_dict["Overall Comment"]
    messages = []
    messages.append({'role': "system", 'content': robv2_instruct})
    messages.append({'role': "user", 'content': f"the topic to be analyzed: {topic}"})
    messages.append({'role': "user", 'content': f"Domain 1 questions: {robv2_prompt_domain_1}"})
    messages.append({'role': "user", 'content': f"human response to Domain 1 questions: {domain_1}, {domain_1_comment}"})
    messages.append({'role': "user", 'content': f"Domain 2 questions: {robv2_prompt_domain_2}"})
    messages.append({'role': "user", 'content': f"human response to Domain 2 questions: {domain_2}, {domain_2_comment}"})
    messages.append({'role': "user", 'content': f"Domain 3 questions: {robv2_prompt_domain_3}"})
    messages.append({'role': "user", 'content': f"human response to Domain 3 questions: {domain_3}, {domain_3_comment}"})
    messages.append({'role': "user", 'content': f"Domain 4 questions: {robv2_prompt_domain_4}"})
    messages.append({'role': "user", 'content': f"human response to Domain 4 questions: {domain_4}, {domain_4_comment}"})
    messages.append({'role': "user", 'content': f"Domain 5 questions: {robv2_prompt_domain_5}"})
    messages.append({'role': "user", 'content': f"human response to Domain 5 questions: {domain_5}, {domain_5_comment}"})
    messages.append({'role': "user", 'content': f"Overall questions: {robv2_prompt_overall}"})
    messages.append({'role': "user", 'content': f"human response to Overall questions: {overall_result}, {overall_comment}"})
    messages.append({'role': "user", 'content': f"The response should be rigorously based on the human response, and should not be made up."})
    messages.append({'role': "user", 'content': f"output instruction: {output_instruct}"})
    
    response = client.chat.completions.create(
        model="o3-mini",  # gpt-4o, o4-mini, o3
        messages=messages,
    )
    answer = response.choices[0].message.content

    return answer


def main(args):
    if args.api_key is None:
        raise ValueError("API key is required. Please provide the path to the API key file.")

    if args.output is None:
        raise ValueError("Output folder is required. Please provide the path to the output folder.")
    
    if args.csv is None:
        raise ValueError("CSV file path is required. Please provide the path to the input CSV file.")
    
    with open(args.column, 'r') as file:
        column_list = [line.strip() for line in file.readlines()]

    csv_path = args.csv
    # Load API key
    with open(args.api_key, "r") as f:
        key = f.read().strip()
    
    try:
        client = OpenAI(api_key=key)
    except Exception as e:
        print(f"Error occurred: {e}")
        return

    # Parse the CSV file
    result_dict = csv_parse(csv_path, column_list)
    column_names = list(result_dict.keys())
    save_folder = args.output
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # get the total number of studies in the result_dict
    study_num = len(result_dict["Study"])
    
    for idx in range(study_num):
        context_dict = {}

        # get the context for each study
        for column_name in column_names:
            context_dict[column_name] = result_dict[column_name][idx]
        
        try:
            response_text = get_rob_response(context_dict, client, model_name=args.model)
            parsed_response = json.loads(response_text)
        except Exception as e:
            print(f"Error processing study {context_dict['Study']}: {e}")
            continue
        
        # Cochrane style naming
        research_name = context_dict["Study"]
        os.makedirs(os.path.join(save_folder, research_name), exist_ok=True)
        save_file = os.path.join(save_folder, research_name, research_name + ".json")

        with open(save_file, 'w') as f:
            json.dump(parsed_response, f, indent=4)
        print(f"Saved parsed human ROB response for {research_name} to {save_file}")

    

    # research_name = os.path.basename(os.path.dirname(pdf_path))
    #     os.makedirs(os.path.join(save_folder, research_name), exist_ok=True)
    #     save_file = os.path.join(save_folder, research_name, research_name + ".json")


if __name__ == "__main__":
    args = parse_args()
    main(args)