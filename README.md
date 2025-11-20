<div align="left">

# pipeline for RoB assess and hallucination evaluation  
To perform risk-of-bias (RoB) assessment on a trial report (PDF format). We need to organize the files below. <p>
File structure: <p>
--root folder <p>
----claims <p>
----human <p>
----pdf folder  (e.g., pdfs) <p>
-------study_id  (e.g., Liang 2025) <p>
--------study_id.pdf  (e.g., Liang_2025.pdf) <p>
...... <p>
----example.csv <p>
Note: <p>
1 - non English characters must be converted to English characters, e.g. ç -> c, ö -> o, all non-letter in author name must be changed, e.g. O'Callaghan -> Callaghan <p>
2 - the author+year notation will be used to name file folders and GPT processing, non-letters and non-English characters are likely to cause runtime error. <p>
<p>
The RoB revieweds by human reviewers are saved in the format as the file example.csv. <p>
Add the full path of the PDF full papers to a new column named 'path' in example.csv after running the following script. A script named "link_path.py" is provided for this purpose.<p>
python link_path.py --csv="/your_local_full_path/example.csv" --path="/full_path_pdfs"<p>
<p>
Run get_rob.py or get_robv2.py to get the RoB response for each research from the list in example.csv<p>
Since OpenAI frequently changes the policy for max token limitation, get_robv2.py adds the function to truncate the excessive tokens read from the PDF files, given the current policy.<p>
python get_rob.py --api_key="api_key.txt" --csv="/full_path/example.csv" --output="/full_path/outputs" --model="gpt-5"<p>
or<p>
python get_robv2.py --api_key="api_key.txt" --csv="/full_path/example.csv" --output="/full_path/outputs" --model="gpt-5"<p>
<p>
--api_key: the text file containing the OpenAI api_key <p>
--csv: the table with study name and path to the PDF file <p>
--output: the path to the folder for the response, each response will be saved to a sub-folder named by the study name, in a JSON file all named by the study name. <p>
--model: GPT models: gpt-5, o3-mini, gpt-3.5-turbo, gpt-4o, etc. <p>
<p>
Parse the original human response and convert it into structured JSON files for each research listed in the example.csv file.<p>
python convert_json.py --csv="/full_path/example.csv" --api_key="/full_path/api_key.txt" --output="/full_path/human" --column="/full_path/column.txt" --model="gpt-5"<p>
<p>
--api_key: the text file containing the OpenAI api_key<p>
--csv: the table with human RoB response and full path to the PDF file<p>
--output: the path to the folder for the response, each response will be saved to a sub-folder named by the study name, in a JSON file all named by the study name.<p>
--column: a text file with the list of columns in the example.csv file to be analyzed, one column name per line.<p>
--model: GPT models: gpt-5, o3-mini, gpt-3.5-turbo, gpt-4o, etc.<p>
<p>
Compute the hallicinations data with BM25<p>
python hallucination_qe.py --json="/Full_path_to_JSON_review_by_topic/group_by_LLM" --pdf_root="/full_path_to_PDF_folder/" --api_key="/Full_path/api_key.txt" --model="gpt-5" --out_json="/full_path/claims"<p>
<p>
--json: folder to the JSON response by LLM, grouped by model names, e.g. gpt-5, o3-mini, gpt-3.5<p>
--pdf_root: folder for the PDF files.<p>
--api_key: the text file containing the OpenAI api_key<p>
--model: GPT models: gpt-5, o3-mini, gpt-3.5-turbo, gpt-4o, etc.<p>
--out_json: fold to save the JSON files of hallucination evaluation, one file per study<p>
<p>
Interpret the hallucinations data<p>
python hallucination_report.py --input_dir="/full_path/claims" --save_csv="/Users/full_path/claims"<p>
--input_dir: the folder contains all JSON files of the hallucination assessment<p>
--save_csv: the folder to save the hallucination analytical results<p>
<p>
</div>
