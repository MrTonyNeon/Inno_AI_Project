import argparse
import csv
import json
import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (for API keys)
load_dotenv()

# Set up command line arguments
parser = argparse.ArgumentParser(description="Evaluate student submissions using AI models")
parser.add_argument("--submissions-dir", default="anonymized_papers",
                    help="Directory containing student submission files")
parser.add_argument("--grades-template", default="guideline_files/Grades.csv",
                    help="Path to grading template CSV")
parser.add_argument("--output-dir", default="grades_results",
                    help="Directory to save evaluation results")
parser.add_argument("--models", nargs="+",
                    help="Models to use for evaluation (format: [provider:]model_name)")
args = parser.parse_args()


client = OpenAI(
    api_key=os.getenv("AITUNNEL_API_KEY"),
    base_url="https://api.aitunnel.ru/v1"
)

# Define the models to evaluate
DEFAULT_MODELS = [
    {"name": "gpt-4.1-nano", "provider": "aitunnel"},
]


def load_grading_template(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading grading template: {e}")
        raise


def get_submission_files(submissions_dir):
    return [f for f in os.listdir(submissions_dir) if f.endswith('.txt')]


def extract_team_number(filename):
    match = re.match(r'Team (\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')


def read_submission(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_evaluation_prompt(grading_criteria, submission_text):
    prompt = f"""
        This is an EDUCATIONAL CONTEXT. You are an academic evaluator assessing student project submissions for a university course.
        TASK DESCRIPTION: You are to evaluate academic student work. This is for EDUCATIONAL SCORING PURPOSES only in an academic setting.

        STUDENT SUBMISSION:
        {submission_text}
        
        GRADING CRITERIA:
        {json.dumps(grading_criteria, indent=2)}

        Your task is to evaluate this submission according to the provided criteria. For each criterion:
        - Provide a score within the specified range, no explanation is needed.

        Format your response as a JSON object with the following structure:
        {{
          "criteria": [
            {{
              "criterion": "Criterion name",
              "score": numeric_score
            }},
            ...
          ],
          "overall_score": "Sum of all assessed points divided by the maximum possible point, in percentages",
          "overall_feedback": "Brief overall assessment of the submission's strengths and weaknesses (1-3 sentences)"
        }}
        """
    return prompt


def evaluate_with_llm(client, model_name, prompt):
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert academic evaluator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            response_content = response.choices[0].message.content
            return json.loads(response_content), response_content
        except (json.JSONDecodeError, Exception) as e:
            retry_count += 1
            print(f"Error with {model_name}, attempt {retry_count}: {e}")
            time.sleep(2 ** retry_count)

    print(f"Failed to get valid response from {model_name} after {max_retries} attempts")
    return None, None


def save_llm_answer(answer_text, model_name, student_file, answers_dir):
    # Create a safe filename based on student file and model name
    safe_model_name = model_name.replace('-', '_').replace('.', '_')
    filename = f"{os.path.splitext(student_file)[0]}_{safe_model_name}.json"
    filepath = os.path.join(answers_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(answer_text)

    return filepath


def parse_evaluation_to_row(evaluation_result, grading_template, student_file):
    def extract_team_name(filename):
        match = re.match(r'(Team \d+)', filename)
        if match:
            return match.group(1)
        else:
            return filename

    if evaluation_result is None:
        row = {'Team': student_file}
        for criterion in grading_template:
            row[f"{criterion}"] = "ERROR"
        row['Total'] = "ERROR"
        row['Feedback'] = "Evaluation failed"
        return row

    row = {'Team': extract_team_name(student_file)}
    for criterion_result in evaluation_result.get('criteria', []):
        criterion_name = criterion_result.get('criterion', '')
        matching_criteria = [c for c in grading_template if c.lower() in criterion_name.lower()
                             or criterion_name.lower() in c.lower()]
        if matching_criteria:
            criterion = matching_criteria[0]
            row[f"{criterion}"] = criterion_result.get('score', '')

    row['Total'] = evaluation_result.get('overall_score', '')
    row['Feedback'] = evaluation_result.get('overall_feedback', '')

    for criterion in grading_template:
        if criterion not in row:
            row[criterion] = ""

    return row


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    answers_dir = os.path.join(args.output_dir, "received_jsons")
    os.makedirs(answers_dir, exist_ok=True)

    grades_df = load_grading_template(args.grades_template)
    grading_criteria = [col for col in grades_df.columns if col != 'Team']
    submission_files = get_submission_files(args.submissions_dir)
    submission_files.sort(key=extract_team_number)

    if args.models:
        models = []
        for model_spec in args.models:
            parts = model_spec.split(":")
            if len(parts) == 2:
                models.append({"name": parts[1], "provider": parts[0]})
            else:
                models.append({"name": parts[0], "provider": "aitunnel"})
    else:
        models = DEFAULT_MODELS

    for model in models:
        model_name = model["name"]
        provider = model["provider"]
        print(f"Evaluating submissions with {provider}:{model_name}...")

        # Create model-specific directory for answers
        model_answers_dir = os.path.join(answers_dir, model_name.replace('-', '_').replace('.', '_'))
        os.makedirs(model_answers_dir, exist_ok=True)

        output_file = os.path.join(args.output_dir, f"Grades_{model_name.replace('-', '_')}.csv")
        columns = ['Team'] + grading_criteria + ['Total'] + ['Feedback']
        results = []

        for student_file in submission_files:
            print(f"  Processing: {student_file}")
            submission_path = os.path.join(args.submissions_dir, student_file)
            submission_text = read_submission(submission_path)

            prompt = create_evaluation_prompt(grading_criteria, submission_text)

            try:
                if provider in ["aitunnel", "openai"]:
                    evaluation_result, raw_response = evaluate_with_llm(client, model_name, prompt)

                    if raw_response:
                        save_path = save_llm_answer(raw_response, model_name, student_file, model_answers_dir)
                        print(f"    Saved response to {save_path}")
                else:
                    print(f"Provider '{provider}' not supported yet")
                    evaluation_result, raw_response = None, None
            except Exception as e:
                print(f"Error evaluating {student_file} with {model_name}: {e}")
                evaluation_result, raw_response = None, None

            row = parse_evaluation_to_row(evaluation_result, grading_criteria, student_file)
            results.append(row)

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()