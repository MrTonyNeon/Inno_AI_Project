import argparse
import csv
import json
import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_SUBMISSIONS_DIR = "anonymized_papers"
DEFAULT_GRADES_TEMPLATE = "guideline_files/Grades.csv"
DEFAULT_OUTPUT_DIR = "grades_results"
DEFAULT_MAX_FILES = 5

parser = argparse.ArgumentParser(description="Evaluate student submissions using AI models")
parser.add_argument("--models", nargs="+", help="Models to use for evaluation (format: [provider:]model_name)")
parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                    help="Maximum number of new files to evaluate (already evaluated ones are skipped)")
args = parser.parse_args()

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
AITUNNEL_API_KEY = os.getenv("AITUNNEL_API_KEY")

print(f"OPENAI_API_KEY: {'✅ Found' if OPENAI_API_KEY else '❌ Not found'}")
print(f"OPENROUTER_API_KEY: {'✅ Found' if OPENROUTER_API_KEY else '❌ Not found'}")
print(f"AITUNNEL_API_KEY: {'✅ Found' if AITUNNEL_API_KEY else '❌ Not found'}")

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    DEFAULT_MODELS = [{"name": "gpt-4o-mini", "provider": "openai"}]
elif OPENROUTER_API_KEY:
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    DEFAULT_MODELS = [{"name": "deepseek/deepseek-chat-v3-0324:free", "provider": "openrouter"}]
elif AITUNNEL_API_KEY:
    client = OpenAI(api_key=AITUNNEL_API_KEY, base_url="https://api.aitunnel.ru/v1")
    DEFAULT_MODELS = [{"name": "gpt-4.1-nano", "provider": "aitunnel"}]
else:
    raise ValueError("No valid API key found.")


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
    You are an academic evaluator. Evaluate the submission according to the criteria.

    STUDENT SUBMISSION:
    {submission_text}

    GRADING CRITERIA:
    {json.dumps(grading_criteria, indent=2)}

    Your task is to evaluate this submission according to the provided criteria. For each criterion:
    - Use the maximum score indicated in brackets as the upper bound (e.g., "Structure (10)" means a maximum score of 10).
    - Provide a score between 0 and the maximum value clearly stated in brackets.
    - Criterion names must match the grading criteria exactly as given, including any brackets.
    - Do not provide any explanation, just the JSON structure requested.

    Please return your evaluation strictly in the following JSON format:
    {{
      "criteria": [
        {{
          "criterion": "Criterion name",
          "score": numeric_score
        }},
        ...
      ],
      "overall_feedback": "Brief overall assessment of the submission's strengths and weaknesses (1-2 sentences)"
    }}
    """
    return prompt


def validate_evaluation_structure(evaluation, grading_criteria):
    if not isinstance(evaluation.get('criteria'), list):
        return False

    for criterion in evaluation.get('criteria', []):
        if not (isinstance(criterion, dict) and
                'criterion' in criterion and
                isinstance(criterion['criterion'], str) and
                'score' in criterion):
            return False

        criterion_name = criterion['criterion']
        score = criterion['score']

        max_points = None
        for template_criterion in grading_criteria:
            match = re.search(r'\((\d+)\)$', template_criterion)
            if match and (template_criterion.lower() in criterion_name.lower() or
                          criterion_name.lower() in template_criterion.lower()):
                max_points = int(match.group(1))
                break

        if max_points is not None:
            if not (isinstance(score, (int, float)) and 0 <= score <= max_points):
                return False

    if not isinstance(evaluation.get('overall_feedback'), str):
        return False

    return True

def evaluate_with_llm(client, model_name, prompt, grading_criteria):
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
            evaluation = json.loads(response_content)

            if validate_evaluation_structure(evaluation, grading_criteria):
                return evaluation, response_content
            else:
                print(f"Invalid JSON structure returned by {model_name}, attempt {retry_count + 1}")
                retry_count += 1
                time.sleep(2 ** retry_count)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error with {model_name}, attempt {retry_count + 1}: {e}")
            retry_count += 1
            time.sleep(2 ** retry_count)

    print(f"Failed to get valid response from {model_name} after {max_retries} attempts")
    return None, None


def save_llm_answer(answer_text, student_file, answers_dir):
    team_match = re.match(r'(Team \d+)', student_file)
    team_name = team_match.group(1) if team_match else os.path.splitext(student_file)[0]

    filename = f"{team_name}.json"
    filepath = os.path.join(answers_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(answer_text)
    return filepath


def calculate_max_points(grading_template):
    total_max_points = 0
    for criterion in grading_template:
        match = re.search(r'\((\d+)\)$', criterion)
        if match:
            total_max_points += int(match.group(1))
    return total_max_points

def calculate_total_score(row, total_max_points):
    total_points = 0
    for key, value in row.items():
        if value not in ["", "ERROR"] and key != 'Team':
            try:
                total_points += float(value)
            except (ValueError, TypeError):
                pass

    if total_max_points > 0:
        return f"{(total_points / total_max_points) * 100:.2f}%"
    else:
        return "0%"


def parse_evaluation_to_row(evaluation_result, grading_template, student_file, total_max_points):
    def extract_team_name(filename):
        match = re.match(r'(Team \d+)', filename)
        if match:
            return match.group(1)
        else:
            return filename

    if evaluation_result is None:
        row = {'Team': extract_team_name(student_file)}
        for criterion in grading_template:
            row[f"{criterion}"] = "ERROR"
        row['Total'] = "ERROR"
        row['Feedback'] = "Evaluation failed"
        return row

    row = {'Team': extract_team_name(student_file)}
    for criterion_result in evaluation_result.get('criteria', []):
        criterion_name = criterion_result.get('criterion', '')
        exact_matches = [c for c in grading_template if c.lower() == criterion_name.lower()]
        if exact_matches:
            criterion = exact_matches[0]
        else:
            matching_criteria = [c for c in grading_template if
                                 (c.lower() in criterion_name.lower() and len(c) > 5) or
                                 (criterion_name.lower() in c.lower() and len(criterion_name) > 5)]
            if matching_criteria:
                criterion = matching_criteria[0]
            else:
                continue
        row[f"{criterion}"] = criterion_result.get('score', '')

    row['Total'] = calculate_total_score(row, total_max_points)
    row['Feedback'] = evaluation_result.get('overall_feedback', '')

    for criterion in grading_template:
        if criterion not in row:
            row[criterion] = ""

    return row


def main():
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    answers_dir = os.path.join(DEFAULT_OUTPUT_DIR, "received_jsons")
    os.makedirs(answers_dir, exist_ok=True)

    grades_df = load_grading_template(DEFAULT_GRADES_TEMPLATE)
    grading_criteria = [col for col in grades_df.columns if col != 'Team']
    total_max_points = calculate_max_points(grading_criteria)

    submission_files = get_submission_files(DEFAULT_SUBMISSIONS_DIR)
    submission_files.sort(key=extract_team_number)

    if args.models:
        models = []
        for model_spec in args.models:
            parts = model_spec.split(":", 1)
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

        model_answers_dir = os.path.join(answers_dir, model_name.replace('-', '_').replace('.', '_')).replace(':', '_')
        os.makedirs(model_answers_dir, exist_ok=True)

        sanitized_model_name = model_name.replace('-', '_').replace('/', '_').replace(':', '_')
        output_file = os.path.join(DEFAULT_OUTPUT_DIR, f"Grades_{sanitized_model_name}.csv")
        columns = ['Team'] + grading_criteria + ['Total'] + ['Feedback']
        results = []
        evaluated_count = 0

        already_evaluated = {}
        for json_file in os.listdir(model_answers_dir):
            if json_file.endswith('.json'):
                team_match = re.match(r'Team (\d+)', json_file)
                if team_match:
                    team_number = team_match.group(1)
                    already_evaluated[team_number] = os.path.join(model_answers_dir, json_file)

        print(f"Found already evaluated submissions for teams: {list(already_evaluated.keys())}")

        for student_file in submission_files:
            print(f"  Processing: {student_file}")
            evaluation_result, raw_response = None, None

            team_match = re.search(r'Team (\d+)', student_file)
            if team_match:
                team_number = team_match.group(1)
            else:
                print(f"    Could not extract team number from {student_file}, skipping")
                continue

            if team_number in already_evaluated:
                print(f"    Already evaluated team {team_number}, loading from {already_evaluated[team_number]}")
                try:
                    with open(already_evaluated[team_number], 'r', encoding='utf-8') as f:
                        raw_response = f.read()
                        evaluation_result = json.loads(raw_response)
                    if not validate_evaluation_structure(evaluation_result, grading_criteria):
                        print(f"    Cached evaluation has invalid structure, re-evaluating")
                        evaluation_result = None
                except (json.JSONDecodeError, Exception) as e:
                    print(f"    Error loading cached evaluation: {e}, re-evaluating")
                    evaluation_result, raw_response = None, None
            else:
                print(f"    Team {team_number} not yet evaluated.")

                if evaluated_count >= args.max_files:
                    print(f"    Max new evaluations reached ({args.max_files}), skipping...")
                    continue

                submission_path = os.path.join(DEFAULT_SUBMISSIONS_DIR, student_file)
                submission_text = read_submission(submission_path)
                prompt = create_evaluation_prompt(grading_criteria, submission_text)

                try:
                    if provider in ["aitunnel", "openai", "openrouter"]:
                        evaluation_result, raw_response = evaluate_with_llm(client, model_name, prompt, grading_criteria)
                        if raw_response:
                            save_path = save_llm_answer(raw_response, student_file, model_answers_dir)
                            print(f"    Saved response to {save_path}")
                            evaluated_count += 1
                    else:
                        print(f"Provider '{provider}' not supported yet")
                        evaluation_result, raw_response = None, None
                except Exception as e:
                    print(f"Error evaluating {student_file} with {model_name}: {e}")
                    evaluation_result, raw_response = None, None

            row = parse_evaluation_to_row(evaluation_result, grading_criteria, student_file, total_max_points)
            results.append(row)

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved results to {output_file}")


if __name__ == "__main__":
    main()