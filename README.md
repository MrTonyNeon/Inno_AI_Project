# Student Papers AI Evaluation Project

A tool for automatic anonymization of student PDF papers, their subsequent evaluation using AI models, and comparison of model performance.

## Features

- Processing PDF documents from structured team folders
- Name anonymization in texts using Language Processing
- Saving anonymized versions in text format
- Work evaluation using various AI models
- Generation of evaluation reports
- Comparison of multiple AI model results against reference grades
- Generation of heatmaps and accuracy summaries for model benchmarking

## Requirements

- Python 3.8+
- PyMuPDF
- Other dependencies from `requirements.txt`

## Installation

1. Clone the repository
```bash
git clone https://github.com/MrTonyNeon/Inno_AI_Project.git
```
2. Navigate to the project directory
```bash
cd Inno_AI_Project
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Step 1: Anonymize student submissions
Place student PDF papers in corresponding team folders within the student_papers/ directory, then 
run the anonymization script *(you can specify the number of files to process with `--max_files`)*
```bash
python .\anonymize_data.py --max_files 1 --mode online_ai --model gpt-4o
```
Modes:
- `regex`: Simple pattern matching
- `local_ai`: Use a local AI model
- `online_ai`: Use OpenAI or other online providers (select model with `--model`)

### Step 2: Evaluate anonymized submissions
Run the evaluation script:
- `--models`: specify the AI models to use (e.g., `gpt-4.1-nano`, `gpt-4o`, etc.)
- `--max_files`: limit the number of files to evaluate for every model (default is 10)
```bash
python evaluate_ai.py --models gpt-4.1-nano gpt-4o --max-files 5
```

### Step 3: Compare models accuracy and generate visualizations
Run the comparison script:
```bash
python compare_ai.py
```
This will:
- Compare model outputs to a reference grading CSV
- Calculate per-criterion similarity scores
- Output CSV reports and heatmaps to `output_results/`
- Generate a ranking of models based on evaluation accuracy

## Project Structure
- anonymized_papers/ - anonymized versions
- grades_results/ - work evaluation results
- guideline_files/ - guidelines for evaluation
- output_results/ — comparison results, visualizations, and summaries
- student_papers/ - source PDF files
- anonymize_data.py - document anonymization script
- evaluate_ai.py - work evaluation script
- compare_ai.py — model comparison and visualization script
- requirements.txt - project dependencies
