import os
import re
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

GUIDELINE_FILE = "guideline_files/Reference_Grades.csv"
MODEL_DIR = "grades_results"
OUTPUT_DIR = "output_results"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "model_comparison_summary.txt")
PLOT_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def preprocess_dataframe(df):
    df = df.drop(columns=["Feedback"], errors='ignore')
    df = df[~df.apply(lambda row: row.astype(str).str.contains("ERROR", na=False)).any(axis=1)]
    df["Total"] = df["Total"].str.rstrip('%')
    df["Total"] = pd.to_numeric(df["Total"], errors='coerce')
    df = df.dropna(subset=["Total"])
    df.set_index(df.columns[0], inplace=True)
    return df

def extract_max_score(column_name):
    match = re.search(r"\((\d+)\)$", column_name.strip())
    return int(match.group(1)) if match else None

guideline_df = preprocess_dataframe(pd.read_csv(GUIDELINE_FILE))
MAX_SCORES = {col: extract_max_score(col) for col in guideline_df.columns if col != "Total" and extract_max_score(col) is not None}

model_accuracies = {}
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".csv")]

for model_file in tqdm(model_files, desc="Comparing Models", unit="model"):
    model_name = model_file.replace("Grades_", "").replace(".csv", "")
    model_df = preprocess_dataframe(pd.read_csv(os.path.join(MODEL_DIR, model_file)))

    common_teams = model_df.index.intersection(guideline_df.index)
    common_columns = [col for col in model_df.columns if col in guideline_df.columns and col != "Total"]

    results = []
    total_similarity_sum = 0
    total_cells = 0

    total_iterations = len(common_teams) * len(common_columns)
    progress_bar = tqdm(total=total_iterations, desc=f"→ {model_name}", unit="cell", leave=True)

    for team in common_teams:
        for col in common_columns:
            ref_score = guideline_df.at[team, col]
            model_score = model_df.at[team, col]
            try:
                ref_val = float(ref_score)
                model_val = float(model_score)
                abs_diff = abs(ref_val - model_val)
                max_score = MAX_SCORES[col]
                similarity = 100 - (abs_diff / max_score) * 100
            except ValueError:
                ref_val = None
                model_val = None
                abs_diff = None
                pct_error = None
                similarity = None

            results.append({
                "Team": team,
                "Criterion": col,
                "Ref Score": ref_val,
                "Model Score": model_val,
                "Similarity (%)": similarity
            })

            if similarity is not None:
                total_similarity_sum += similarity
                total_cells += 1

            progress_bar.update(1)

    progress_bar.close()

    accuracy = total_similarity_sum / total_cells if total_cells else 0
    model_accuracies[model_name] = accuracy

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, f"comparison_{model_name}.csv"), index=False)

    heatmap_data = df_results.pivot(index="Criterion", columns="Team", values="Similarity (%)")
    criterion_order = df_results["Criterion"].drop_duplicates().tolist()
    heatmap_data = heatmap_data.reindex(criterion_order, axis=0)
    heatmap_data = heatmap_data.reindex(
        sorted(heatmap_data.columns, key=lambda x: int(''.join(filter(str.isdigit, x)))),
        axis=1
    )
    num_cols = len(heatmap_data.columns)
    num_rows = len(heatmap_data.index)
    max_col_label_len = max(len(str(col)) for col in heatmap_data.columns)
    max_row_label_len = max(len(str(row)) for row in heatmap_data.index)

    col_width = max(0.6, 0.12 * max_col_label_len)
    row_height = max(0.5, 0.25)

    fig_width = max(10, round(num_cols * col_width))
    fig_height = max(6, round(num_rows * row_height + 2))

    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        cbar_kws={'label': 'Similarity (%)'},
        annot_kws={"size": 8}
    )

    plt.xticks(rotation=90, ha='center')
    plt.title(f"Similarity Heatmap for {model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{model_name}_heatmap.png"))
    plt.close()

sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)

with open(SUMMARY_FILE, "w") as f:
    f.write("Model Accuracy Summary\n")
    f.write("=======================\n")
    for model, acc in sorted_models:
        f.write(f"{model}: {acc:.2f}% accuracy\n")
    f.write("\nModel Ranking (Most to Least Accurate):\n")
    for i, (model, _) in enumerate(sorted_models, start=1):
        f.write(f"{i}. {model}\n")

plt.figure(figsize=(8, 5))
sns.barplot(x=[m for m, _ in sorted_models], y=[a for _, a in sorted_models], hue=[m for m, _ in sorted_models], legend=False)
plt.ylabel("Average Similarity (%)")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=60, ha='center')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "model_accuracy_comparison.png"))
plt.close()

print("✅ Processing complete. Check the output_results/ folder.")
