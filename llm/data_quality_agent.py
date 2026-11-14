
import ollama


def ask_data_quality_agent(
    target_column,
    problem_type,
    analysis
):
    missing_values = analysis["missing_values"]
    outliers = analysis["outliers"]

    prompt = f"""
You are the Data Quality Agent in SmartML,
an AI-powered machine learning platform.

Your job is to recommend data quality improvements
using only the evidence supplied by Python.

TARGET COLUMN:
{target_column}

ML PROBLEM TYPE:
{problem_type}

DATASET SIZE:
Rows: {analysis["rows"]}
Columns: {analysis["columns"]}

MISSING VALUES BY COLUMN:
{missing_values.to_string()}

DUPLICATE ROWS:
{analysis["duplicate_rows"]}

POTENTIAL NUMERICAL OUTLIERS:
{outliers.to_string()}

MISSING VALUE PERCENTAGES:
{analysis["missing_percentages"].to_string()}

OUTLIER DETAILS:
{analysis["outlier_details"]}


RULES:
- Use only evidence supplied by Python.
- Never invent counts, percentages, row counts, or values.
- Report missing-value percentages exactly as supplied by Python.
- Treat IQR outliers as flagged observations, not confirmed errors.
- Do not claim that an unusual value is invalid without evidence.
- Do not invent real-world acceptable ranges.
- Do not recommend automatic outlier removal.
- For missing numerical input features, median imputation may be a reasonable baseline.
- If the target value is missing, recommend excluding that row from supervised training rather than imputing the target.
- Clearly distinguish observed facts from recommendations.
- If Python supplies only a count, do not invent the individual rows or values.
- Keep recommendations concise and actionable.
- Use only the outlier values and row indices supplied by Python.
- Never invent individual outlier values or row indices.
- If only example values are supplied, clearly label them as examples.
- IQR outliers are flagged observations, not confirmed data errors.

Return your answer in this format:

1. ISSUES FOUND
List data quality issues supported by the evidence.

2. RECOMMENDED ACTIONS
For each action, state the column, recommended action,
and reason.

3. INVESTIGATE FIRST
List anything that needs human review before changing data.

4. OVERALL ASSESSMENT
Briefly assess the dataset's data quality.
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]
