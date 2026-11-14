import ollama


def ask_feature_agent(
    target_column,
    problem_type,
    feature_analysis,
    target_correlations
):

    prompt = f"""
You are the Feature Strategy Agent inside an AI-powered
machine learning platform called SmartML.

Your job is to analyze the evidence provided by Python and
recommend which features should be prioritized for the ML model.

IMPORTANT RULES:
- Do not invent statistics.
- Do not calculate correlations yourself.
- Use only the information provided.
- Do not train a model.
- Do not claim that a feature improves model performance unless
  actual model results are provided.
- Give concise recommendations.

PREDICTION TARGET:
{target_column}

ML PROBLEM TYPE:
{problem_type}

FEATURE DIAGNOSTICS:
{feature_analysis.to_string(index=False)}

TARGET CORRELATIONS:
{target_correlations.to_string()}

Return your response using this structure:

HIGH PRIORITY FEATURES:
- feature: reason

MEDIUM PRIORITY FEATURES:
- feature: reason

LOW PRIORITY / INVESTIGATE:
- feature: reason

POTENTIAL FEATURES TO REMOVE:
- feature: reason

IMPORTANT NOTES:
- brief observations about redundancy, missing values,
  possible IDs, or other concerns.
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]