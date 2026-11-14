
import json
import ollama


# These are the regression models currently available
# in SmartML's existing trainer.
SUPPORTED_MODELS = [
    "LinearRegression",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
]


def ask_judge_agent(
    target_column,
    problem_type,
    analysis,
    feature_analysis,
    feature_recommendations,
    data_quality_recommendations,
    model_strategy_recommendations,
):
    """
    Combine the three specialist agents into a structured
    machine-learning experiment plan.

    The Judge proposes a plan. Python must validate it
    before any experiment is executed.
    """

    if problem_type != "Regression":
        raise ValueError(
            "The current SmartML trainer supports regression "
            "only. Classification support will be added later."
        )

    # Convert Python analysis into a compact, JSON-friendly summary.
    dataset_summary = {
        "rows": int(analysis["rows"]),
        "columns": int(analysis["columns"]),
        "numerical_columns": list(analysis["numerical_columns"]),
        "categorical_columns": list(analysis["categorical_columns"]),
        "missing_values": {
            str(key): int(value)
            for key, value in analysis["missing_values"].items()
        },
        "duplicate_rows": int(analysis["duplicate_rows"]),
        "target_column": target_column,
        "input_features": [
            str(column)
            for column in feature_analysis["Feature"].tolist()
        ],
    }

    prompt = f"""
You are the AI Judge inside SmartML, an AI-powered
machine-learning platform.

Your responsibility is to combine recommendations from
three specialist agents into a cautious, testable plan.

PYTHON DATASET SUMMARY:
{json.dumps(dataset_summary, indent=2, default=str)}

FEATURE STRATEGY AGENT:
{feature_recommendations}

DATA QUALITY AGENT:
{data_quality_recommendations}

MODEL STRATEGY AGENT:
{model_strategy_recommendations}

AVAILABLE REGRESSION MODELS:
{json.dumps(SUPPORTED_MODELS)}

YOUR RULES:
- Use Python's dataset summary as the source of truth for dataset facts.
- The target column is {target_column}. Never include it as an input feature.
- Select only models from the supplied available-model list.
- Do not invent model scores, correlations, counts, or other statistics.
- Do not claim a model is best before Python evaluates it.
- Treat feature removal as an experiment, not an established fact.
- Every proposed feature must exist in the supplied input feature list.
- Never propose dropping the target column.
- Use median imputation for numerical features as the initial baseline.
- Use most-frequent imputation and one-hot encoding for categorical features.
- Scale numerical features for LinearRegression only if appropriate for the
  selected implementation and pipeline; scaling is optional for this baseline.
- All learned preprocessing must be fitted on training data only.
- Recommend MAE and RMSE, with R2 as an optional complementary metric.
- Recommend five-fold cross-validation as an initial comparison strategy,
  where feasible. Python must implement and verify the evaluation procedure.
- Do not write Python code.
- Do not claim that any action has already been executed.
- If specialist agents disagree, prefer verified Python facts and propose
  a testable experiment instead of guessing.
- Return ONLY one valid JSON object. Do not use Markdown fences.
- The preprocessing object MUST contain the boolean field
  "fit_preprocessing_on_training_data_only": true.
- This field must be the JSON boolean true, not the string "true".
- Never omit this field.

Return this exact JSON structure:
{{
  "problem_type": "Regression",
  "target_column": "{target_column}",
  "candidate_models": [
    "LinearRegression",
    "RandomForestRegressor",
    "GradientBoostingRegressor"
  ],
  "feature_experiments": [
    {{
      "name": "baseline_without_high_cardinality_feature",
      "drop_features": ["name"],
      "reason": "A hypothesis to evaluate, not a proven improvement."
    }},
    {{
      "name": "baseline_with_all_input_features",
      "drop_features": [],
      "reason": "Provides a comparison baseline."
    }}
  ],
  "preprocessing": {{
    "numerical_imputation": "median",
    "categorical_imputation": "most_frequent",
    "categorical_encoding": "one_hot",
    "scale_for_linear_regression": true,
    "fit_preprocessing_on_training_data_only": true
  }},
  "evaluation": {{
    "method": "5_fold_cross_validation",
    "metrics": ["MAE", "RMSE", "R2"]
  }},
  "reasoning": [
    "Short explanation grounded in the supplied evidence."
  ],
  "warnings": [
    "Risks or uncertainties that Python should account for."
  ]
}}

The JSON above describes the required structure, not guaranteed
decisions. Select only valid features and supported models.
Do not include "name" in drop_features if it is not an input feature.
Keep the reasoning concise.
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format="json",
    )

    raw_output = response["message"]["content"]

    try:
        plan = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The AI Judge did not return valid JSON."
        ) from exc

    # Basic validation before the plan is returned to the app.
    if not isinstance(plan, dict):
        raise ValueError("The AI Judge output must be a JSON object.")

    if plan.get("problem_type") != problem_type:
        raise ValueError("The Judge returned an incorrect problem type.")

    if plan.get("target_column") != target_column:
        raise ValueError("The Judge returned an incorrect target column.")

    candidate_models = plan.get("candidate_models")

    if not isinstance(candidate_models, list) or not candidate_models:
        raise ValueError("The Judge must recommend at least one model.")

    invalid_models = [
        model for model in candidate_models
        if model not in SUPPORTED_MODELS
    ]

    if invalid_models:
        raise ValueError(
            f"The Judge recommended unsupported models: {invalid_models}"
        )

    valid_features = set(dataset_summary["input_features"])

    experiments = plan.get("feature_experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("The Judge must provide feature experiments.")

    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("Each feature experiment must be an object.")

        dropped = experiment.get("drop_features", [])

        if not isinstance(dropped, list):
            raise ValueError("'drop_features' must be a list.")

        if target_column in dropped:
            raise ValueError("The target column cannot be dropped as a feature.")

        unknown_features = set(dropped) - valid_features

        if unknown_features:
            raise ValueError(
                f"The Judge referenced unknown features: "
                f"{sorted(unknown_features)}"
            )

    
    preprocessing = plan.get("preprocessing")

    if not isinstance(preprocessing, dict):
        raise ValueError(
            "The Judge's plan must contain a preprocessing object."
        )

    if preprocessing.get(
        "fit_preprocessing_on_training_data_only"
    ) is not True:
        raise ValueError(
            "The Judge's plan must set "
            "'fit_preprocessing_on_training_data_only' to "
            "the JSON boolean true. "
            f"Received: {preprocessing.get('fit_preprocessing_on_training_data_only', 'MISSING')!r}"
        )


    return plan
