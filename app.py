import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from ml.analyzer import analyze_dataset
from ml.cleaner import clean_dataset
from ml.problem import determine_problem_type
from ml.features import analyze_features
from ml.trainer import train_models

from llm.feature_agent import ask_feature_agent
from llm.data_quality_agent import ask_data_quality_agent
from llm.model_strategy_agent import ask_model_strategy_agent
from llm.judge_agent import ask_judge_agent


# -------------------------
# SmartML Dashboard
# -------------------------

st.title("SmartML")
st.write("AI-Powered Data Science Platform")


# -------------------------
# Dataset Upload
# -------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to get started.")
    st.stop()

# Reset stored results when a different file is uploaded
if st.session_state.get("uploaded_name") != uploaded_file.name:
    st.session_state.clear()
    st.session_state["uploaded_name"] = uploaded_file.name

if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.success("Dataset uploaded successfully!")


# -------------------------
# Prediction Goal
# -------------------------

st.subheader("Prediction Goal")

target_description = st.text_input(
    "What do you want SmartML to predict?",
    placeholder="Example: Predict the MPG of a car based on its characteristics"
)

target_column = st.selectbox(
    "Which column contains the value you want to predict?",
    df.columns
)

if not target_description.strip():
    st.info("Describe your prediction goal to continue.")
    st.stop()

st.info(f"Prediction goal: {target_description}")
st.success(f"Target column selected: {target_column}")


# -------------------------
# Determine Problem Type
# -------------------------

problem_type = determine_problem_type(df, target_column)
st.info(f"ML Problem Type: {problem_type}")


# -------------------------
# Dataset Analysis
# -------------------------

analysis = analyze_dataset(df, target_column)
feature_analysis = analyze_features(df, target_column)

st.subheader("Dataset Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", analysis["rows"])

with col2:
    st.metric("Columns", analysis["columns"])

with col3:
    st.metric("Duplicate Rows", analysis["duplicate_rows"])

with col4:
    st.metric("Missing Values", int(analysis["missing_values"].sum()))

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Numerical Columns")
st.write(analysis["numerical_columns"])

st.subheader("Categorical Columns")
st.write(analysis["categorical_columns"])

st.subheader("Missing Values by Column")
st.dataframe(analysis["missing_values"])

st.subheader("Missing Value Percentages")
st.dataframe(analysis["missing_percentages"].rename("Missing (%)"))

st.subheader("Outlier Details")
st.json(analysis["outlier_details"])

st.subheader("Potential Features")
st.write(analysis["potential_features"])

st.subheader("Feature Relationship with Target")
st.dataframe(analysis["target_correlations"])


# -------------------------
# Feature Diagnostics
# -------------------------

st.subheader("Feature Diagnostics")
st.dataframe(feature_analysis)


# -------------------------
# AI Feature Strategy
# -------------------------

st.subheader("AI Feature Strategy")

if st.button("Ask AI to Analyze Features"):
    with st.spinner("Feature Strategy Agent is analyzing..."):
        st.session_state["feature_recommendations"] = ask_feature_agent(
            target_column,
            problem_type,
            feature_analysis,
            analysis["target_correlations"]
        )

if "feature_recommendations" in st.session_state:
    st.subheader("AI Feature Recommendations")
    st.write(st.session_state["feature_recommendations"])


# -------------------------
# AI Data Quality Analysis
# -------------------------

st.subheader("AI Data Quality Analysis")

if st.button("Ask AI to Analyze Data Quality"):
    with st.spinner("Data Quality Agent is analyzing..."):
        st.session_state["data_quality_recommendations"] = (
            ask_data_quality_agent(
                target_column,
                problem_type,
                analysis
            )
        )

if "data_quality_recommendations" in st.session_state:
    st.write(st.session_state["data_quality_recommendations"])


# -------------------------
# AI Model Strategy
# -------------------------

st.subheader("AI Model Strategy")

if st.button("Ask AI to Recommend Models"):
    with st.spinner("Model Strategy Agent is analyzing..."):
        st.session_state["model_strategy_recommendations"] = (
            ask_model_strategy_agent(
                target_column,
                problem_type,
                analysis,
                feature_analysis
            )
        )

if "model_strategy_recommendations" in st.session_state:
    st.markdown(st.session_state["model_strategy_recommendations"])


# -------------------------
# AI Judge
# -------------------------

st.subheader("AI Judge")

required_agents = [
    "feature_recommendations",
    "data_quality_recommendations",
    "model_strategy_recommendations"
]

agents_ready = all(key in st.session_state for key in required_agents)

if not agents_ready:
    st.info(
        "Run all three specialist agents before asking "
        "the AI Judge to create a plan."
    )

if st.button("Create ML Plan with AI Judge"):

    if not agents_ready:
        st.warning(
            "Please run the Feature Strategy, Data Quality, "
            "and Model Strategy agents first."
        )
    else:
        try:
            with st.spinner("AI Judge is reviewing recommendations..."):
                st.session_state["judge_plan"] = ask_judge_agent(
                    target_column=target_column,
                    problem_type=problem_type,
                    analysis=analysis,
                    feature_analysis=feature_analysis,
                    feature_recommendations=st.session_state["feature_recommendations"],
                    data_quality_recommendations=st.session_state["data_quality_recommendations"],
                    model_strategy_recommendations=st.session_state["model_strategy_recommendations"]
                )

            st.success("AI Judge created a proposed plan.")

        except (ValueError, KeyError) as exc:
            st.error(f"Could not validate the Judge's plan: {exc}")

        except Exception as exc:
            st.error(f"AI Judge failed: {exc}")

if "judge_plan" in st.session_state:
    st.subheader("Proposed ML Pipeline Plan")
    st.json(st.session_state["judge_plan"])


# -------------------------
# Data Cleaning
# -------------------------

st.subheader("Data Cleaning")

if st.button("Clean Dataset"):
    cleaned_df, cleaning_report = clean_dataset(df)
    st.session_state["cleaned_df"] = cleaned_df
    st.session_state["cleaning_report"] = cleaning_report
    st.success("Dataset cleaned successfully!")

if "cleaned_df" in st.session_state:

    report = st.session_state["cleaning_report"]

    st.write("Duplicates removed:", report["duplicates_removed"])

    st.subheader("Cleaned Dataset")
    st.dataframe(st.session_state["cleaned_df"].head())

    st.subheader("Missing Values After Cleaning")
    st.dataframe(report["missing_values_after"])


# -------------------------
# Model Training
# -------------------------

st.subheader("Model Training")

if st.button("Train ML Models"):

    with st.spinner("Training models..."):

        model_results, loss_history, best_model, best_model_name = train_models(
            df, target_column
        )

        st.session_state["model_results"] = model_results
        st.session_state["gd_loss_history"] = loss_history
        st.session_state["best_model"] = best_model
        st.session_state["best_model_name"] = best_model_name

    st.success(f"Training completed! Selected model: {best_model_name}")


if "model_results" in st.session_state:

    st.subheader("Model Comparison")
    st.dataframe(st.session_state["model_results"])


# -------------------------
# Model Comparison
# -------------------------

if "model_results" in st.session_state:
    st.subheader("Model Comparison")
    st.dataframe(
        st.session_state["model_results"],
        use_container_width=True
    )


# -------------------------
# Prediction Interface
# -------------------------

if "best_model" in st.session_state:

    st.subheader("Make a Prediction")

    st.write(
        f"Selected model: **{st.session_state['best_model_name']}**"
    )

    # Identify supported input features
    X = df.drop(columns=[target_column])

    numerical_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    supported_features = [
        col for col in X.columns
        if col in numerical_features or col in categorical_features
    ]

    unsupported_features = [
        col for col in X.columns
        if col not in supported_features
    ]

    if unsupported_features:
        st.warning(
            "These columns are not supported by the current prediction "
            f"pipeline and will be excluded: {unsupported_features}"
        )

    if not supported_features:
        st.error("No supported input features are available.")

    else:
        st.write("Enter the input values below:")

        with st.form("prediction_form"):

            input_values = {}

            for column in supported_features:

                if column in numerical_features:

                    median_value = pd.to_numeric(
                        X[column], errors="coerce"
                    ).median()

                    if pd.isna(median_value):
                        median_value = 0.0

                    input_values[column] = st.number_input(
                        f"{column}",
                        value=float(median_value),
                        key=f"prediction_{column}"
                    )

                else:
                    unique_values = X[column].dropna().unique().tolist()

                    if 0 < len(unique_values) <= 50:

                        input_values[column] = st.selectbox(
                            f"{column}",
                            options=unique_values,
                            key=f"prediction_{column}"
                        )

                    else:
                        mode_values = X[column].mode(dropna=True)

                        default_value = (
                            str(mode_values.iloc[0])
                            if not mode_values.empty
                            else ""
                        )

                        input_values[column] = st.text_input(
                            f"{column}",
                            value=default_value,
                            key=f"prediction_{column}"
                        )

            predict_button = st.form_submit_button("Predict")

        if predict_button:

            input_df = pd.DataFrame(
                [input_values],
                columns=supported_features
            )

            prediction = st.session_state["best_model"].predict(
                input_df
            )

            st.success("Prediction completed!")

            st.metric(
                label=f"Predicted {target_column}",
                value=f"{prediction[0]:.2f}"
            )

# -------------------------
# Download Trained Model
# -------------------------

if "best_model" in st.session_state:

    st.subheader("Download Trained Model")

    st.write(
        f"Selected model: **{st.session_state['best_model_name']}**"
    )

    model_file = pickle.dumps(st.session_state["best_model"])

    st.download_button(
        label="Download Trained Model (.pkl)",
        data=model_file,
        file_name="smartml_trained_model.pkl",
        mime="application/octet-stream"
    )


# -------------------------
# Gradient Descent Visualization
# -------------------------

loss_history = st.session_state.get("gd_loss_history")

if loss_history is not None:

    st.subheader("Gradient Descent Optimization")

    if len(loss_history) > 1:

        use_log = st.checkbox("Log scale", value=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(loss_history)), loss_history, label="Training Loss")

        if use_log:
            ax.set_yscale("log")

        ax.set_title("Gradient Descent Convergence")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Mean Squared Error (MSE)")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Initial Loss", f"{loss_history[0]:.6f}")
        m2.metric("Final Loss", f"{loss_history[-1]:.6f}")
        m3.metric("Iterations", len(loss_history) - 1)

    else:
        st.warning("Not enough loss history to generate a graph.")