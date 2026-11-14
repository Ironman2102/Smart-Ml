# SmartML

An AI-assisted data science platform built with Streamlit. Upload a CSV or Excel dataset and SmartML analyses it, cleans it, detects the problem type, recommends features and models, trains them, and uses local LLM agents to explain and judge the results.

## Features

- **Dataset analysis and cleaning**: profiling, missing values, basic cleaning
- **Problem detection**: regression vs. classification
- **Feature analysis**: which columns are useful
- **Model training**: scikit-learn models plus a from-scratch linear regression trained with gradient descent
- **LLM agents** (run locally via [Ollama](https://ollama.com), model `qwen3:4b`):
  - Data quality agent
  - Feature agent
  - Model strategy agent
  - Judge agent that compares trained models

## Project structure

```
SmartML/
├── app.py                          # Streamlit dashboard (entry point)
├── ml/
│   ├── analyzer.py                 # Dataset profiling
│   ├── cleaner.py                  # Data cleaning
│   ├── problem.py                  # Problem-type detection
│   ├── features.py                 # Feature analysis
│   ├── trainer.py                  # Model training and evaluation
│   ├── optimizers.py               # Gradient descent from scratch (NumPy)
│   └── gradient_descent_engine.py  # Preprocessing + GD training pipeline
├── llm/
│   ├── data_quality_agent.py
│   ├── feature_agent.py
│   ├── model_strategy_agent.py
│   └── judge_agent.py
├── scripts/
│   └── plot_loss.py                # Learning-rate comparison plot
├── requirements.txt
└── README.md
```

## Getting started

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Install Ollama and pull the model used by the agents
ollama pull qwen3:4b

# 3. Run the app
streamlit run app.py
```

### Learning-rate comparison plot

```bash
python -m scripts.plot_loss
```

Run it from the project root. It plots the MSE loss curves of the gradient descent implementation for several learning rates.
