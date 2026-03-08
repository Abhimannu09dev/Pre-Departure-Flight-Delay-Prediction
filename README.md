# Flight Delay Prediction — AI Model Comparison

A machine learning project that compares three classification models — **Logistic Regression**, **Random Forest**, and **XGBoost** — for predicting whether a flight will arrive 15 or more minutes late, using the 2007 U.S. airline dataset.

---

## Problem Statement

Flight delays are a major pain point for airlines and passengers alike. This project frames delay prediction as a **binary classification problem**: given pre-departure information about a flight, can we predict whether it will be delayed by 15 or more minutes upon arrival?

---

## Dataset

- **Source:** [Harvard Dataverse — Airline On-Time Performance Data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7)
- **File used:** `2007.csv.bz2`
- **Coverage:** All domestic U.S. flights in 2007
- **Preprocessing:**
  - Cancelled and diverted flights are removed
  - Target variable `ARR_DEL15` is derived from `ArrDelay >= 15`

---

## Features Used

Only **pre-departure** features are used to ensure the model is practically useful (i.e., predictions can be made before a flight takes off):

| Feature | Description |
|---|---|
| `Month` | Month of the flight (1–12) |
| `DayOfWeek` | Day of the week (1 = Monday, 7 = Sunday) |
| `DEP_HOUR` | Scheduled departure hour, extracted from `CRSDepTime` |
| `Distance` | Flight distance in miles |

---

## Models Compared

### 1. Logistic Regression
- Solver: `lbfgs`, max iterations: 1000
- Features are **standardised** using `StandardScaler`
- Provides interpretable coefficients per feature

### 2. Random Forest
- 100 decision trees, trained with `n_jobs=-1` (parallelised)
- No scaling required
- Feature importances extracted from ensemble

### 3. XGBoost
- Gradient boosted trees: 100 estimators, learning rate 0.1, max depth 6
- Subsampling: 80% rows and columns per tree
- Evaluated with `logloss` metric during training

---

## Evaluation Metrics

Each model is assessed using:

- **Accuracy** — overall correctness across all flights
- **Classification Report** — precision, recall, and F1-score for both classes
- **Confusion Matrix** — visualised as a heatmap
- **Recall for Delayed Flights** — key metric, since missing a delay (false negative) is more costly than a false alarm
- **Feature Importance** — bar chart showing which features drive each model's predictions

---

## Project Structure

```
├── Comparision_between_all_ai_model.ipynb   # Main notebook
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `2007.csv.bz2` from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7) and update the `file_path` variable in the notebook to point to your local copy:

```python
file_path = r"path/to/your/2007.csv.bz2"
```

### 4. Run the notebook

```bash
jupyter notebook Comparision_between_all_ai_model.ipynb
```

---

## Requirements

| Package | Version |
|---|---|
| pandas | ≥ 1.3.0 |
| numpy | ≥ 1.21.0 |
| scikit-learn | ≥ 1.0.0 |
| xgboost | ≥ 1.5.0 |
| matplotlib | ≥ 3.4.0 |
| seaborn | ≥ 0.11.0 |
| jupyter | ≥ 1.0.0 |

---

## Key Findings

The notebook produces a final side-by-side comparison of all three models on:

- **Accuracy** — bar chart across all models
- **Recall for delayed flights** — critical for real-world usefulness

> Tree-based models (Random Forest and XGBoost) are expected to outperform Logistic Regression on this tabular dataset, particularly on recall for the minority (delayed) class.

---

## Possible Extensions

- Add more features (e.g., carrier, origin/destination airport, weather)
- Handle class imbalance with SMOTE or class weights
- Tune hyperparameters with `GridSearchCV` or `Optuna`
- Evaluate with ROC-AUC in addition to accuracy and recall
- Deploy as a simple web app using Flask or Streamlit
