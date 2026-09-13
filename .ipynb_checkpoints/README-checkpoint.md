# Student Performance — Grade Prediction

A small end-to-end ML pipeline that predicts a student's grade (A/B/C/Pass/F) from their Math, Science, and English scores. Built as a modular Python project — separate files for preprocessing, training, and evaluation, orchestrated by a single entry point.

## What it does

1. Loads student score data from CSV
2. Engineers features: total marks, percentage, and grade labels (derived from percentage)
3. Trains a Logistic Regression classifier on the engineered features
4. Evaluates training accuracy
5. Generates a `submission.csv` with predicted grades per student

## Project structure

```
student_performance/
├── main.py                          # entry point — orchestrates the pipeline
├── data/
│   └── student_performance.csv
└── performance_prediction/
    ├── __init__.py
    ├── data_preprocessing.py        # load, feature engineering, train/test split
    ├── model_training.py            # trains the Logistic Regression model
    └── evaluation.py                # scoring + submission file generation
```

## How it works

- **`data_preprocessing.py`** — loads the CSV, computes `Total_Score` and `Percentage` from Math/Science/English scores, derives a `Grade` label, selects features, and splits into train/test sets (keeping `Student_ID` aligned with the test split for submission).
- **`model_training.py`** — trains a `LogisticRegression` model on the training features and labels.
- **`evaluation.py`** — checks training accuracy and writes predictions to `submission.csv` alongside each student's ID.
- **`main.py`** — imports the above three modules and runs them in sequence: preprocess → train → evaluate → save submission.

## Features used

`English_Score`, `Math_Score`, `Science_Score`

## Grading logic

| Percentage | Grade |
|---|---|
| > 90 | A |
| 80–90 | B |
| 70–80 | C |
| 36–70 | Pass |
| < 36 | F |

## Setup

```bash
pip install pandas scikit-learn
```

## Run

From the project root (same level as `main.py`):

```bash
python main.py
```

**Output:** training accuracy printed to console, `submission.csv` generated with `Student_ID` and predicted `Grade` for each test-set student.

## What this project was for

Built primarily to practice writing and calling functions across multiple files — going from single-notebook code to a proper modular structure with a clear import flow (`main.py` → `performance_prediction/*`).
