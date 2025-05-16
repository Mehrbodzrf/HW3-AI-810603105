# HW3 - AI Course (810603105) – Tool Condition Monitoring 🛠️

This repository contains the full implementation of **Homework 3** for the Artificial Intelligence course, prepared by **Mehrbod Zarifi**.

📚 This project focuses on tool condition monitoring in a **milling machine** using **binary** and **multi-class classification** methods. A real-world industrial dataset was used for fault detection and diagnosis using machine learning.

---

## 📁 Contents

| File | Description |
|------|-------------|
| `HW3-810603105.ipynb` | Jupyter Notebook with full visual output |
| `HW3-810603105.py` | Clean Python script (no notebook dependencies) |
| `milling_machine.csv` | Input dataset with 10,000 samples |

---

## ⚙️ Project Description

The objective of this project is to classify the health condition of a cutting tool in a milling machine based on five sensor-based features.

### 🔍 Workflow Breakdown

#### A. Exploratory Data Analysis
- Dataset inspection using `info()` and `describe()`
- Missing value analysis
- Correlation matrix for feature dependency analysis

#### B. Data Preprocessing
- Imputation using `mean()` and `mode()`
- Outlier removal via `z-score`
- Feature scaling using `StandardScaler`

#### C. Binary Classification (`Failure` vs `No Failure`)
- Models used:
  - Logistic Regression
  - K-Nearest Neighbors
  - Support Vector Machine (Linear & RBF)
- Addressed class imbalance using **SMOTE**
- Hyperparameter tuning for RBF-SVM using `GridSearchCV`
- Evaluation:
  - Accuracy, Precision, Recall, F1
  - ROC Curve, AUC
  - Confusion Matrix
  - Precision-Recall Curve
  - Class-wise F1 score
  - Predicted vs Actual comparison plot

#### D. Multi-Class Classification (5 Failure Types)
- Models used:
  - Random Forest
  - SVM (One-vs-Rest)
  - SVM (One-vs-One)
- Hyperparameter tuning for:
  - Random Forest (`n_estimators`, `max_depth`)
  - SVM-RBF (`C`, `gamma`)
- Feature importance comparison using:
  - `feature_importances_` (Random Forest)
  - `coef_` (SVM OvR)
- Evaluation:
  - Accuracy, Macro Precision, Recall, F1
  - Hamming Loss, Log Loss
  - Multi-Class ROC and PR curves
  - Per-class F1 bar chart

---

## ▶️ How to Run

To run this project locally:

1. Make sure you have **Python 3.8+**
2. Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn openpyxl

📊 Evaluation Metrics
Both binary and multi-class tasks are evaluated with comprehensive metrics:

Accuracy, Precision, Recall, F1 Score

ROC-AUC & Precision-Recall curves

Log Loss, Hamming Loss (for multi-class)

Class-wise F1 score barplots

GridSearch-based model optimization

🧑‍💻 Author
Mehrbod Zarifi

Student ID: 810603105

Spring 2025 – AI Course

Instructor: [Dr. Masoud Shariat Panahi]

📌 Notes
This project was completed as part of the Artificial Intelligence course at [University of Tehran].

Dataset and all modeling code are included.

For a full walkthrough of results and decisions, refer to the notebook version.

This project was completed independently as part of an academic assignment.

All steps are documented with plots and analysis.

