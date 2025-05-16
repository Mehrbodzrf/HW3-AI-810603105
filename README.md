# HW3 - AI Course (810603105) – Tool Condition Monitoring 🛠️

This repository contains the full implementation of **Homework 3** for the Artificial Intelligence course, prepared by **Mehrbod Zarifi**.

📚 This project focuses on **tool condition monitoring** in a milling machine using both **binary** and **multi-class classification** techniques. A real-world industrial dataset was used for fault detection and diagnosis through machine learning.

---

## 📁 Contents

| File                  | Description                                     |
|-----------------------|-------------------------------------------------|
| `HW3-810603105.ipynb` | Jupyter Notebook with full visual output        |
| `HW3-810603105.py`    | Clean Python script (no notebook dependencies)  |
| `milling_machine.csv` | Input dataset with 10,000 samples               |

---

## ⚙️ Project Description

The objective of this project is to classify the health status of a cutting tool in a milling machine based on five sensor-based numerical features.

---

## 🔍 Workflow Breakdown

### A. Exploratory Data Analysis
- Inspection of dataset using `info()` and `describe()`
- Analysis of missing values
- Correlation matrix to examine feature relationships

### B. Data Preprocessing
- Imputation of missing values using `mean()` and `mode()`
- Outlier removal using `z-score`
- Feature scaling using `StandardScaler`

### C. Binary Classification (`Failure` vs `No Failure`)
- **Models Used:**
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM) – Linear & RBF Kernels
- **Techniques:**
  - SMOTE for handling class imbalance
  - Hyperparameter tuning with `GridSearchCV` (RBF-SVM)
- **Evaluations:**
  - Accuracy, Precision, Recall, F1 Score
  - ROC Curve & AUC
  - Precision-Recall Curve
  - Confusion Matrix
  - Class-wise F1 Score
  - Predicted vs Actual comparison plot

### D. Multi-Class Classification (5 Failure Types)
- **Models Used:**
  - Random Forest
  - SVM with One-vs-Rest (OvR)
  - SVM with One-vs-One (OvO)
- **Tuning with GridSearchCV:**
  - Random Forest: `n_estimators`, `max_depth`
  - SVM (RBF): `C`, `gamma`
- **Feature Importance:**
  - `feature_importances_` from Random Forest
  - `coef_` from SVM OvR
- **Evaluations:**
  - Accuracy, Macro Precision, Recall, F1 Score
  - Hamming Loss and Log Loss
  - ROC and Precision-Recall Curves (per class)
  - F1 Score per Class (Bar Chart)

---

📊 Evaluation Metrics
Both binary and multi-class classification tasks are evaluated with:

Accuracy, Precision, Recall, F1 Score

ROC-AUC & Precision-Recall curves

Log Loss, Hamming Loss (multi-class only)

Class-wise F1 Score Bar Plots

Hyperparameter tuning via GridSearchCV

🧑‍💻 Author
Mehrbod Zarifi

Student ID: 810603105

Spring 2025 – AI Course

Instructor: Dr. Masoud Shariat Panahi

📌 Notes
This project was completed independently as part of an academic assignment at the University of Tehran.

Dataset and full modeling pipeline are included.

The notebook provides a comprehensive walkthrough of the entire workflow with plots and analysis.

---

## ▶️ How to Run

To run this project locally:

1. Make sure you have **Python 3.8+** installed.
2. Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn openpyxl
