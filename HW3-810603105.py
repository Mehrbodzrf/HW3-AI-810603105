
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    precision_recall_fscore_support, log_loss, hamming_loss
)
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore, f_oneway

file_path = "milling_machine.csv"  # assuming it's in same folder
df = pd.read_csv(file_path)

print("🔹 اطلاعات کلی دیتا:")
print(df.info())
print("\n🔹 آمار توصیفی ویژگی‌های عددی:")
print(df.describe())

missing_df = pd.DataFrame({
    'تعداد مقادیر گمشده': df.isnull().sum(),
    'درصد مقادیر گمشده (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_df)

df_corr = df.drop(columns=["Failure Types"]).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Between Numerical Features")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, y="Failure Types", order=df["Failure Types"].value_counts().index, palette="Set2")
plt.title("Distribution of Tool Failure Types")
plt.tight_layout()
plt.show()

top_features = df_corr["Tool Wear (Seconds)"].abs().sort_values(ascending=False)[1:4].index.tolist()
print("\n🔹 سه ویژگی با بیشترین همبستگی با Tool Wear:", top_features)

for feature in top_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature].dropna(), bins=30, color='skyblue')
    plt.title(f"Value Distribution of {feature}")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

for feature in top_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Failure Types", y=feature, palette="Set3")
    plt.title(f"{feature} by Failure Type")
    plt.tight_layout()
    plt.show()

print("\n🔬 ANOVA Between Features and Failure Types:")
for feature in top_features:
    groups = [df[df["Failure Types"] == ft][feature].dropna() for ft in df["Failure Types"].dropna().unique()]
    stat, p = f_oneway(*groups)
    print(f"\nFeature: {feature} | F-statistic: {stat:.2f} | P-value: {p:.4e} {'✅' if p < 0.05 else '⚠️'}")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
object_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)
for col in object_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

z_scores = np.abs(zscore(df[numeric_cols]))
df = df[(z_scores < 3).all(axis=1)]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n✅ Cleaning Complete - Final Shape:", df.shape)

df["Failure_Binary"] = df["Failure Types"].apply(lambda x: 0 if x == "No Failure" else 1)

plt.figure(figsize=(6, 4))
sns.countplot(x="Failure_Binary", data=df, palette="Set2")
plt.title("Class Distribution: Failure_Binary")
plt.tight_layout()
plt.show()

X_bin = df.drop(columns=["Failure Types", "Failure_Binary"])
y_bin = df["Failure_Binary"]
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_bin, y_bin)

sns.countplot(x=y_resampled, palette="Set2")
plt.title("Class Distribution After SMOTE")
plt.xlabel("Failure_Binary")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (linear)": SVC(kernel='linear', probability=True),
    "SVM (rbf)": SVC(kernel='rbf', probability=True)
}

model_metrics = []
model_preds = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    model_preds[name] = (y_pred, y_prob)
    model_metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

grid = GridSearchCV(SVC(kernel='rbf', probability=True), {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
}, cv=5)
grid.fit(X_train, y_train)
best_svm = grid.best_estimator_
y_pred_best = best_svm.predict(X_test)
y_prob_best = best_svm.predict_proba(X_test)[:, 1]
model_preds["Best SVM (GridSearch)"] = (y_pred_best, y_prob_best)

model_metrics.append({
    "Model": "Best SVM (GridSearch)",
    "Accuracy": accuracy_score(y_test, y_pred_best),
    "Precision": precision_score(y_test, y_pred_best),
    "Recall": recall_score(y_test, y_pred_best),
    "F1 Score": f1_score(y_test, y_pred_best)
})

results_df = pd.DataFrame(model_metrics).sort_values(by="F1 Score", ascending=False)
print("\n📊 Model Performance Comparison:")
print(results_df)

for name, (y_pred, _) in model_preds.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 6))
for name, (_, y_prob) in model_preds.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
for name, (_, y_prob) in model_preds.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=name)
plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()

f1_per_class = f1_score(y_test, y_pred_best, average=None)
plt.figure(figsize=(6, 4))
plt.bar(['No Failure', 'Failure'], f1_per_class, color=['blue', 'red'])
plt.title("F1-Score per Class (Best SVM)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(y_test.values[:100], label='True Labels', marker='o')
plt.plot(y_pred_best[:100], label='Predicted Labels', marker='x')
plt.title("Predicted vs True Labels (First 100 Samples)")
plt.legend()
plt.tight_layout()
plt.show()

X_multi = df.drop(columns=["Failure Types", "Failure_Binary"])
y_multi = df["Failure Types"]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
svm_ovr = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))
svm_ovo = OneVsOneClassifier(SVC(kernel='linear', probability=True, random_state=42))

rf.fit(X_train_m, y_train_m)
svm_ovr.fit(X_train_m, y_train_m)
svm_ovo.fit(X_train_m, y_train_m)

models_multi = {
    "Random Forest": rf,
    "SVM (OvR)": svm_ovr,
    "SVM (OvO)": svm_ovo
}

multi_reports = []
for name, model in models_multi.items():
    y_pred = model.predict(X_test_m)
    report = classification_report(y_test_m, y_pred, output_dict=True, zero_division=0)
    multi_reports.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_m, y_pred),
        "Macro Precision": report["macro avg"]["precision"],
        "Macro Recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"]
    })

    cm = confusion_matrix(y_test_m, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    print(f"\n📌 Classification Report for {name}:")
    print(classification_report(y_test_m, y_pred, zero_division=0))

results_multi = pd.DataFrame(multi_reports)
print("\n📊 Multi-Class Classification Comparison:")
print(results_multi)

print("\n🔍 GridSearchCV for Random Forest:")
rf_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train_m, y_train_m)
print("Best RF Params:", rf_grid.best_params_)
print("Best RF Score:", rf_grid.best_score_)

print("\n🔍 GridSearchCV for SVM (rbf):")
svm_params = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001]
}
svm_grid = GridSearchCV(SVC(kernel='rbf', probability=True), svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train_m, y_train_m)
print("Best SVM Params:", svm_grid.best_params_)
print("Best SVM Score:", svm_grid.best_score_)

rf_importances = rf.feature_importances_
svm_coefs = np.mean(np.abs(svm_ovr.estimators_[0].coef_), axis=0)

importance_df = pd.DataFrame({
    "Feature": X_multi.columns,
    "RandomForest Importance": rf_importances,
    "SVM (OvR) Coef Importance": svm_coefs
}).sort_values(by="RandomForest Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=importance_df, y="Feature", x="RandomForest Importance", color="skyblue")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=importance_df.sort_values(by="SVM (OvR) Coef Importance", ascending=False),
            y="Feature", x="SVM (OvR) Coef Importance", color="salmon")
plt.title("SVM (OvR) Feature Coefficient Importance")
plt.tight_layout()
plt.show()

print("\n📌 Feature Importance Table:")
print(importance_df)

comparison_metrics = []

for name, model in models_multi.items():
    y_pred = model.predict(X_test_m)
    y_proba = model.predict_proba(X_test_m) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test_m, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test_m, y_pred, average='macro', zero_division=0)
    hamming = hamming_loss(y_test_m, y_pred)
    logloss = log_loss(y_test_m, y_proba) if y_proba is not None else np.nan

    comparison_metrics.append({
        "Model": name,
        "Accuracy": acc,
        "Macro Precision": pr,
        "Macro Recall": rc,
        "Macro F1": f1,
        "Hamming Loss": hamming,
        "Log Loss": logloss
    })

metrics_df = pd.DataFrame(comparison_metrics).sort_values(by="Macro F1", ascending=False)
print("\n📈 Full Metric Comparison - Multi-Class Models:")
print(metrics_df)

y_test_bin = label_binarize(y_test_m, classes=np.unique(y_test_m))
classes = np.unique(y_test_m)
n_classes = len(classes)

ovr_proba = svm_ovr.predict_proba(X_test_m)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], ovr_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {classes[i]} (AUC={roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Multi-Class ROC Curve - OvR (SVM)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], ovr_proba[:, i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], label=f"Class {classes[i]}")
plt.title("Multi-Class Precision-Recall Curve - OvR (SVM)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()

f1_per_class = f1_score(y_test_m, svm_ovr.predict(X_test_m), average=None, labels=classes)
plt.figure(figsize=(8, 5))
plt.bar(classes, f1_per_class, color='purple')
plt.title("F1 Score per Class - SVM OvR")
plt.xlabel("Class")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
