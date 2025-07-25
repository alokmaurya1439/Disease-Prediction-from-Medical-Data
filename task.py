import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# --- 1. Data Acquisition ---
# Load the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

# The target variable: 0 for malignant, 1 for benign
# For medical diagnosis, it's common to make the "positive" class (the disease) as 1.
# In this dataset, 0 is malignant, 1 is benign. We'll flip it for consistency if needed,
# but for now, we'll keep it as is and interpret metrics carefully.
# Let's check the target names:
print(f"Original target names: {data.target_names}")
# Output: ['malignant' 'benign']
# So, 0 corresponds to malignant, 1 to benign.

# Let's adjust the target such that 1 means 'Malignant' (the "positive" class for disease prediction)
# and 0 means 'Benign'. This is a common convention for recall/precision of the disease.
y_adjusted = y.apply(lambda x: 0 if x == 1 else 1) # 1 if original 0 (malignant), 0 if original 1 (benign)
target_names_adjusted = ['benign', 'malignant'] # 0: benign, 1: malignant

print(f"Adjusted target names: {target_names_adjusted}")
print("\nDataset Info:")
X.info()
print("\nTarget Variable Distribution (Adjusted):")
print(y_adjusted.value_counts(normalize=True))

# --- 2. Data Preprocessing & Splitting ---
# The Breast Cancer dataset is relatively clean with all numerical features.
# We primarily need feature scaling.

# Split data into training and testing sets
# Using stratify=y_adjusted to maintain class proportions in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_adjusted, test_size=0.25, random_state=42, stratify=y_adjusted
)

print(f"\nTraining features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Testing target shape: {y_test.shape}")

# Define the preprocessor (StandardScaler for numerical features)
# The Breast Cancer dataset has no categorical features that need one-hot encoding
preprocessor = StandardScaler()

# --- 3. Model Training and Evaluation ---

# Initialize classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Support Vector Machine': SVC(random_state=42, probability=True), # probability=True for ROC-AUC
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False) # Suppress warning
}

results = {}
predictions = {}
probabilities = {}

print("\n--- Model Training and Evaluation ---")
for name, classifier in classifiers.items():
    print(f"\nTraining and evaluating {name}...")

    # Create a pipeline: preprocess -> classify
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of the positive class (Malignant = 1)

    # Store results
    predictions[name] = y_pred
    probabilities[name] = y_proba

    # Evaluate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc_auc}

    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision (Malignant): {prec:.4f}") # Precision for the positive class (Malignant)
    print(f"  Recall (Malignant): {rec:.4f}")     # Recall for the positive class (Malignant)
    print(f"  F1-Score (Malignant): {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"\n  Classification Report for {name}:\n{classification_report(y_test, y_pred, target_names=target_names_adjusted)}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names_adjusted)
    cmp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

# --- 4. Compare Models (Visual Summary) ---
print("\n--- Overall Model Performance Comparison ---")
results_df = pd.DataFrame(results).T
print(results_df)

# Plot ROC Curves for all models
plt.figure(figsize=(10, 8))
for name, proba in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- 5. Hyperparameter Tuning Example (Random Forest) ---
print("\n--- Hyperparameter Tuning Example (Random Forest) ---")

# Define a smaller, representative parameter grid for demonstration
# In a real project, you'd expand this.
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Create a pipeline for GridSearchCV
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Use GridSearchCV for hyperparameter tuning
# We'll optimize for 'recall' because false negatives (missing cancer) are critical
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters for Random Forest: {grid_search.best_params_}")
print(f"Best cross-validation recall score: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
best_rf_model = grid_search.best_estimator_
y_pred_tuned = best_rf_model.predict(X_test)
y_proba_tuned = best_rf_model.predict_proba(X_test)[:, 1]

print("\n--- Evaluation of Tuned Random Forest Model ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precision (Malignant): {precision_score(y_test, y_pred_tuned):.4f}")
print(f"Recall (Malignant): {recall_score(y_test, y_pred_tuned):.4f}")
print(f"F1-Score (Malignant): {f1_score(y_test, y_pred_tuned):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_tuned):.4f}")
print(f"\nClassification Report for Tuned Random Forest:\n{classification_report(y_test, y_pred_tuned, target_names=target_names_adjusted)}")

# Plot Confusion Matrix for Tuned Random Forest
fig, ax = plt.subplots(figsize=(6, 5))
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
cmp_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_tuned, display_labels=target_names_adjusted)
cmp_tuned.plot(ax=ax, cmap=plt.cm.Greens)
plt.title(f'Confusion Matrix for Tuned Random Forest')
plt.show()

print("\n--- End of Disease Prediction Project ---")
