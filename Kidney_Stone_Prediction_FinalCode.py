# ============================================================
# KIDNEY STONE PREDICTION USING MACHINE LEARNING
# ============================================================

# -----------------------------
# Import Libraries
# -----------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

# Handle imbalance
from imblearn.over_sampling import SMOTE

# -----------------------------
# Load Dataset
# -----------------------------

print("\nLoading Dataset...\n")

# CHANGE FILE PATH
file_path = r"C:\Users\bawej\Downloads\kidney_stone_dataset_augmented.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:\n")
print(df.head())

# -----------------------------
# Remove Leakage Columns
# -----------------------------

drop_cols = [
    "Patient_ID",
    "Stone_Type",
    "Stone_Size_mm",
    "Severity_Level"
]

df.drop(columns=drop_cols, inplace=True, errors='ignore')

print("\nColumns after cleanup:\n")
print(df.columns)

# -----------------------------
# Handle Missing Values
# -----------------------------

print("\nMissing Values:\n")
print(df.isnull().sum())

num_cols = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy='median')

df[num_cols] = imputer.fit_transform(df[num_cols])

# -----------------------------
# Encode Categorical Columns
# -----------------------------

cat_cols = df.select_dtypes(include='object').columns

label_encoders = {}

for col in cat_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col].astype(str))

    label_encoders[col] = le

# -----------------------------
# Define Features & Target
# -----------------------------

target_column = "Kidney_Stone_Present"

X = df.drop(target_column, axis=1)

y = df[target_column]

feature_names = X.columns.tolist()

print("\nFeatures Shape:", X.shape)

print("Target Shape:", y.shape)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Size:", X_train.shape)

print("Test Size :", X_test.shape)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Apply SMOTE
# -----------------------------

print("\nApplying SMOTE...\n")

smote = SMOTE(
    random_state=42,
    k_neighbors=3
)

X_train_scaled, y_train = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("Balanced Class Distribution:\n")

print(pd.Series(y_train).value_counts())

# -----------------------------
# Define Models
# -----------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=11,
            weights='distance',
            metric='euclidean',
            p=2
        ),

    "SVM":
        SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            random_state=42
        ),

    "AdaBoost":
        AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )
}

# -----------------------------
# Model Training
# -----------------------------

results = []

best_model = None
best_auc = 0

print("\nTraining Models...\n")

for name, model in models.items():

    print(f"\nTraining {name}...\n")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    # Cross Validation
    cv_scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=5,
        scoring='roc_auc'
    )

    cv_auc = cv_scores.mean()

    # Save results
    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        specificity,
        auc,
        cv_auc
    ])

    print("Accuracy   :", round(accuracy, 4))
    print("Precision  :", round(precision, 4))
    print("Recall     :", round(recall, 4))
    print("F1 Score   :", round(f1, 4))
    print("Specificity:", round(specificity, 4))
    print("ROC-AUC    :", round(auc, 4))
    print("CV5-AUC    :", round(cv_auc, 4))

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    # Save best model
    if auc > best_auc:

        best_auc = auc

        best_model = model

        best_model_name = name

# -----------------------------
# Final Results Table
# -----------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "ROC-AUC",
        "CV5-AUC"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nFINAL MODEL COMPARISON\n")

print(results_df)

print("\nBest Model:", best_model_name)

print("Best ROC-AUC:", round(best_auc, 4))

# -----------------------------
# Confusion Matrix
# -----------------------------

y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(f'Confusion Matrix - {best_model_name}')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()

# -----------------------------
# ROC Curve
# -----------------------------

plt.figure(figsize=(8, 6))

for name, model in models.items():

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    auc = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.show()

# -----------------------------
# Feature Importance
# -----------------------------

if best_model_name in ["Random Forest", "XGBoost"]:

    importance = best_model.feature_importances_

    feature_importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by='Importance',
        ascending=False
    )

    print("\nTop 10 Important Features:\n")

    print(feature_importance.head(10))

    plt.figure(figsize=(10,6))

    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance.head(10)
    )

    plt.title("Top Feature Importance")

    plt.show()

# -----------------------------
# Save Model
# -----------------------------

artifacts = {

    "model": best_model,

    "scaler": scaler,

    "feature_names": feature_names,

    "label_encoders": label_encoders,

    "model_name": best_model_name
}

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    current_dir,
    "kidney_stone_model.pkl"
)

joblib.dump(
    artifacts,
    model_path
)

print("\nModel saved at:")
print(model_path)

print("\nModel Saved Successfully!")


# -----------------------------
# Live Prediction System
# -----------------------------

print("\nLIVE PREDICTION SYSTEM\n")

loaded_artifacts = joblib.load(
    "kidney_stone_model.pkl"
)

loaded_model = loaded_artifacts["model"]

loaded_scaler = loaded_artifacts["scaler"]

loaded_features = loaded_artifacts["feature_names"]

loaded_encoders = loaded_artifacts["label_encoders"]

sample_data = {}

# IMPORTANT FEATURES ONLY
important_features = [

    "Age",
    "Gender",
    "Urine_Calcium_mg_day",
    "Urine_Oxalate_mg_day",
    "Urine_Volume_L_day",
    "Serum_UricAcid_mg_dL",
    "Serum_Potassium_mEq_L",
    "Daily_Water_Intake_L",
    "Physical_Activity_Level",
    "Previous_Kidney_Stone"
]

print("Enter Patient Values:\n")

for feature in important_features:

    while True:

        try:

            # Gender handling
            if feature == "Gender":

                value = input(
                    "Gender (Male/Female): "
                ).strip().lower()

                if value == "male":

                    sample_data[feature] = 1
                    break

                elif value == "female":

                    sample_data[feature] = 0
                    break

                else:

                    print("Enter Male or Female.")

            # Physical Activity handling
            elif feature == "Physical_Activity_Level":

                value = input(
                    "Physical Activity (Low/Moderate/High): "
                ).strip().lower()

                mapping = {
                    "low": 0,
                    "moderate": 1,
                    "high": 2
                }

                if value in mapping:

                    sample_data[feature] = mapping[value]
                    break

                else:

                    print("Enter Low, Moderate or High.")

            # Previous stone handling
            elif feature == "Previous_Kidney_Stone":

                value = input(
                    "Previous Kidney Stone (Yes/No): "
                ).strip().lower()

                if value == "yes":

                    sample_data[feature] = 1
                    break

                elif value == "no":

                    sample_data[feature] = 0
                    break

                else:

                    print("Enter Yes or No.")

            # Numeric features
            else:

                value = float(input(f"{feature}: "))

                sample_data[feature] = value

                break

        except ValueError:

            print("Invalid input!")

# Fill remaining features with median
for feature in feature_names:

    if feature not in sample_data:

        sample_data[feature] = X[feature].median()

# Create DataFrame
sample_df = pd.DataFrame([sample_data])

# Correct column order
sample_df = sample_df[feature_names]

# Scale
sample_scaled = loaded_scaler.transform(sample_df)

# Predict
prediction = loaded_model.predict(sample_scaled)[0]

probability = loaded_model.predict_proba(sample_scaled)[0][1]

print("\nPREDICTION RESULT\n")

if prediction == 1:

    print("Kidney Stone Detected")

else:

    print("No Kidney Stone Detected")

print(f"Prediction Probability: {probability:.4f}")

# Risk Levels
if probability < 0.35:

    print("Risk Level: LOW")

elif probability < 0.70:

    print("Risk Level: MODERATE")

else:

    print("Risk Level: HIGH")

print("\nProject Completed Successfully!")