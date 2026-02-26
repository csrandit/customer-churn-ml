import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PATH


def run_pipeline():

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = df.columns.str.replace(" ", "")
    df.columns = df.columns.str.strip()

    # Convert total charges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop leakage columns
    df = df.drop(columns=[
        "CustomerID",
        "ChurnLabel",
        "ChurnScore",
        "CLTV",
        "ChurnReason",
        "City",
        "LatLong",
        "Latitude",
        "Longitude",
        "ZipCode",
        "State",
        "Country"
    ])

    # Define features and target
    x = df.drop("ChurnValue", axis=1)
    y = df["ChurnValue"]

    # Feature types
    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = x.select_dtypes(include=["object"]).columns

    # Preprocessing
    from sklearn.pipeline import Pipeline as SkPipeline

    numeric_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Model pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    # Split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model.fit(x_train, y_train)

    # Predict
    y_proba = model.predict_proba(x_test)[:, 1]
    # Custom threshold
    threshold = 0.3
    y_pred = (y_proba >= threshold).astype(int)

    # -----------------------------
    # Metrics
    # -----------------------------
    print("\nModel Performance:\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label="Logistic Regression")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # -----------------------------
    # Cross Validation
    # -----------------------------
    cv_scores = cross_val_score(model, x, y, cv=5, scoring="roc_auc")

    print("\nCross-Validation ROC-AUC Scores:", cv_scores)
    print("Mean ROC-AUC:", np.mean(cv_scores))


if __name__ == "__main__":
    run_pipeline()