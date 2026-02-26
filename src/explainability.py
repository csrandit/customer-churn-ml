import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from config import DATA_PATH


def run_explainability():

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.replace(" ", "")
    df.columns = df.columns.str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

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

    x = df.drop("ChurnValue", axis=1)
    y = df["ChurnValue"]

    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = x.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(x_train, y_train)

    # -----------------------------
    # Extract feature names
    # -----------------------------
    ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features)

    all_feature_names = list(numeric_features) + list(cat_feature_names)

    # -----------------------------
    # Extract coefficients
    # -----------------------------
    coefficients = model.named_steps["classifier"].coef_[0]

    odds_ratios = np.exp(coefficients)

    feature_importance = pd.DataFrame({
        "Feature": all_feature_names,
        "Coefficient": coefficients,
        "OddsRatio": odds_ratios
    })

    feature_importance = feature_importance.sort_values(
        by="OddsRatio", ascending=False
    )
    # -----------------------------
    # Save top features to CSV
    # -----------------------------
    top_increasing = feature_importance.head(10)
    top_decreasing = feature_importance.tail(10)

    final_report = pd.concat([top_increasing, top_decreasing])

    # create reports folder if not exists
    import os
    os.makedirs("reports", exist_ok=True)

    final_report.to_csv("reports/top_features.csv", index=False)

    print("\nTop features saved to reports/top_features.csv")

    # -----------------------------
    # Plot Top 10 Increasing Features
    # -----------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.barh(top_increasing["Feature"], top_increasing["OddsRatio"])
    plt.xlabel("Odds Ratio")
    plt.title("Top 10 Features Increasing Churn")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/top_10_increasing.png")
    plt.close()

    # -----------------------------
    # Plot Top 10 Decreasing Features
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.barh(top_decreasing["Feature"], top_decreasing["OddsRatio"])
    plt.xlabel("Odds Ratio")
    plt.title("Top 10 Features Decreasing Churn")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/top_10_decreasing.png")
    plt.close()

    print("Feature importance plots saved in reports/")

    print("\nTop 10 Features Increasing Churn:\n")
    print(feature_importance.head(10))

    print("\nTop 10 Features Decreasing Churn:\n")
    print(feature_importance.tail(10))


if __name__ == "__main__":
    run_explainability()