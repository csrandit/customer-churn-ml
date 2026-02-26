import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Apply business-driven feature engineering.

    These features help the model capture customer behavior patterns.
    """

    # 1️⃣ Tenure segmentation
    # Short-tenure customers are usually at higher churn risk
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 36, 72],
            labels=["new", "mid", "loyal"]
        )

    # 2️⃣ Average monthly spend
    # Detect customers who pay high amounts in short periods
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 3️⃣ High-value customer flag
    # Business may prioritize retaining high-value customers
    if "MonthlyCharges" in df.columns:
        median_value = df["MonthlyCharges"].median()

        df["is_high_value"] = np.where(
            df["MonthlyCharges"] > median_value,
            1,
            0
        )

    return df