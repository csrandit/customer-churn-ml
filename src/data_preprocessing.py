import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import TARGET, TEST_SIZE, RANDOM_STATE, SCALER_PATH


def load_data(path):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(path)


def clean_data(df):
    """
    Basic data cleaning:
    - Convert TotalCharges to numeric
    - Fill missing values using median
    """

    if "TotalCharges" in df.columns:

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

    return df


def encode_data(df):
    """
    Convert categorical variables into numeric format.
    drop_first=True avoids multicollinearity.
    """
    return pd.get_dummies(df, drop_first=True)


def split_data(df):
    """
    Split dataset using stratified sampling
    to preserve churn distribution.
    """

    x = df.drop(TARGET, axis=1)
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test):
    """
    Apply feature scaling using StandardScaler.

    Fit only on training data to prevent data leakage.
    """

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, SCALER_PATH)

    return x_train_scaled, x_test_scaled