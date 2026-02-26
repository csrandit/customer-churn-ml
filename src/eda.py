import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH


def add_percentage_labels(ax, total):
    """
    Add percentage labels on top of bars
    """
    for p in ax.patches:
        height = p.get_height()
        percentage = 100 * height / total
        ax.annotate(f'{percentage:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')


def run_eda():
    # Create reports folder if not exists
    os.makedirs("reports", exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = df.columns.str.replace(" ", "")
    df.columns = df.columns.str.strip()

    print("\nDataset Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())

    # Convert TotalCharges to numeric
    if df["TotalCharges"].dtype == "object":
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # -----------------------------
    # Churn Distribution
    # -----------------------------
    plt.figure()
    ax = sns.countplot(x="ChurnValue", data=df)
    plt.title("Customer Churn Distribution")

    total = len(df)
    add_percentage_labels(ax, total)

    plt.savefig("reports/churn_distribution.png")
    plt.close()

    # -----------------------------
    # Contract vs Churn
    # -----------------------------
    plt.figure()
    ax = sns.countplot(x="Contract", hue="ChurnValue", data=df)
    plt.xticks(rotation=45)
    plt.title("Contract Type vs Churn")

    total = len(df)
    add_percentage_labels(ax, total)

    plt.savefig("reports/contract_vs_churn.png")
    plt.close()

    # -----------------------------
    # Tenure vs Churn
    # -----------------------------
    plt.figure()
    sns.boxplot(x="ChurnValue", y="TenureMonths", data=df)
    plt.title("Tenure Months vs Churn")

    plt.savefig("reports/tenure_vs_churn.png")
    plt.close()

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.title("Correlation Matrix")

    plt.savefig("reports/correlation_matrix.png")
    plt.close()

    print("\nEDA completed. Figures saved in 'reports/' folder.")


if __name__ == "__main__":
    run_eda()