from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model using multiple classification metrics.

    Metrics used:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    """

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("\nModel Evaluation Results")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name.upper():10}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics