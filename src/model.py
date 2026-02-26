import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config import MODEL_PATH, MAX_ITER


def train_model(x_train, y_train):
    """
    Train Logistic Regression baseline model.
    """

    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(x_train, y_train)

    return model


def cross_validate_model(model, x_train, y_train):
    """
    Perform 5-fold cross validation using ROC-AUC.
    """

    scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    print("\nCross Validation (ROC-AUC)")
    print("-" * 40)
    print(f"Scores: {scores}")
    print(f"Mean ROC-AUC: {scores.mean():.4f}")
    print("-" * 40)

    return scores


def save_model(model):
    """
    Save trained model to disk.
    """

    joblib.dump(model, MODEL_PATH)