# src/config.py

# ========== Paths ==========
DATA_PATH = "data/telecom_churn.csv"
MODEL_PATH = "models/logistic_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ========== Target ==========
TARGET = "Churn"

# ========== Split Settings ==========
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ========== Model Settings ==========
CV_FOLDS = 5
SCORING = "roc_auc"
MAX_ITER = 1000