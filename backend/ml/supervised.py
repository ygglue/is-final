"""Supervised Learning module: Random Forest & Logistic Regression Churn Prediction."""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_pipeline import FEATURE_COLUMNS


class ChurnModelState:
    """Holds the trained supervised model in memory for live interactive predictions."""

    def __init__(self) -> None:
        self.model: Optional[Pipeline] = None
        self.algorithm: str = "random_forest"
        self.metrics: dict = {}


MODEL_STATE = ChurnModelState()


def train_churn_model(
    df: pd.DataFrame,
    algorithm: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = 8,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a supervised classifier to predict customer churn and return evaluation metrics."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if algorithm == "logistic_regression":
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        )
    else:
        algorithm = "random_forest"
        clf = Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=int(np.clip(n_estimators, 10, 300)),
                        max_depth=int(np.clip(max_depth, 2, 20)),
                        random_state=random_state,
                    ),
                )
            ]
        )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Extract feature importance (Gini importance for RF, |coef| for Logistic Regression)
    inner_clf = clf.named_steps["clf"]
    if algorithm == "random_forest":
        raw_imp = inner_clf.feature_importances_
    else:
        raw_imp = np.abs(inner_clf.coef_[0])
        raw_imp = raw_imp / (raw_imp.sum() + 1e-9)

    feature_importance = sorted(
        [
            {"feature": col, "importance": round(float(val) * 100, 2)}
            for col, val in zip(FEATURE_COLUMNS, raw_imp)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    metrics = {
        "algorithm": algorithm,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "feature_importance": feature_importance,
    }

    MODEL_STATE.model = clf
    MODEL_STATE.algorithm = algorithm
    MODEL_STATE.metrics = metrics
    return metrics


def predict_single_customer(df: pd.DataFrame, customer_features: dict) -> dict:
    """Predict churn probability for a hypothetical customer using the trained model."""
    if MODEL_STATE.model is None:
        train_churn_model(df)

    row = pd.DataFrame([{col: float(customer_features[col]) for col in FEATURE_COLUMNS}])
    prob_churn = float(MODEL_STATE.model.predict_proba(row)[0, 1])
    prediction = int(prob_churn >= 0.5)

    if prob_churn >= 0.65:
        risk_tier = "High Risk"
    elif prob_churn >= 0.35:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "Low Risk"

    return {
        "algorithm": MODEL_STATE.algorithm,
        "churn_probability": round(prob_churn * 100, 2),
        "predicted_class": prediction,
        "label": "Will Churn" if prediction == 1 else "Will Retain",
        "risk_tier": risk_tier,
    }
