"""FastAPI Backend for ShopSense AI: 3-Paradigm Machine Learning Suite."""

import io
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.data_pipeline import (
    DEFAULT_CSV_PATH,
    FEATURE_COLUMNS,
    generate_ecommerce_dataset,
    get_dataset_summary,
    load_dataset,
)
from ml.reinforcement import train_q_learning
from ml.supervised import predict_single_customer, train_churn_model
from ml.unsupervised import run_customer_segmentation

app = FastAPI(title="ShopSense AI ML Suite", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UnsupervisedRequest(BaseModel):
    n_clusters: int = 4


class SupervisedTrainRequest(BaseModel):
    algorithm: str = "random_forest"
    n_estimators: int = 100
    max_depth: int = 8
    test_size: float = 0.2


class CustomerPredictRequest(BaseModel):
    annual_spend: float
    purchase_frequency: int
    avg_cart_value: float
    days_since_last_visit: int
    support_tickets: int
    tenure_months: int


class RLTrainRequest(BaseModel):
    episodes: int = 400
    alpha: float = 0.15
    gamma: float = 0.90
    epsilon_start: float = 0.90


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok", "service": "ShopSense AI"}


@app.get("/api/dataset")
def get_dataset() -> dict:
    df = load_dataset()
    return get_dataset_summary(df)


@app.post("/api/dataset/regenerate")
def regenerate_dataset(n_samples: int = 1000, seed: int = 42) -> dict:
    df = generate_ecommerce_dataset(n_samples=n_samples, seed=seed)
    return get_dataset_summary(df)


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a valid .csv file.")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    required = set(FEATURE_COLUMNS + ["churned"])
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {', '.join(sorted(missing))}",
        )

    if "customer_id" not in df.columns:
        df.insert(0, "customer_id", [f"CUST-{1000 + i}" for i in range(len(df))])

    DEFAULT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_CSV_PATH, index=False)
    return get_dataset_summary(df)


@app.post("/api/unsupervised/run")
def run_unsupervised(req: UnsupervisedRequest) -> dict:
    df = load_dataset()
    return run_customer_segmentation(df, n_clusters=req.n_clusters)


@app.post("/api/supervised/train")
def run_supervised_train(req: SupervisedTrainRequest) -> dict:
    df = load_dataset()
    return train_churn_model(
        df,
        algorithm=req.algorithm,
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        test_size=req.test_size,
    )


@app.post("/api/supervised/predict")
def run_supervised_predict(req: CustomerPredictRequest) -> dict:
    df = load_dataset()
    return predict_single_customer(df, req.model_dump())


@app.post("/api/reinforcement/train")
def run_rl_train(req: RLTrainRequest) -> dict:
    return train_q_learning(
        episodes=req.episodes,
        alpha=req.alpha,
        gamma=req.gamma,
        epsilon_start=req.epsilon_start,
    )
