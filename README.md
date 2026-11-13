# ShopSense AI — 3-Paradigm Machine Learning Suite

Final Lab Project demonstrating **Unsupervised Learning**, **Supervised Learning**, and **Reinforcement Learning** in a unified Smart E-Commerce platform.

## Project Structure

- `backend/main.py` — FastAPI REST server (`http://127.0.0.1:8000`)
- `backend/pyproject.toml` — `uv` project configuration & Python dependencies
- `backend/ml/data_pipeline.py` — E-Commerce dataset generator & CSV loader (**Member 1**)
- `backend/ml/unsupervised.py` — K-Means Clustering + Elbow Method + Silhouette + PCA (**Member 2**)
- `backend/ml/supervised.py` — Random Forest / Logistic Regression Churn Classifier (**Member 3**)
- `backend/ml/reinforcement.py` — Custom MDP + Pure NumPy Tabular Q-Learning (**Member 4**)
- `frontend/` — React + Vite + Tailwind CSS + Recharts interactive dashboard (`http://localhost:5173`, managed with `pnpm`)

---

## How to Run Locally (For Live Demo)

### 1. Start the FastAPI Backend (`uv`)
```bash
cd backend
uv sync
uv run uvicorn main:app --reload --port 8000
```

### 2. Start the React Frontend (`pnpm`)
```bash
cd frontend
pnpm install
pnpm dev
```

Open **http://localhost:5173** in your browser.
