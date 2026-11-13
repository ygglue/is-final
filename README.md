# ShopSense AI — 3-Paradigm E-Commerce Intelligence Suite

**ShopSense AI** is a full-stack Machine Learning web application built for the **IS Final Lab Project**. It demonstrates how **Unsupervised Learning**, **Supervised Learning**, and **Reinforcement Learning** operate together on a shared e-commerce customer dataset to solve three interconnected retail intelligence problems:

1. **Unsupervised Learning (`M2`)** — Discover behavioral customer personas using **K-Means Clustering**, **StandardScaler**, **2D PCA projection**, the **Elbow Method**, and **Silhouette Analysis**.
2. **Supervised Learning (`M3`)** — Predict customer churn probability using **Random Forest** and **Logistic Regression** classifiers, complete with a **Confusion Matrix**, **Feature Importance**, and a **Live Customer Churn Simulator**.
3. **Reinforcement Learning (`M4`)** — Optimize dynamic promotional discounts across customer segments, inventory levels, and market demand using a custom **Markov Decision Process (`RetailPricingEnv`)** and pure-NumPy **Tabular Q-Learning**.

---

## System Architecture

```mermaid
flowchart LR
  subgraph Frontend["Frontend (React 19 + Vite + Astryx Design System)"]
    UI_M1["M1 · Data Pipeline View"]
    UI_M2["M2 · Unsupervised K-Means Lab"]
    UI_M3["M3 · Supervised Churn Lab"]
    UI_M4["M4 · RL Pricing Agent Lab"]
  end

  subgraph Backend["Backend (FastAPI + Python 3.11 + uv)"]
    API["FastAPI REST API (/api/*)"]
    DP["ml/data_pipeline.py\nSynthetic Generator & CSV Loader"]
    UL["ml/unsupervised.py\nStandardScaler + KMeans + PCA"]
    SL["ml/supervised.py\nRandomForest + LogisticRegression"]
    RL["ml/reinforcement.py\nRetailPricingEnv MDP + Q-Learning"]
  end

  CSV[("backend/data/ecommerce_data.csv\n1,000 Customer Records")]

  UI_M1 & UI_M2 & UI_M3 & UI_M4 <-->|"JSON over HTTP\nVite Proxy :5173 -> :8000"| API
  API --> DP & UL & SL & RL
  DP <--> CSV
  UL & SL <-- "Reads Customer Features" --- DP
```

### Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend (`backend/`)** | Python 3.11+, **FastAPI**, **Uvicorn**, **scikit-learn**, **NumPy**, **Pandas**, **Pydantic**, managed with **`uv`** |
| **Frontend (`frontend/`)** | **React 19**, **Vite 5**, **`@astryxdesign/core`** (`v0.7.14`), **`@stylexjs/stylex`**, **`@astryxdesign/theme-neutral`** (dark mode), **Recharts**, managed with **`pnpm`** |

---

## Modules & Machine Learning Breakdown

### M1 · Data Pipeline (`backend/ml/data_pipeline.py`)
Generates and manages a realistic **1,000-customer e-commerce dataset** (`backend/data/ecommerce_data.csv`) with four latent behavioral archetypes (*VIP Loyalists*, *Regular Shoppers*, *Bargain Hunters*, and *At-Risk / Dormant*) or ingests custom uploaded CSV datasets.

| Feature | Type | Range / Unit | Role Across Paradigms |
| :--- | :--- | :--- | :--- |
| `customer_id` | `int` | `1001 – 2000` | Unique identifier |
| `age` | `int` | `18 – 70 yrs` | Supervised feature |
| `tenure_months` | `int` | `1 – 72 mos` | Supervised feature |
| `annual_spend` | `float` | `$50 – $5,000` | Unsupervised clustering & Supervised feature |
| `purchase_frequency` | `int` | `1 – 55 orders/yr` | Unsupervised clustering & Supervised feature |
| `avg_order_value` | `float` | `$15 – $500` | Unsupervised clustering & Supervised feature |
| `days_since_last_purchase` | `int` | `1 – 180 days` | Unsupervised clustering & Supervised feature |
| `discount_sensitivity` | `float` | `0.02 – 0.98` | Unsupervised clustering, Supervised feature & RL demand model |
| `return_rate` | `float` | `0.00 – 0.50` | Supervised feature |
| `support_tickets` | `int` | `0 – 10 tickets` | Supervised feature |
| `churned` | `int` | `0` (Retained) / `1` (Churned) | Supervised target label (`y`) |

---

### M2 · Unsupervised Learning (`backend/ml/unsupervised.py`)
- **Algorithm**: `KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)`
- **Preprocessing**: Z-score standardization via `StandardScaler` on 5 behavioral features (`annual_spend`, `purchase_frequency`, `avg_order_value`, `days_since_last_purchase`, `discount_sensitivity`) so high-magnitude dollar amounts do not dominate distance calculations.
- **Cluster Validation**: Sweeps $k \in \{2, \dots, 8\}$ to compute **Elbow Inertia (WCSS)** and **Silhouette Score** ($\approx 0.5064$ at $k = 4$).
- **Dimensionality Reduction**: Projects the 5D standardized feature space onto 2 principal components (`PCA(n_components=2)`) for interactive scatter visualization alongside de-standardized cluster centroid profiles.

---

### M3 · Supervised Learning (`backend/ml/supervised.py`)
- **Algorithms**:
  - `RandomForestClassifier` (configurable `n_estimators` and `max_depth`)
  - `LogisticRegression` (with internal `StandardScaler` preprocessing)
- **Train/Test Split**: Stratified `80% / 20%` split (`train_test_split(..., stratify=y, random_state=42)`).
- **Evaluation Suite**:
  - **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**
  - **2×2 Confusion Matrix** (`TN`, `FP`, `FN`, `TP`)
  - **Normalized Feature Importance** ranking across all 8 input features
- **Live Churn Simulator**: Real-time single-customer inference (`/api/supervised/predict`) returning predicted churn probability, risk tier (`HIGH RISK` vs. `LOW RISK`), and recommended retention action.

---

### M4 · Reinforcement Learning (`backend/ml/reinforcement.py`)
- **Environment (`RetailPricingEnv`)**: A custom Markov Decision Process (MDP) built from scratch without external black-box gym wrappers:
  - **State Space ($|S| = 24$)**: Cartesian product of **4 Customer Segments** (`VIP Loyalist`, `Regular Shopper`, `Bargain Hunter`, `At-Risk / Dormant`) $\times$ **3 Inventory Levels** (`Low`, `Medium`, `High`) $\times$ **2 Market Demand Tiers** (`Normal`, `Peak`).
  - **Action Space ($|A| = 5$)**: Promotional discount levels $\{0\%, 5\%, 10\%, 15\%, 20\%\}$.
  - **Reward Function**: Expected net profit per step, balancing conversion probability lift against margin erosion and holding/clearance effects:
    $$R(s, a) = \text{conversion}(s, a) \times \text{margin}(a) - \text{holding\_penalty}(s)$$
- **Agent (Tabular Q-Learning)**: Pure NumPy implementation using $\epsilon$-greedy exploration with exponential decay and the **Bellman optimality update**:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
- **Outputs**: Episode reward convergence curve, learned vs. random baseline profit lift, full **24-state optimal policy table**, and Q-value heatmap.

---

## Repository Structure

```text
is-final/
├── DESIGN.md                        # Visual & architectural design specification
├── README.md                        # Project documentation
├── backend/                         # Python FastAPI + ML backend (managed via uv)
│   ├── pyproject.toml               # Python dependencies & project metadata
│   ├── uv.lock                      # Locked dependency graph
│   ├── main.py                      # FastAPI application & REST endpoints
│   ├── data/
│   │   └── ecommerce_data.csv       # 1,000-row e-commerce customer dataset
│   └── ml/
│       ├── data_pipeline.py         # M1: Synthetic data generator & CSV validator
│       ├── unsupervised.py          # M2: StandardScaler + K-Means + PCA + Silhouette
│       ├── supervised.py            # M3: Random Forest / Logistic Regression + Simulator
│       └── reinforcement.py         # M4: RetailPricingEnv MDP + Tabular Q-Learning
└── frontend/                        # React 19 + Vite + Astryx UI (managed via pnpm)
    ├── AGENTS.md                    # Astryx Design System conventions
    ├── package.json                 # Frontend dependencies & scripts
    ├── pnpm-lock.yaml               # Locked pnpm dependency graph
    ├── pnpm-workspace.yaml          # pnpm build configuration
    ├── vite.config.js               # Vite dev server & /api proxy -> 127.0.0.1:8000
    └── src/
        ├── main.jsx                 # React root & Astryx CSS reset
        ├── astryx-themes.js         # Generated Astryx neutral theme registration
        ├── App.jsx                  # AppShell + SideNav + Module switcher
        └── components/
            ├── DatasetView.jsx      # M1: Dataset explorer, schema & CSV uploader
            ├── UnsupervisedTab.jsx  # M2: K-Means controls, Elbow/Silhouette & 2D PCA
            ├── SupervisedTab.jsx    # M3: Classifier trainer, Confusion Matrix & Simulator
            └── ReinforcementTab.jsx # M4: Q-Learning trainer, reward curve & 24-state policy
```

---

## Getting Started

### Prerequisites
- **[`uv`](https://docs.astral.sh/uv/)** (Python package & environment manager, Python `>=3.11`)
- **[`pnpm`](https://pnpm.io/)** (Node.js `>=22.13.0` recommended for Astryx CLI)

### 1. Start the FastAPI Backend (`port 8000`)

```bash
cd backend
uv sync
uv run uvicorn main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000` (interactive Swagger docs at `http://127.0.0.1:8000/docs`).

### 2. Start the React Frontend (`port 5173`)

In a second terminal:

```bash
cd frontend
pnpm install
pnpm dev
```

Open **`http://localhost:5173`** in your browser. Vite automatically proxies `/api/*` requests to `http://127.0.0.1:8000`.

### 3. Production Build & Astryx Design System Audit

To verify the production bundle and Astryx component purity:

```bash
cd frontend
pnpm build
pnpm exec astryx doctor --built .
```

---

## REST API Reference

| Method | Endpoint | Module | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Core | Health check (`{"status": "ok"}`) |
| `GET` | `/api/dataset` | `M1` | Summary stats, feature schema, and 50-row preview |
| `POST` | `/api/dataset/regenerate` | `M1` | Regenerate synthetic dataset (`n_samples`, `seed`) |
| `POST` | `/api/dataset/upload` | `M1` | Upload and validate a custom `.csv` dataset |
| `POST` | `/api/unsupervised/run` | `M2` | Run K-Means (`k=2..8`), Elbow/Silhouette sweep, and 2D PCA |
| `POST` | `/api/supervised/train` | `M3` | Train Random Forest or Logistic Regression churn model |
| `POST` | `/api/supervised/predict` | `M3` | Run live churn probability prediction for a single customer |
| `POST` | `/api/reinforcement/train` | `M4` | Train Tabular Q-Learning pricing agent (`episodes`, `alpha`, `gamma`, `epsilon`) |
