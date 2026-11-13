"""Data generation and loading pipeline for ShopSense AI."""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV_PATH = DATA_DIR / "ecommerce_data.csv"

FEATURE_COLUMNS = [
    "annual_spend",
    "purchase_frequency",
    "avg_cart_value",
    "days_since_last_visit",
    "support_tickets",
    "tenure_months",
]

CLUSTER_FEATURES = [
    "annual_spend",
    "purchase_frequency",
    "avg_cart_value",
    "days_since_last_visit",
]


def generate_ecommerce_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic e-commerce customer dataset with realistic segments and churn signals."""
    rng = np.random.default_rng(seed)

    # Create 4 underlying customer archetypes so clustering & supervised learning find genuine structure
    archetypes = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.25, 0.25])

    annual_spend = np.zeros(n_samples)
    purchase_frequency = np.zeros(n_samples, dtype=int)
    avg_cart_value = np.zeros(n_samples)
    days_since_last_visit = np.zeros(n_samples, dtype=int)
    support_tickets = np.zeros(n_samples, dtype=int)
    tenure_months = np.zeros(n_samples, dtype=int)

    for i, arch in enumerate(archetypes):
        if arch == 0:
            # VIP Loyalists: high spend, frequent, recent visits, long tenure
            freq = int(np.clip(rng.normal(24, 5), 10, 45))
            cart = float(np.clip(rng.normal(145, 30), 60, 300))
            days = int(np.clip(rng.normal(7, 5), 1, 30))
            tickets = int(rng.poisson(0.6))
            tenure = int(np.clip(rng.normal(36, 10), 12, 60))
        elif arch == 1:
            # High-Ticket Occasional: large carts, lower frequency, moderate recency
            freq = int(np.clip(rng.normal(7, 2.5), 2, 14))
            cart = float(np.clip(rng.normal(240, 45), 130, 420))
            days = int(np.clip(rng.normal(28, 12), 5, 70))
            tickets = int(rng.poisson(1.1))
            tenure = int(np.clip(rng.normal(22, 8), 6, 48))
        elif arch == 2:
            # At-Risk / Frustrated Regulars: decent frequency, rising support tickets, lapsing recency
            freq = int(np.clip(rng.normal(12, 3.5), 4, 22))
            cart = float(np.clip(rng.normal(75, 20), 25, 140))
            days = int(np.clip(rng.normal(52, 18), 15, 110))
            tickets = int(np.clip(rng.poisson(3.4), 1, 9))
            tenure = int(np.clip(rng.normal(14, 6), 2, 36))
        else:
            # Budget / Dormant Browsers: low spend, low frequency, long absence
            freq = int(np.clip(rng.normal(3, 1.5), 1, 8))
            cart = float(np.clip(rng.normal(42, 12), 15, 85))
            days = int(np.clip(rng.normal(82, 22), 30, 150))
            tickets = int(rng.poisson(1.4))
            tenure = int(np.clip(rng.normal(8, 5), 1, 24))

        spend = round(freq * cart * rng.uniform(0.92, 1.08), 2)
        annual_spend[i] = spend
        purchase_frequency[i] = freq
        avg_cart_value[i] = round(cart, 2)
        days_since_last_visit[i] = days
        support_tickets[i] = tickets
        tenure_months[i] = tenure

    # Logistic churn probability driven by recency, support tickets, low frequency, and short tenure
    logit = (
        -1.6
        + 0.038 * (days_since_last_visit - 35)
        + 0.62 * (support_tickets - 1.5)
        - 0.09 * (purchase_frequency - 10)
        - 0.035 * (tenure_months - 18)
        + rng.normal(0, 0.35, size=n_samples)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{1000 + i}" for i in range(n_samples)],
            "annual_spend": annual_spend,
            "purchase_frequency": purchase_frequency,
            "avg_cart_value": avg_cart_value,
            "days_since_last_visit": days_since_last_visit,
            "support_tickets": support_tickets,
            "tenure_months": tenure_months,
            "churned": churned,
        }
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_CSV_PATH, index=False)
    return df


def load_dataset() -> pd.DataFrame:
    """Load the active e-commerce dataset from disk, generating it if missing."""
    if not DEFAULT_CSV_PATH.exists():
        return generate_ecommerce_dataset()
    return pd.read_csv(DEFAULT_CSV_PATH)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics and a sample preview for the frontend explorer."""
    numeric_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "std": round(float(df[col].std()), 2),
        }

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "churn_rate": round(float(df["churned"].mean()) * 100, 2) if "churned" in df.columns else 0.0,
        "stats": stats,
        "preview": df.head(15).to_dict(orient="records"),
    }
