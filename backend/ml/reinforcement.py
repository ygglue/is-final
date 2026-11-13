"""Reinforcement Learning module: Custom NumPy Tabular Q-Learning for Dynamic Pricing."""

import numpy as np

SEGMENTS = ["VIP Loyalist", "High-Ticket Occasional", "At-Risk Regular", "Dormant Browser"]
CHURN_LEVELS = ["Low Risk", "Medium Risk", "High Risk"]
INVENTORY_LEVELS = ["Low Stock", "High Stock"]

ACTIONS = [
    {"id": 0, "name": "Full Price (0% Off)", "discount": 0.00},
    {"id": 1, "name": "Light Nudge (10% Off)", "discount": 0.10},
    {"id": 2, "name": "Retention Promo (20% Off)", "discount": 0.20},
    {"id": 3, "name": "VIP Rescue (30% Off)", "discount": 0.30},
]

NUM_STATES = len(SEGMENTS) * len(CHURN_LEVELS) * len(INVENTORY_LEVELS)  # 4 * 3 * 2 = 24
NUM_ACTIONS = len(ACTIONS)


def encode_state(segment_idx: int, churn_idx: int, inv_idx: int) -> int:
    """Map (segment, churn_risk, inventory) tuple to integer state index 0..23."""
    return segment_idx * (len(CHURN_LEVELS) * len(INVENTORY_LEVELS)) + churn_idx * len(INVENTORY_LEVELS) + inv_idx


def decode_state(state_idx: int) -> tuple[int, int, int]:
    """Decode integer state index 0..23 back to (segment, churn_risk, inventory)."""
    inv_idx = state_idx % len(INVENTORY_LEVELS)
    rem = state_idx // len(INVENTORY_LEVELS)
    churn_idx = rem % len(CHURN_LEVELS)
    segment_idx = rem // len(CHURN_LEVELS)
    return segment_idx, churn_idx, inv_idx


class RetailPricingEnv:
    """Custom MDP Environment simulating customer conversion & profit under dynamic discounts."""

    BASE_PRICE = 100.0
    UNIT_COST = 45.0
    CHURN_PENALTY = 35.0

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def step(self, state_idx: int, action_idx: int) -> tuple[int, float]:
        """Execute a pricing action in a state, returning (next_state, immediate_reward)."""
        seg, churn, inv = decode_state(state_idx)
        discount = ACTIONS[action_idx]["discount"]
        effective_price = self.BASE_PRICE * (1.0 - discount)

        # Base conversion probability depends on segment & churn risk, boosted by discount
        seg_base = [0.68, 0.52, 0.42, 0.25][seg]
        churn_penalty_prob = [0.0, 0.14, 0.28][churn]
        discount_lift = discount * [0.35, 0.55, 0.95, 1.15][seg]

        buy_prob = float(np.clip(seg_base - churn_penalty_prob + discount_lift, 0.05, 0.96))
        purchased = self.rng.uniform() < buy_prob

        if purchased:
            profit = effective_price - self.UNIT_COST
            # Low stock + deep discount incurs an opportunity cost penalty
            if inv == 0 and discount >= 0.20:
                profit -= 12.0
            reward = profit
            next_churn = max(0, churn - (1 if discount >= 0.10 else 0))
        else:
            # Customer didn't buy; high-risk customers may churn permanently
            churned = self.rng.uniform() < (0.20 + 0.25 * churn)
            reward = -self.CHURN_PENALTY if churned else -5.0
            next_churn = min(2, churn + 1)

        next_inv = int(self.rng.choice([0, 1], p=[0.35, 0.65]))
        next_seg = seg if self.rng.uniform() < 0.7 else int(self.rng.integers(0, len(SEGMENTS)))
        next_state = encode_state(next_seg, next_churn, next_inv)
        return next_state, float(round(reward, 2))


def train_q_learning(
    episodes: int = 400,
    alpha: float = 0.15,
    gamma: float = 0.90,
    epsilon_start: float = 0.90,
    seed: int = 42,
) -> dict:
    """Train a Tabular Q-Learning agent using the Bellman update rule in pure NumPy."""
    episodes = int(np.clip(episodes, 50, 2000))
    alpha = float(np.clip(alpha, 0.01, 1.0))
    gamma = float(np.clip(gamma, 0.0, 0.99))
    epsilon_start = float(np.clip(epsilon_start, 0.05, 1.0))

    env = RetailPricingEnv(seed=seed)
    rng = np.random.default_rng(seed)

    # Initialize Q-Table of shape (24 states, 4 actions) to zeros
    q_table = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=float)

    steps_per_episode = 25
    episode_rewards = []
    epsilon = epsilon_start
    epsilon_min = 0.05
    decay = (epsilon_start - epsilon_min) / max(1, int(episodes * 0.85))

    for ep in range(episodes):
        state = int(rng.integers(0, NUM_STATES))
        total_reward = 0.0

        for _ in range(steps_per_episode):
            # Epsilon-greedy action selection
            if rng.uniform() < epsilon:
                action = int(rng.integers(0, NUM_ACTIONS))
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward = env.step(state, action)

            # Bellman Q-Learning update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
            best_next_q = float(np.max(q_table[next_state]))
            td_target = reward + gamma * best_next_q
            q_table[state, action] += alpha * (td_target - q_table[state, action])

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon - decay)
        episode_rewards.append(round(total_reward, 2))

    # Downsample reward curve to <= 80 points for clean chart rendering + moving average
    window = max(1, len(episode_rewards) // 15)
    bucket_size = max(1, episodes // 60)
    reward_curve = []
    for i in range(0, episodes, bucket_size):
        slice_vals = episode_rewards[i : i + bucket_size]
        ma_start = max(0, i - window)
        ma_val = float(np.mean(episode_rewards[ma_start : i + 1]))
        reward_curve.append(
            {
                "episode": i + 1,
                "reward": round(float(np.mean(slice_vals)), 2),
                "moving_avg": round(ma_val, 2),
            }
        )

    # Format Q-Table for interactive inspection in the UI
    q_table_rows = []
    for s_idx in range(NUM_STATES):
        seg, churn, inv = decode_state(s_idx)
        q_vals = [round(float(v), 2) for v in q_table[s_idx]]
        best_action_idx = int(np.argmax(q_table[s_idx]))
        q_table_rows.append(
            {
                "state_id": s_idx,
                "segment": SEGMENTS[seg],
                "churn_risk": CHURN_LEVELS[churn],
                "inventory": INVENTORY_LEVELS[inv],
                "q_values": q_vals,
                "best_action": ACTIONS[best_action_idx]["name"],
                "best_action_id": best_action_idx,
            }
        )

    # Benchmark Learned Policy vs. Random Policy over 200 simulated interactions
    eval_env = RetailPricingEnv(seed=seed + 99)
    learned_total = 0.0
    random_total = 0.0
    for _ in range(200):
        s = int(rng.integers(0, NUM_STATES))
        best_a = int(np.argmax(q_table[s]))
        rand_a = int(rng.integers(0, NUM_ACTIONS))
        _, r_learned = eval_env.step(s, best_a)
        _, r_random = eval_env.step(s, rand_a)
        learned_total += r_learned
        random_total += r_random

    return {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "actions": ACTIONS,
        "reward_curve": reward_curve,
        "q_table": q_table_rows,
        "benchmark": {
            "learned_policy_profit": round(learned_total, 2),
            "random_policy_profit": round(random_total, 2),
            "profit_lift_pct": round(((learned_total - random_total) / (abs(random_total) + 1e-9)) * 100, 1),
        },
    }
