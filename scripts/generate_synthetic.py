import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, fraud_rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Time: seconds since first transaction, uniform over ~2 days
    time = rng.uniform(0, 172800, size=rows)

    # Base PCA-like components V1..V28 ~ N(0, 1)
    V = rng.normal(0, 1, size=(rows, 28))

    # Skewed Amount via log-normal
    amount = rng.lognormal(mean=3.0, sigma=1.0, size=rows)

    # Fraud labels
    num_frauds = max(1, int(rows * fraud_rate))
    y = np.zeros(rows, dtype=int)
    fraud_idx = rng.choice(rows, size=num_frauds, replace=False)
    y[fraud_idx] = 1

    # Inject signal: certain components and amount/time patterns slightly indicative of fraud
    # Increase V1,V2,V3 magnitude for frauds; slightly higher amount
    V[fraud_idx, 0] += rng.normal(2.5, 1.0, size=num_frauds)
    V[fraud_idx, 1] += rng.normal(-2.0, 1.0, size=num_frauds)
    V[fraud_idx, 2] += rng.normal(1.5, 0.7, size=num_frauds)
    amount[fraud_idx] *= rng.lognormal(mean=0.6, sigma=0.3, size=num_frauds)

    # Build DF
    data = {
        **{f"V{i}": V[:, i - 1] for i in range(1, 29)},
        "Time": time,
        "Amount": amount,
        "Class": y,
    }
    df = pd.DataFrame(data)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic credit card transactions dataset.")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--fraud-rate", type=float, default=0.0017)
    parser.add_argument("--out", default=str(Path("data/raw/creditcard.csv")))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(args.rows, args.fraud_rate)
    df.to_csv(out_path, index=False)

    print(f"Rows: {len(df)} | Fraud rate: {df['Class'].mean():.4%}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
