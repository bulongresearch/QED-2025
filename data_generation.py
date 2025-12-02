import numpy as np
import pandas as pd

def generate_real_dataset(n_samples: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # integers
    a = rng.integers(0, 31, size=n_samples)
    b = rng.integers(0, 31, size=n_samples)
    c = rng.integers(0, 31, size=n_samples)
    d = rng.integers(0, 31, size=n_samples)

    # latent target
    s = (
        a + b + c + d           
        + (a * b) / 60          
        + np.abs(c - d) / 2     
    )

    # add noise
    a_noisy = a + rng.normal(0, 1, size=n_samples)
    b_noisy = b + rng.normal(0, 1, size=n_samples)
    c_noisy = c + rng.normal(0, 1, size=n_samples)
    d_noisy = d + rng.normal(0, 1, size=n_samples)

    # percentiles with imbalanced bins
    percentiles = [0, 15, 30, 45, 58, 70, 80, 88, 94, 98, 100]
    bins = np.percentile(s, percentiles)
    y = np.digitize(s, bins[1:-1])
    y = np.clip(y, 0, 9).astype(int)

    df = pd.DataFrame({
        "a": a_noisy,
        "b": b_noisy,
        "c": c_noisy,
        "d": d_noisy,
        "x_true": y,
    })
    return df

def split_head_tail(df: pd.DataFrame):
    df_head = df[df["x_true"] <= 4].copy()
    df_tail = df[df["x_true"] >= 5].copy()
    return df_head, df_tail

def add_label_noise(
    df: pd.DataFrame,
    noise_rate: float = 0.10,
    seed: int = 0,
    n_classes: int = 10,
) -> pd.DataFrame:
    """
    Add symmetric label noise on top of x_true.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    y_true = df["x_true"].to_numpy().astype(int)
    n = len(y_true)
    n_flip = int(noise_rate * n)

    if n_flip == 0:
        df["x_noisy"] = y_true
        return df

    flip_idx = rng.choice(n, size=n_flip, replace=False)
    y_noisy = y_true.copy()

    rand_labels = rng.integers(0, n_classes, size=n_flip)
    for k, i in enumerate(flip_idx):
        if rand_labels[k] == y_noisy[i]:
            rand_labels[k] = (rand_labels[k] + 1) % n_classes
        y_noisy[i] = rand_labels[k]

    df["x_noisy"] = y_noisy
    return df

