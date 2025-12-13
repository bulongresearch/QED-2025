# naive_multi_seed.py

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_generation import generate_real_dataset  # same generator as splits


# ---------- Load real data splits ----------

def load_real_splits():
    df_train = pd.read_csv("train_real.csv")
    df_val = pd.read_csv("val_real.csv")
    df_test = pd.read_csv("test_real.csv")
    df_head = pd.read_csv("test_head.csv")
    df_tail = pd.read_csv("test_tail.csv")
    return df_train, df_val, df_test, df_head, df_tail


# ---------- Label helpers ----------

def prepare_xy_from_xtrue(df: pd.DataFrame):
    X = df[["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    y = df["x_true"].to_numpy()
    return X, y

def prepare_xy_from_xnoisy(df: pd.DataFrame):
    X = df[["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    y = df["x_noisy"].to_numpy()
    return X, y

def prepare_xy_from_xmodel(df: pd.DataFrame):
    X = df[["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    y = df["x_model"].to_numpy().astype(int)
    return X, y


def evaluate_model(name, clf, df_test, df_head, df_tail):
    X_test, y_test = prepare_xy_from_xtrue(df_test)
    X_head, y_head = prepare_xy_from_xtrue(df_head)
    X_tail, y_tail = prepare_xy_from_xtrue(df_tail)

    y_pred_test = clf.predict(X_test)
    y_pred_head = clf.predict(X_head)
    y_pred_tail = clf.predict(X_tail)

    acc_all = accuracy_score(y_test, y_pred_test)
    acc_head = accuracy_score(y_head, y_pred_head)
    acc_tail = accuracy_score(y_tail, y_pred_tail)

    print(f"[{name}] Test overall={acc_all:.4f}, HEAD={acc_head:.4f}, TAIL={acc_tail:.4f}")
    return acc_all, acc_head, acc_tail


# ---------- Surprisal stats on real test set ----------

def surprisal_stats_on_real(clf, df_test: pd.DataFrame):
    X_test, _ = prepare_xy_from_xtrue(df_test)
    proba = clf.predict_proba(X_test)
    p_max = proba.max(axis=1)
    surprisal = -np.log(p_max + 1e-12)

    median_S = float(np.median(surprisal))
    p10_S = float(np.percentile(surprisal, 10))
    p90_S = float(np.percentile(surprisal, 90))
    return median_S, p10_S, p90_S


# ---------- Synthetic generation ----------

def generate_synthetic_batch(clf, n_samples: int, seed: int = 0) -> pd.DataFrame:
    """
    Draw new (a,b,c,x_true) from the ground-truth generator and label with current model.
    """
    df_truth = generate_real_dataset(n_samples, seed=seed)

    X, _ = prepare_xy_from_xtrue(df_truth)
    proba = clf.predict_proba(X)
    p_max = proba.max(axis=1)
    class_idx = proba.argmax(axis=1)

    # decode back into original label space
    classes = clf.classes_
    x_model = classes[class_idx]

    surprisal = -np.log(p_max + 1e-12)

    df_syn = df_truth.copy()
    df_syn["x_model"] = x_model
    df_syn["p_max"] = p_max
    df_syn["surprisal"] = surprisal

    return df_syn


# ---------- Filtering strategies (collapse baselines) ----------

def filter_synthetic(df_syn: pd.DataFrame, condition_base: str):
    """
    Only the three original conditions:

    - A_no_filter: keep everything
    - B_high_surprisal: top 30% highest surprisal
    - C_goldilocks: middle 60% surprisal band
    """
    s = df_syn["surprisal"]

    if condition_base == "A_no_filter":
        return df_syn.copy()

    if condition_base == "B_high_surprisal":
        thr = s.quantile(0.70)
        return df_syn.loc[s >= thr].copy()

    if condition_base == "C_goldilocks":
        low = s.quantile(0.20)
        high = s.quantile(0.80)
        return df_syn.loc[(s >= low) & (s <= high)].copy()

    raise ValueError(f"Unknown condition base: {condition_base}")


# ---------- Metrics on synthetic ----------

def compute_diversity_entropy(df_syn: pd.DataFrame):
    """
    Simple class-distribution entropy on x_model.
    """
    if len(df_syn) == 0:
        return 0.0
    counts = df_syn["x_model"].value_counts(normalize=True)
    return float(-(counts * np.log(counts + 1e-12)).sum())


def synthetic_label_error_rate(df_syn_filtered: pd.DataFrame):
    """
    Fraction of synthetic labels that disagree with x_true.
    """
    if len(df_syn_filtered) == 0:
        return np.nan
    mismatches = (df_syn_filtered["x_model"] != df_syn_filtered["x_true"]).sum()
    return mismatches / len(df_syn_filtered)


# ---------- Train next model (student) ----------

def train_next_model(df_train_real: pd.DataFrame,
                     df_syn_filtered: pd.DataFrame,
                     max_synth: int,
                     real_per_gen: int = 2500,
                     seed: int = 0):
    """
    Strong student: 400 trees, unlimited depth.
    Train on small real subset + synthetic.
    """
    # cap synthetic size
    if len(df_syn_filtered) > max_synth:
        df_syn_filtered = df_syn_filtered.sample(
            n=max_synth, random_state=seed
        ).reset_index(drop=True)

    # sample anchor real subset
    n_real = min(len(df_train_real), real_per_gen)
    df_real_sample = df_train_real.sample(n=n_real, random_state=seed).reset_index(drop=True)

    X_real, y_real = prepare_xy_from_xnoisy(df_real_sample)
    X_syn, y_syn = prepare_xy_from_xmodel(df_syn_filtered)

    X_train = np.concatenate([X_real, X_syn], axis=0)
    y_train = np.concatenate([y_real, y_syn], axis=0)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------- CONTROL: strong model trained on REAL ONLY ----------

def run_control_real_only(seed: int = 123):
    """
    D_control_real_only:
    Strong 400-tree model trained on all real data, no synthetic.
    """
    print("\n=== Running CONTROL: D_control_real_only (real only, no self-training) ===")

    df_train_real, df_val, df_test, df_head, df_tail = load_real_splits()

    X_train, y_train = prepare_xy_from_xnoisy(df_train_real)

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    acc_all, acc_head, acc_tail = evaluate_model(
        "D_control_real_only", clf, df_test, df_head, df_tail
    )
    median_S, p10_S, p90_S = surprisal_stats_on_real(clf, df_test)

    record = {
        "condition": "D_control_real_only",
        "generation": 0,
        "acc_all": acc_all,
        "acc_head": acc_head,
        "acc_tail": acc_tail,
        "diversity_entropy": np.nan,
        "median_surprisal_test": median_S,
        "p10_surprisal_test": p10_S,
        "p90_surprisal_test": p90_S,
        "n_synth_filtered": 0,
        "synthetic_label_error": np.nan,
    }

    df_rec = pd.DataFrame([record])
    df_rec.to_csv("results_D_control_real_only.csv", index=False)
    print("Saved CONTROL results to results_D_control_real_only.csv")


# ---------- Main self-training loop per condition & seed ----------

def run_condition_for_seed(condition_base: str,
                           seed_id: int,
                           T_generations: int = 20,
                           synth_per_gen: int = 20000,
                           max_synth_for_training: int = 5000):
    """
    Self-training for a single base condition (A/B/C) and one seed.
    Condition name for CSV/logging will be:

        <condition_base>_weak_seed<seed_id>

    so the analysis script can group by base_condition = <condition_base>_weak.
    """
    # this base_seed drives synthetic sampling + student training randomness
    base_seed = 1000 * seed_id
    condition_name = f"{condition_base}_weak_seed{seed_id}"

    print(f"\n=== Running condition {condition_name} (base={condition_base}, seed={seed_id}) ===")

    # Real data
    df_train_real, df_val, df_test, df_head, df_tail = load_real_splits()

    # Base teacher M0 (weak)
    clf_current = joblib.load("base_model_M0.joblib")

    records = []

    # Gen 0 metrics
    acc_all, acc_head, acc_tail = evaluate_model(
        f"{condition_name} Gen0", clf_current, df_test, df_head, df_tail
    )
    median_S, p10_S, p90_S = surprisal_stats_on_real(clf_current, df_test)

    records.append({
        "condition": condition_name,
        "generation": 0,
        "acc_all": acc_all,
        "acc_head": acc_head,
        "acc_tail": acc_tail,
        "diversity_entropy": np.nan,
        "median_surprisal_test": median_S,
        "p10_surprisal_test": p10_S,
        "p90_surprisal_test": p90_S,
        "n_synth_filtered": 0,
        "synthetic_label_error": np.nan,
    })

    # Generations 1..T
    for t in range(1, T_generations + 1):
        print(f"\n[{condition_name}] Generation {t}: generating synthetic...")
        seed_t = base_seed + t

        df_syn = generate_synthetic_batch(
            clf_current, n_samples=synth_per_gen, seed=seed_t
        )
        print(f"  Raw synthetic count = {len(df_syn)}")

        df_syn_f = filter_synthetic(df_syn, condition_base)
        print(f"  Filtered synthetic count = {len(df_syn_f)}")

        noise_rate = synthetic_label_error_rate(df_syn_f)
        diversity_H = compute_diversity_entropy(df_syn_f)

        print(f"  Synthetic label error = {noise_rate:.4f}")
        print(f"  Diversity entropy (class) = {diversity_H:.4f}")

        clf_next = train_next_model(
            df_train_real,
            df_syn_f,
            max_synth=max_synth_for_training,
            real_per_gen=2500,
            seed=base_seed + 100 * t,
        )

        acc_all, acc_head, acc_tail = evaluate_model(
            f"{condition_name} Gen{t}", clf_next, df_test, df_head, df_tail
        )
        median_S, p10_S, p90_S = surprisal_stats_on_real(clf_next, df_test)

        records.append({
            "condition": condition_name,
            "generation": t,
            "acc_all": acc_all,
            "acc_head": acc_head,
            "acc_tail": acc_tail,
            "diversity_entropy": diversity_H,
            "median_surprisal_test": median_S,
            "p10_surprisal_test": p10_S,
            "p90_surprisal_test": p90_S,
            "n_synth_filtered": len(df_syn_f),
            "synthetic_label_error": noise_rate,
        })

        clf_current = clf_next

    df_rec = pd.DataFrame(records)
    out_name = f"results_{condition_name}.csv"
    df_rec.to_csv(out_name, index=False)
    print(f"Saved results to {out_name}")


# ---------- Entry point ----------

if __name__ == "__main__":
    # 1) Real-only control once (optional)
    run_control_real_only(seed=123)

    # 2) Baseline self-training: A/B/C, seeds 0–4
    base_conditions = ["A_no_filter", "B_high_surprisal", "C_goldilocks"]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    for cond in base_conditions:
        for seed in seeds:

            run_condition_for_seed(cond, seed_id=seed, T_generations=20)
