# devs_multi_seed.py

import os
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# random fix for imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from data_generation import generate_real_dataset, split_head_tail  # noqa: E402

# helper functions
def load_real_splits():
    df_train = pd.read_csv("train_real.csv")
    df_val = pd.read_csv("val_real.csv")
    df_test = pd.read_csv("test_real.csv")
    df_head = pd.read_csv("test_head.csv")
    df_tail = pd.read_csv("test_tail.csv")
    return df_train, df_val, df_test, df_head, df_tail


def prepare_xy_from_xtrue(df: pd.DataFrame):
    X = df[["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    y = df["x_true"].to_numpy().astype(int)
    return X, y

def prepare_xy_from_xnoisy(df: pd.DataFrame):
    X = df[["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    y = df["x_noisy"].to_numpy().astype(int)
    return X, y

def prepare_xy_from_xmodel(df: pd.DataFrame):
    X = df[["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    y = df["x_model"].to_numpy().astype(int)
    return X, y


def evaluate_model(name, clf, df_test, df_head, df_tail):
    X_test, y_test = prepare_xy_from_xtrue(df_test)
    X_head, y_head = prepare_xy_from_xtrue(df_head)
    X_tail, y_tail = prepare_xy_from_xtrue(df_tail)

    acc_all = accuracy_score(y_test, clf.predict(X_test))
    acc_head = accuracy_score(y_head, clf.predict(X_head))
    acc_tail = accuracy_score(y_tail, clf.predict(X_tail))

    print(f"[{name}] Test overall: {acc_all:.4f}, HEAD: {acc_head:.4f}, TAIL: {acc_tail:.4f}")
    return acc_all, acc_head, acc_tail


def surprisal_stats_on_real(clf, df_test: pd.DataFrame):
    X_test, _ = prepare_xy_from_xtrue(df_test)
    proba = clf.predict_proba(X_test)
    p_max = proba.max(axis=1)
    surprisal = -np.log(p_max + 1e-12)
    return float(np.median(surprisal)), float(np.percentile(surprisal, 10)), float(np.percentile(surprisal, 90))

# trains diverse ensemble with different algos + parameters
def train_diverse_ensemble(X_train, y_train, base_seed: int = 0):

    teachers = []

    # randomForest variants
    for i, (n_est, depth) in enumerate([(60, 10), (80, 12), (100, 8)]):
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_leaf=5,
            random_state=base_seed + i * 100,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        teachers.append(clf)

    # extraTrees
    clf = ExtraTreesClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=3,
        random_state=base_seed + 300,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    teachers.append(clf)

    # gradientBoosting
    clf = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=base_seed + 400,
    )
    clf.fit(X_train, y_train)
    teachers.append(clf)

    return teachers


def ensemble_predict(teachers: list, X: np.ndarray):
    all_preds = np.array([clf.predict(X) for clf in teachers])
    return all_preds


# ensemble agreement thresholding here
def ensemble_agreement_mask(all_preds: np.ndarray, min_agreement: int = None):
    n_teachers = all_preds.shape[0]
    if min_agreement is None:
        min_agreement = n_teachers

    from scipy import stats
    mode_pred, _ = stats.mode(all_preds, axis=0, keepdims=False)
    agreement_count = (all_preds == mode_pred).sum(axis=0)

    return agreement_count >= min_agreement, mode_pred

# predictions from all models
def predict_ensemble_consensus(teachers, X, num_classes=10):
    all_proba = []
    for clf in teachers:
        proba = clf.predict_proba(X)
        full = np.zeros((X.shape[0], num_classes))
        for i, c in enumerate(clf.classes_):
            c = int(c)
            full[:, c] = proba[:, i]
        all_proba.append(full)
    avg = np.mean(all_proba, axis=0)
    return avg.argmax(axis=1)


def evaluate_ensemble(name, teachers, df_test, df_head, df_tail):
    X_test, y_test = prepare_xy_from_xtrue(df_test)
    X_head, y_head = prepare_xy_from_xtrue(df_head)
    X_tail, y_tail = prepare_xy_from_xtrue(df_tail)

    pred_test = predict_ensemble_consensus(teachers, X_test)
    pred_head = predict_ensemble_consensus(teachers, X_head)
    pred_tail = predict_ensemble_consensus(teachers, X_tail)

    acc_all = accuracy_score(y_test, pred_test)
    acc_head = accuracy_score(y_head, pred_head)
    acc_tail = accuracy_score(y_tail, pred_tail)

    print(f"[{name}] Ensemble TEST: {acc_all:.4f}, HEAD: {acc_head:.4f}, TAIL: {acc_tail:.4f}")
    return acc_all, acc_head, acc_tail

def generate_synthetic_with_diverse_ensemble(
    teachers: list,
    n_samples: int,
    seed: int = 0,
    min_agreement: int = None,
) -> pd.DataFrame:
    df_truth = generate_real_dataset(n_samples, seed=seed)
    X, _ = prepare_xy_from_xtrue(df_truth)

    all_preds = ensemble_predict(teachers, X)
    agree_mask, consensus_pred = ensemble_agreement_mask(all_preds, min_agreement)

    # average confidence across all teachers
    all_proba = []
    for clf in teachers:
        proba = clf.predict_proba(X)
        # handles a case where some classes missing
        full_proba = np.zeros((X.shape[0], 10))
        for i, c in enumerate(clf.classes_):
            full_proba[:, c] = proba[:, i]
        all_proba.append(full_proba)

    avg_proba = np.mean(all_proba, axis=0)
    p_max = avg_proba.max(axis=1)
    surprisal = -np.log(p_max + 1e-12)

    df_syn = df_truth.copy()
    df_syn["x_model"] = consensus_pred
    df_syn["p_max"] = p_max
    df_syn["surprisal"] = surprisal
    df_syn["ensemble_agree"] = agree_mask
    df_syn["n_agree"] = (all_preds == consensus_pred).sum(axis=0)

    return df_syn

def compute_diversity_entropy(df_syn: pd.DataFrame):
    if len(df_syn) == 0:
        return 0.0
    counts = df_syn["x_model"].value_counts(normalize=True)
    return float(-(counts * np.log(counts + 1e-12)).sum())


def synthetic_label_error_rate(df_syn: pd.DataFrame):
    if len(df_syn) == 0:
        return np.nan
    return (df_syn["x_model"] != df_syn["x_true"]).mean()

# main devs loop per seed
def run_diverse_ensemble_with_accumulation_for_seed(
    label: str,
    seed_id: int,
    accumulate_data: bool,
    T_generations: int = 20,
    synth_per_gen: int = 20000,
    max_synth_per_gen: int = 5000,
    min_agreement: int = None, #set to none for unanimous agreement
):
    base_seed = 1000 * seed_id
    condition = f"{label}_seed{seed_id}"

    print(f"\n{'='*60}")
    print(f"Running: {condition}")
    print(f"  Accumulate data: {accumulate_data}")
    print(f"  Min agreement: {min_agreement if min_agreement else 'unanimous'}") #for this study, it is unanimous
    print(f"  Seed: {seed_id} (base_seed={base_seed})")
    print(f"{'='*60}")

    df_train_real_full, df_val, df_test, df_head, df_tail = load_real_splits()

    # real set for training anchor
    REAL_ANCHOR = 2500
    df_train_real = df_train_real_full.sample(
        n=REAL_ANCHOR,
        random_state=base_seed
    ).reset_index(drop=True)

    X_real_anchor, y_real_anchor = prepare_xy_from_xnoisy(df_train_real)

    #trains on the real anchor set
    teachers = train_diverse_ensemble(X_real_anchor, y_real_anchor, base_seed=base_seed)

    accumulated_synthetic = pd.DataFrame()
    records = []
    rng = np.random.default_rng(base_seed)

    # gen 0 baseline using first teacher
    acc_all, acc_head, acc_tail = evaluate_ensemble(
        f"{condition} Gen0", teachers, df_test, df_head, df_tail
    )
    median_S, p10_S, p90_S = surprisal_stats_on_real(teachers[0], df_test)

    records.append({
        "condition": condition,
        "generation": 0,
        "acc_all": acc_all,
        "acc_head": acc_head,
        "acc_tail": acc_tail,
        "diversity_entropy": np.nan,
        "median_surprisal_test": median_S,
        "p10_surprisal_test": p10_S,
        "p90_surprisal_test": p90_S,
        "n_synth_filtered": 0,
        "n_synth_accumulated": 0,
        "synthetic_label_error": np.nan,
        "agreement_rate": np.nan,
    })

    for t in range(1, T_generations + 1):
        print(f"\n[{condition}] Generation {t}")
        seed_t = base_seed + t

        # generate synthetic with ensemble
        df_syn = generate_synthetic_with_diverse_ensemble(
            teachers, n_samples=synth_per_gen, seed=seed_t, min_agreement=min_agreement
        )

        df_syn_agreed = df_syn[df_syn["ensemble_agree"]].copy()
        agreement_rate = len(df_syn_agreed) / len(df_syn)
        print(f"  Agreement rate: {agreement_rate:.2%} ({len(df_syn_agreed)}/{len(df_syn)})")

            # high-confidence filter
        HIGH_CONF_THRESH = 0.78  #tuned, 90% is too high and doesnt leave enough data
        before_conf = len(df_syn_agreed)
        df_syn_agreed = df_syn_agreed[df_syn_agreed["p_max"] >= HIGH_CONF_THRESH].copy()
        after_conf = len(df_syn_agreed)
        print(f"  High-confidence filter: {after_conf}/{before_conf} kept (p_max >= {HIGH_CONF_THRESH})")

        # agreement here for class balancing
        n_per_class = max_synth_per_gen // 10
        balanced_dfs = []
        for cls in range(10):
            cls_df = df_syn_agreed[df_syn_agreed["x_model"] == cls]
            n_sample = min(len(cls_df), n_per_class)
            if n_sample > 0:
                balanced_dfs.append(cls_df.sample(n=n_sample, random_state=rng.integers(10000)))

        df_syn_filtered = pd.concat(balanced_dfs, ignore_index=True) if balanced_dfs else pd.DataFrame()
        print(f"  Filtered (class balanced): {len(df_syn_filtered)}")

        # accumulation vs no accumulation bool used here
        if accumulate_data and len(df_syn_filtered) > 0:
            accumulated_synthetic = pd.concat([accumulated_synthetic, df_syn_filtered], ignore_index=True)
            if len(accumulated_synthetic) > 50000:
                accumulated_synthetic = accumulated_synthetic.sample(n=50000, random_state=seed_t)
            df_syn_for_training = accumulated_synthetic.copy()
        else:
            df_syn_for_training = df_syn_filtered.copy()

        print(f"  Total synthetic for training: {len(df_syn_for_training)}")

        noise_rate = synthetic_label_error_rate(df_syn_filtered)
        diversity_H = compute_diversity_entropy(df_syn_filtered)
        print(f"  Label error (this gen): {noise_rate:.4f}")
        print(f"  Diversity entropy (class): {diversity_H:.4f}")

        # combine small real + synthetic for training
        if len(df_syn_for_training) > 0:
            X_syn, y_syn = prepare_xy_from_xmodel(df_syn_for_training)
            X_combined = np.concatenate([X_real_anchor, X_syn], axis=0)
            y_combined = np.concatenate([y_real_anchor, y_syn], axis=0)
        else:
            X_combined = X_real_anchor
            y_combined = y_real_anchor

        # retrain ensemble
        teachers = train_diverse_ensemble(
            X_combined, y_combined,
            base_seed=base_seed + t * 1000
        )

        acc_all, acc_head, acc_tail = evaluate_ensemble(
            f"{condition} Gen{t}", teachers, df_test, df_head, df_tail
        )
        median_S, p10_S, p90_S = surprisal_stats_on_real(teachers[0], df_test)

        records.append({
            "condition": condition,
            "generation": t,
            "acc_all": acc_all,
            "acc_head": acc_head,
            "acc_tail": acc_tail,
            "diversity_entropy": diversity_H,
            "median_surprisal_test": median_S,
            "p10_surprisal_test": p10_S,
            "p90_surprisal_test": p90_S,
            "n_synth_filtered": len(df_syn_filtered),
            "n_synth_accumulated": len(accumulated_synthetic) if accumulate_data else len(df_syn_filtered),
            "synthetic_label_error": noise_rate,
            "agreement_rate": agreement_rate,
        })

    df_rec = pd.DataFrame(records)
    out_name = f"results_{condition}.csv"
    df_rec.to_csv(out_name, index=False)
    print(f"\nSaved results to {out_name}")

def run_ensemble_naive_for_seed(
    label: str = "J_ensemble_naive",
    seed_id: int = 0,
    accumulate_data: bool = True,
    T_generations: int = 20,
    synth_per_gen: int = 20000,
    max_synth_per_gen: int = 5000,
):
    #J (baseline to compare to devs) only using ensemble, no filtering
    base_seed = 1000 * seed_id
    condition = f"{label}_seed{seed_id}"

    print("\n" + "=" * 60)
    print(f"Running ENSEMBLE-ONLY baseline: {condition}")
    print(f"  Accumulate data: {accumulate_data}")
    print(f"  Seed: {seed_id} (base_seed={base_seed})")
    print("=" * 60)

    df_train_real_full, df_val, df_test, df_head, df_tail = load_real_splits()

    REAL_ANCHOR = 2500
    df_train_real = df_train_real_full.sample(
        n=REAL_ANCHOR,
        random_state=base_seed
    ).reset_index(drop=True)

    X_real_anchor, y_real_anchor = prepare_xy_from_xnoisy(df_train_real)

    # initial ensemble
    teachers = train_diverse_ensemble(
        X_real_anchor, y_real_anchor,
        base_seed=base_seed
    )

    accumulated_synthetic = pd.DataFrame()
    records = []
    rng = np.random.default_rng(base_seed)

    # gen 0
    acc_all, acc_head, acc_tail = evaluate_ensemble(
        f"{condition} Gen0",
        teachers,
        df_test,
        df_head,
        df_tail,
    )
    median_S, p10_S, p90_S = surprisal_stats_on_real(teachers[0], df_test)

    records.append({
        "condition": condition,
        "generation": 0,
        "acc_all": acc_all,
        "acc_head": acc_head,
        "acc_tail": acc_tail,
        "diversity_entropy": np.nan,
        "median_surprisal_test": median_S,
        "p10_surprisal_test": p10_S,
        "p90_surprisal_test": p90_S,
        "n_synth_filtered": 0,
        "n_synth_accumulated": 0,
        "synthetic_label_error": np.nan,
        "agreement_rate": np.nan,  # we’re not gating on agreement
    })

    # training for t generations
    for t in range(1, T_generations + 1):
        print(f"\n[{condition}] Generation {t}")
        seed_t = base_seed + t

        df_syn = generate_synthetic_with_diverse_ensemble(
            teachers,
            n_samples=synth_per_gen,
            seed=seed_t,
            min_agreement=None, # still compute agreement stats for info, not used
        )

        # log agreement
        agreement_rate = df_syn["ensemble_agree"].mean()
        print(f"  Raw synthetic count = {len(df_syn)}")
        print(f"  Ensemble agreement rate (for info only) = {agreement_rate:.2%}")

        if len(df_syn) > max_synth_per_gen:
            df_syn_filtered = df_syn.sample(
                n=max_synth_per_gen,
                random_state=seed_t
            ).reset_index(drop=True)
        else:
            df_syn_filtered = df_syn.copy()

        print(f"  Synthetic used this gen (random subset): {len(df_syn_filtered)}")

        noise_rate = synthetic_label_error_rate(df_syn_filtered)
        diversity_H = compute_diversity_entropy(df_syn_filtered)
        print(f"  Label error (this gen): {noise_rate:.4f}")
        print(f"  Diversity entropy (class): {diversity_H:.4f}")

        if accumulate_data and len(df_syn_filtered) > 0:
            accumulated_synthetic = pd.concat(
                [accumulated_synthetic, df_syn_filtered],
                ignore_index=True
            )
            # Cap total synthetic size
            if len(accumulated_synthetic) > 50000: #adjusted cap 
                accumulated_synthetic = accumulated_synthetic.sample(
                    n=50000,
                    random_state=seed_t
                )
            df_syn_for_training = accumulated_synthetic.copy()
        else:
            df_syn_for_training = df_syn_filtered.copy()

        print(f"  Total synthetic for training: {len(df_syn_for_training)}")

        # combine real + synthetic for training
        if len(df_syn_for_training) > 0:
            X_syn, y_syn = prepare_xy_from_xmodel(df_syn_for_training)
            X_combined = np.concatenate([X_real_anchor, X_syn], axis=0)
            y_combined = np.concatenate([y_real_anchor, y_syn], axis=0)
        else:
            X_combined = X_real_anchor
            y_combined = y_real_anchor

        # retrain ensemble
        teachers = train_diverse_ensemble(
            X_combined,
            y_combined,
            base_seed=base_seed + t * 1000
        )

        # evaluate ensemble
        acc_all, acc_head, acc_tail = evaluate_ensemble(
            f"{condition} Gen{t}",
            teachers,
            df_test,
            df_head,
            df_tail,
        )
        median_S, p10_S, p90_S = surprisal_stats_on_real(teachers[0], df_test)

        records.append({
            "condition": condition,
            "generation": t,
            "acc_all": acc_all,
            "acc_head": acc_head,
            "acc_tail": acc_tail,
            "diversity_entropy": diversity_H,
            "median_surprisal_test": median_S,
            "p10_surprisal_test": p10_S,
            "p90_surprisal_test": p90_S,
            "n_synth_filtered": len(df_syn_filtered),
            "n_synth_accumulated": (
                len(accumulated_synthetic)
                if accumulate_data
                else len(df_syn_filtered)
            ),
            "synthetic_label_error": noise_rate,
            "agreement_rate": agreement_rate,
        })

    df_rec = pd.DataFrame(records)
    out_name = f"results_{condition}.csv"
    df_rec.to_csv(out_name, index=False)
    print(f"\nSaved ensemble-only baseline results to {out_name}")



if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    # I: DEVS with diverse ensemble + unanimous + ACCUMULATION
    for s in seeds:
        run_diverse_ensemble_with_accumulation_for_seed(
            label="I_diverse_ensemble_accumulate",
            seed_id=s,
            accumulate_data=True,
            min_agreement=None,  # unanimous
            T_generations=20,
        )

    # K: DEVS with diverse ensemble + unanimous + NO ACCUMULATION
    for s in seeds:
        run_diverse_ensemble_with_accumulation_for_seed(
            label="K_diverse_ensemble_no_accumulate",
            seed_id=s,
            accumulate_data=False,
            min_agreement=None,  # unanimous
            T_generations=20,
        )

    # J: ENSEMBLE-ONLY self-training (no verification)
    for s in seeds:
        run_ensemble_naive_for_seed(
            label="J_ensemble_naive",
            seed_id=s,
            accumulate_data=True,   # mirror I_... for a clean comparison
            T_generations=20,
            synth_per_gen=20000,
            max_synth_per_gen=5000,

        )
