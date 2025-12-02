import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_generation import generate_real_dataset, split_head_tail, add_label_noise


# shared helpers
def prepare_xy_from_xtrue(df: pd.DataFrame):
    X = df[["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    y = df["x_true"].to_numpy()
    return X, y

def prepare_xy_from_xnoisy(df: pd.DataFrame):
    X = df[["a", "b", "c", "d"]].to_numpy(dtype=np.float32)
    y = df["x_noisy"].to_numpy()
    return X, y

def main():
    # generates full dataset and splits into train/val/test
    n_samples = 100_000
    df_all = generate_real_dataset(n_samples=n_samples, seed=0)

    df_train, df_temp = train_test_split(
        df_all, test_size=0.2, random_state=0, shuffle=True
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=1, shuffle=True
    )

    df_head, df_tail = split_head_tail(df_test)

    #injects noise to labels
    df_train = add_label_noise(
        df_train,
        noise_rate=0.10,
        seed=123,
    )

    # teacher model m0 trains
    df_train_small = df_train.sample(n=2500, random_state=42)
    X_train, y_train = prepare_xy_from_xnoisy(df_train_small)

    # clean eval sets
    X_val,   y_val   = prepare_xy_from_xtrue(df_val)
    X_test,  y_test  = prepare_xy_from_xtrue(df_test)
    X_head,  y_head  = prepare_xy_from_xtrue(df_head)
    X_tail,  y_tail  = prepare_xy_from_xtrue(df_tail)

    # teacher model
    teacher = RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    teacher.fit(X_train, y_train)

    # evaluate teacher
    y_pred_train = teacher.predict(X_train)
    y_pred_val   = teacher.predict(X_val)
    y_pred_test  = teacher.predict(X_test)
    y_pred_head  = teacher.predict(X_head)
    y_pred_tail  = teacher.predict(X_tail)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_val   = accuracy_score(y_val,   y_pred_val)
    acc_test  = accuracy_score(y_test,  y_pred_test)
    acc_head  = accuracy_score(y_head,  y_pred_head)
    acc_tail  = accuracy_score(y_tail,  y_pred_tail)

    print(f"Teacher (M0) train accuracy (noisy labels): {acc_train:.3f}")
    print(f"Teacher (M0) val accuracy (clean):         {acc_val:.3f}")
    print(f"Teacher (M0) TEST overall accuracy:        {acc_test:.5f}")
    print(f"Teacher (M0) TEST HEAD accuracy:           {acc_head}")
    print(f"Teacher (M0) TEST TAIL accuracy:           {acc_tail}")

    joblib.dump(teacher, "base_model_M0.joblib")

    df_train.to_csv("train_real.csv", index=False)
    df_val.to_csv("val_real.csv", index=False)
    df_test.to_csv("test_real.csv", index=False)
    df_head.to_csv("test_head.csv", index=False)
    df_tail.to_csv("test_tail.csv", index=False)

    print("Saved base model and datasets.")

if __name__ == "__main__":
    main()