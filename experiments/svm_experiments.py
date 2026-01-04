import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def load_dataset_pickle(pickle_path: Path):
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        X = obj.get("X", obj.get("data"))
        y = obj.get("y", obj.get("labels"))
        if X is None or y is None:
            raise ValueError(f"Dict pickle must contain X/data and y/labels. Keys: {list(obj.keys())}")
        return np.asarray(X), np.asarray(y)

    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        X, y = obj
        return np.asarray(X), np.asarray(y)

    raise ValueError(f"Unknown dataset pickle format: {type(obj)}")


def measure_latency_ms_per_sample(model, X_test, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model.predict(X_test)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms for whole test set
    ms_total = float(np.median(times))
    return ms_total / max(1, len(X_test))


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../dataset.pickle", help="Path to dataset.pickle")
    ap.add_argument("--out", default="svm_results.csv", help="Output CSV path")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=5, help="Latency measurement repeats")
    ap.add_argument("--c_values", default="0.1,1,10,100", help="Comma-separated C values")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    print(f"Loading dataset from: {dataset_path}")
    X, y = load_dataset_pickle(dataset_path)

    # Basic sanity
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (n_samples, n_features). Got shape={X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    kernels = ["linear", "poly", "rbf", "sigmoid"]
    c_values = [float(x.strip()) for x in args.c_values.split(",") if x.strip()]

    rows = []
    for kernel in kernels:
        for C in c_values:
            # Standard practice for SVM: scale features
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel=kernel, C=C, gamma="scale"))  # gamma used by rbf/poly/sigmoid
            ])

            # Train time (optional; not requested but useful to mention)
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            t1 = time.perf_counter()
            train_s = t1 - t0

            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            latency_ms = measure_latency_ms_per_sample(pipe, X_test, repeats=args.repeats)

            rows.append({
                "kernel": kernel,
                "C": C,
                "accuracy": acc,
                "latency_ms_per_sample": latency_ms,
                "train_time_s": train_s,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X.shape[1],
                "seed": args.seed,
                "test_size": args.test_size
            })

            print(f"[OK] kernel={kernel:7s} C={C:<6g} acc={acc:.4f} latency={latency_ms:.3f} ms/sample")

    df = pd.DataFrame(rows).sort_values(["kernel", "C"])
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    run()
