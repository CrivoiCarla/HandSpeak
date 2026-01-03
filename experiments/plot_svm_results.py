import argparse
import pandas as pd
import matplotlib.pyplot as plt

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="svm_results.csv")
    ap.add_argument("--out_acc", default="plot_accuracy.png")
    ap.add_argument("--out_lat", default="plot_latency.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Accuracy plot: one curve per kernel, x=C (log scale)
    plt.figure()
    for kernel, g in df.groupby("kernel"):
        g = g.sort_values("C")
        plt.plot(g["C"], g["accuracy"], marker="o", label=kernel)
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Accuracy")
    plt.title("SVM Accuracy vs C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_acc, dpi=200)

    # Latency plot
    plt.figure()
    for kernel, g in df.groupby("kernel"):
        g = g.sort_values("C")
        plt.plot(g["C"], g["latency_ms_per_sample"], marker="o", label=kernel)
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("Latency (ms / sample)")
    plt.title("SVM Latency vs C")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_lat, dpi=200)

    print("Saved:", args.out_acc, args.out_lat)

if __name__ == "__main__":
    run()
