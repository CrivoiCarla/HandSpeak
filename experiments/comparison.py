import pandas as pd

df = pd.read_csv("svm_results.csv")
best = (df.sort_values(["kernel","accuracy","latency_ms_per_sample"], ascending=[True, False, True])
          .groupby("kernel", as_index=False)
          .first())
print(best[["kernel","C","accuracy","latency_ms_per_sample"]])
