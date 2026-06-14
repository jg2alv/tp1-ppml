import pandas as pd
import matplotlib.pyplot as plt
import os

# Caminhos para os CSVs
baseline_path = os.path.join("..", "metricas", "baseline.csv")
fixedclipping_norm_1_path = os.path.join("..", "metricas", "fixedclipping_(0.1,1.0,10).csv")
fixedclipping_norm_5_path = os.path.join("..", "metricas", "fixedclipping_(0.1,5.0,10).csv")
adaptativeclipping_path = os.path.join("..", "metricas", "adaptativeclipping_(0.1,10).csv")

csv_paths = [baseline_path, fixedclipping_norm_1_path, fixedclipping_norm_5_path, adaptativeclipping_path]
for csv_path in csv_paths:
    if not os.path.exists(csv_path):
        print(csv_path, " não encontrado.")
        raise SystemExit


def metricas(csv_path):
    df = pd.read_csv(csv_path)

    # Os clientes não nos interessam para a comparação, apenas o modelo agregado servidor (FL-Global)
    df_global = df[df["user"] == "FL-Global"].copy()
    df_global["values"] = pd.to_numeric(df_global["values"])
    df_global["epoch"] = pd.to_numeric(df_global["epoch"])
    
    # Ordenanando
    df_global = df_global.sort_values(by="epoch")
    
    # Pivotamos para ter as métricas em colunas
    return df_global.pivot(index="epoch", columns="metric", values="values")


df_baseline = metricas(baseline_path)
df_fixed_n_1 = metricas(fixedclipping_norm_1_path)
df_fixed_n_5 = metricas(fixedclipping_norm_5_path)
df_adaptative = metricas(adaptativeclipping_path)

metrics_to_plot = ["metrics.accuracy", "metrics.f1", "metrics.LOSS"]

plt.figure(figsize=(15, 5))
for i, metric in enumerate(metrics_to_plot, 1):
    if metric in df_baseline.columns and metric in df_fixed_n_1.columns and metric in df_fixed_n_5.columns and metric in df_adaptative.columns:
        plt.subplot(1, 3, i)
        plt.plot(df_baseline.index, df_baseline[metric], label="Baseline (Sem DP)")
        plt.plot(df_fixed_n_1.index, df_fixed_n_1[metric], label="Fixed Clipping (0.1,1.0,10)")
        plt.plot(df_fixed_n_5.index, df_fixed_n_5[metric], label="Fixed Clipping (0.1,5.0,10)")
        plt.plot(df_adaptative.index, df_adaptative[metric], label="Adaptative Clipping (0.1,10)")
        plt.title(f"Evolução da {metric.split('.')[-1]}")
        plt.xlabel("Rodada (Epoch/Round)")
        plt.ylabel(metric.split('.')[-1])
        plt.legend()
        plt.grid(True)

plt.tight_layout()
plt.savefig("f1_accuracy_loss.png")
plt.show()


