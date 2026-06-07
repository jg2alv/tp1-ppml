import pandas as pd
import matplotlib.pyplot as plt
import os

# Caminhos para os CSVs
baseline_path = os.path.join("..", "flautim", "baseline", "baseline.csv")
fixedclipping_path = os.path.join("..", "flautim", "serverside_fixedclipping", "fixedclipping.csv")
adaptativeclipping_path = os.path.join("..", "flautim", "serverside_adaptativeclipping", "adaptativeclipping.csv")

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





if not os.path.exists(baseline_path) or not os.path.exists(fixedclipping_path) or not os.path.exists(adaptativeclipping_path):
    print("CSV não encontrado.")


df_baseline = metricas(baseline_path)
df_fixed = metricas(fixedclipping_path)
df_adaptative = metricas(adaptativeclipping_path)

metrics_to_plot = ["metrics.accuracy", "metrics.f1", "metrics.LOSS"]

plt.figure(figsize=(15, 5))
for i, metric in enumerate(metrics_to_plot, 1):
    if metric in df_baseline.columns and metric in df_fixed.columns and metric in df_adaptative.columns:
        plt.subplot(1, 3, i)
        plt.plot(df_baseline.index, df_baseline[metric], label="Baseline")
        plt.plot(df_fixed.index, df_fixed[metric], label="Fixed Clipping (DP)")
        plt.plot(df_adaptative.index, df_adaptative[metric], label="Adaptative Clipping (DP)")
        plt.title(f"Evolução da {metric.split('.')[-1]}")
        plt.xlabel("Rodada (Epoch/Round)")
        plt.ylabel(metric.split('.')[-1])
        plt.legend()
        plt.grid(True)

plt.tight_layout()
plt.savefig("f1_accuracy_loss.png")
plt.show()


