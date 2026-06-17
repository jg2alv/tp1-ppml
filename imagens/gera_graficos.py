import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Caminhos dos CSVs
baseline_path = os.path.join("..", "metricas", "baseline.csv")
fixedclipping_norm_1_path = os.path.join("..", "metricas", "fixedclipping_(0.1,1.0,10).csv")
fixedclipping_norm_5_path = os.path.join("..", "metricas", "fixedclipping_(0.1,5.0,10).csv")
adaptativeclipping_path = os.path.join("..", "metricas", "adaptativeclipping_(0.1,10).csv")

csv_paths = [baseline_path, fixedclipping_norm_1_path, fixedclipping_norm_5_path, adaptativeclipping_path]
for csv_path in csv_paths:
    if not os.path.exists(csv_path):
        print(csv_path, " não encontrado.")
        raise SystemExit

# Rótulos e parâmetros de DP para cada estratégia
LABELS = [
    "Baseline (Sem DP)",
    "Fixed Clipping (0.1,1.0,10)",
    "Fixed Clipping (0.1,5.0,10)",
    "Adaptative Clipping (0.1,10)",
]
DP_PARAMS = {
    "Baseline (Sem DP)": {"noise_multiplier": 0.0, "clipping_norm": None},
    "Fixed Clipping (0.1,1.0,10)": {"noise_multiplier": 0.1, "clipping_norm": 1.0},
    "Fixed Clipping (0.1,5.0,10)": {"noise_multiplier": 0.1, "clipping_norm": 5.0},
    "Adaptative Clipping (0.1,10)": {"noise_multiplier": 0.1, "clipping_norm": None},
}

OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)
_trapz = getattr(np, "trapezoid", None) or np.trapz

METRICAS = ["metrics.accuracy", "metrics.f1", "metrics.LOSS"]
CORES = dict(zip(LABELS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

# Carrega os dados brutos de cada CSV
def carregar_raw(csv_path):
    df = pd.read_csv(csv_path)
    df["values"] = pd.to_numeric(df["values"], errors="coerce")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["validation"] = df["validation"].astype(str).str.lower() == "true"
    return df


# Apenas modelo global
def metricas_globais(df_raw):
    df_global = df_raw[(df_raw["user"] == "FL-Global") & (df_raw["validation"])].copy()
    df_global = df_global.sort_values(by="epoch")
    return df_global.pivot_table(index="epoch", columns="metric", values="values", aggfunc="mean")

# Apenas clientes (sem o global)
def metricas_clientes(df_raw):
    return df_raw[df_raw["user"] != "FL-Global"].copy().sort_values(by="epoch")


raw = {label: carregar_raw(path) for label, path in zip(LABELS, csv_paths)}
df_global = {label: metricas_globais(raw[label]) for label in LABELS}
df_clients = {label: metricas_clientes(raw[label]) for label in LABELS}

# Evolução das métricas globais
def plot_f1_accuracy_loss():
    fig = plt.figure(figsize=(15, 6.5)) 
    
    linhas = []
    for i, metric in enumerate(METRICAS, 1):
        if all(metric in df_global[label].columns for label in LABELS):
            ax = plt.subplot(1, 3, i)
            for label in LABELS:
                ln = plt.plot(df_global[label].index, df_global[label][metric], label=label, color=CORES[label])
                if i == 1: 
                    linhas.append(ln[0])
            
            plt.title(f"Evolução da {metric.split('.')[-1]}")
            plt.xlabel("Rodada (Epoch/Round)")
            plt.ylabel(metric.split(".")[-1])
            plt.grid(True)

    fig.legend(handles=linhas, labels=LABELS, loc="lower center", 
               ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=12)

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "f1_accuracy_loss.png"))
    plt.show()

# Evolução da diferença pra a baseline
def plot_delta_vs_baseline():
    baseline_label = LABELS[0]
    dp_labels = LABELS[1:]

    fig = plt.figure(figsize=(15, 6.5))
    
    linhas = []
    for i, metric in enumerate(METRICAS, 1):
        plt.subplot(1, 3, i)
        if metric not in df_global[baseline_label].columns:
            continue
        base_serie = df_global[baseline_label][metric]
        for label in dp_labels:
            if metric not in df_global[label].columns:
                continue
            serie = df_global[label][metric]
            comuns = base_serie.index.intersection(serie.index)
            delta = serie.loc[comuns] - base_serie.loc[comuns]
            ln = plt.plot(comuns, delta, label=label, color=CORES[label])
            if i == 1:
                linhas.append(ln[0])
                
        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.title(f"Delta de {metric.split('.')[-1]} vs Baseline")
        plt.xlabel("Rodada (Epoch/Round)")
        plt.ylabel(f"Δ {metric.split('.')[-1]}")
        plt.grid(True)

    fig.legend(handles=linhas, labels=dp_labels, loc="lower center", 
               ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=12)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "delta_vs_baseline.png"))
    plt.show()


# Métricas globais finais (média das últimas N rodadas) para cada estratégia, comparando entre si e com a baseline.
def plot_metricas_finais(ultimas_n=5):
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(METRICAS, 1):
        plt.subplot(1, 3, i)
        valores = []
        for label in LABELS:
            if metric in df_global[label].columns:
                serie = df_global[label][metric].dropna()
                valores.append(serie.tail(ultimas_n).mean())
            else:
                valores.append(np.nan)
        barras = plt.bar(LABELS, valores, color=[CORES[l] for l in LABELS])
        plt.title(f"{metric.split('.')[-1]} final (média das últimas {ultimas_n} rodadas)")
        plt.ylabel(metric.split(".")[-1])
        plt.xticks(rotation=20, ha="right")
        for barra, valor in zip(barras, valores):
            if not pd.isna(valor):
                plt.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:.3f}",
                          ha="center", va="bottom", fontsize=8)
        plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metricas_finais.png"))
    plt.show()


# Rodada em que cada estratégia atinge um limiar específico pela primeira vez, para uma métrica global (ex: acurácia >= 0.65). Se não atingir, marcar como "não atingiu".
def plot_rodadas_ate_threshold(metric="metrics.accuracy", limiar=0.65):
    rodadas = []
    for label in LABELS:
        serie = df_global[label].get(metric)
        if serie is None:
            rodadas.append(np.nan)
            continue
        atingida = serie[serie >= limiar]
        rodadas.append(atingida.index.min() if not atingida.empty else np.nan)

    plt.figure(figsize=(8, 5))
    barras = plt.bar(LABELS, [0 if pd.isna(r) else r for r in rodadas],
                      color=[CORES[l] for l in LABELS])
    plt.title(f"Rodadas até {metric.split('.')[-1]} >= {limiar}")
    plt.ylabel("Rodada (Epoch/Round)")
    plt.xticks(rotation=20, ha="right")
    for barra, valor in zip(barras, rodadas):
        texto = "não atingiu" if pd.isna(valor) else f"{int(valor)}"
        altura = 0 if pd.isna(valor) else valor
        plt.text(barra.get_x() + barra.get_width() / 2, altura, texto,
                  ha="center", va="bottom", fontsize=9)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rodadas_para_limiar.png"))
    plt.show()


# Gráfico mais importante da disciplina: Utilidade x Privacidade
def plot_tradeoff_privacidade(metric="metrics.accuracy", ultimas_n=5):
    pontos_x, pontos_y, pontos_label = [], [], []
    referencias = {}

    for label in LABELS:
        serie = df_global[label].get(metric)
        if serie is None:
            continue
        valor_final = serie.dropna().tail(ultimas_n).mean()
        clip = DP_PARAMS[label]["clipping_norm"]
        if clip is None:
            referencias[label] = valor_final
        else:
            pontos_x.append(clip)
            pontos_y.append(valor_final)
            pontos_label.append(label)

    plt.figure(figsize=(8, 6))
    if pontos_x:
        plt.scatter(pontos_x, pontos_y, s=120, color="tab:blue", zorder=3)
        for x, y, label in zip(pontos_x, pontos_y, pontos_label):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=8)
        if len(pontos_x) > 1:
            ordenado = sorted(zip(pontos_x, pontos_y))
            plt.plot(*zip(*ordenado), linestyle="--", color="tab:blue", alpha=0.5)

    cores_ref = ["tab:gray", "tab:green", "tab:red", "tab:purple"]
    for (label, valor), cor in zip(referencias.items(), cores_ref):
        plt.axhline(valor, linestyle=":", color=cor, label=f"{label} (ref.)")

    plt.title(f"Trade-off privacidade x utilidade ({metric.split('.')[-1]})")
    plt.xlabel("Clipping norm (fixed clipping)")
    plt.ylabel(f"{metric.split('.')[-1]} final (média últimas {ultimas_n} rodadas)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tradeoff_privacidade.png"))
    plt.show()


# Boxplot de dispersão em relação à métrica global, para cada estratégia.
def plot_dispersao_clientes(metric="metrics.accuracy"):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 5), sharey=True)
    for ax, label in zip(axes, LABELS):
        df_c = df_clients[label]
        df_c_metric = df_c[(df_c["metric"] == metric) & (df_c["validation"])]
        if df_c_metric.empty:
            ax.set_title(f"{label}\n(sem dados de cliente)")
            ax.set_xticks([])
            continue
        ultima_rodada = df_c_metric["epoch"].max()
        valores = df_c_metric.loc[df_c_metric["epoch"] == ultima_rodada, "values"]
        ax.boxplot(valores, vert=True, widths=0.5)
        jitter = np.random.uniform(-0.05, 0.05, size=len(valores))
        ax.scatter(np.ones(len(valores)) + jitter, valores, alpha=0.5, s=20, color="black")
        ax.set_title(f"{label}\n(rodada {int(ultima_rodada)}, n={len(valores)} clientes)")
        ax.set_xticks([])
    axes[0].set_ylabel(metric.split(".")[-1])
    plt.suptitle(f"Dispersão de {metric.split('.')[-1]} entre clientes, por estratégia")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dispersao_clientes.png"))
    plt.show()


plot_f1_accuracy_loss()
plot_delta_vs_baseline()
plot_metricas_finais()
plot_rodadas_ate_threshold()
plot_tradeoff_privacidade()
plot_dispersao_clientes()