import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# Carregar datasets
# ======================
train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

X_val = val_df.iloc[1:, 5:9]
y_val = val_df.iloc[1:, -1]

X_test = test_df.iloc[1:, 5:9]
y_test = test_df.iloc[1:, -1]

# ======================
# Carregar modelos
# ======================
sr_model = PySRRegressor.from_file(
    run_directory="sr/best_model/20260415_070252_EbFzWd"
)

ann_model = joblib.load("ann/best_model/model.joblib")
rf_model = joblib.load("rf/best_model/model.joblib")
svm_model = joblib.load("svm/best_model/model.joblib")

models = {
    "SR": sr_model,
    "ANN": ann_model,
    "RF": rf_model,
    "SVM": svm_model,
}

# ======================
# Encontrar melhor equação do SR
# (menor MSE na validação)
# ======================
best_sr_idx = None
best_sr_mse = np.inf

for i in range(len(sr_model.equations_)):
    y_pred = sr_model.predict(X_val, index=i)
    mse = mean_squared_error(y_val, y_pred)

    if mse < best_sr_mse:
        best_sr_mse = mse
        best_sr_idx = i

print(f"Melhor equação SR: #{best_sr_idx + 1}\n")

print(f"Melhor combinação de hiperparâmetros SR:\n"
      f" Populations:{sr_model.populations},\n"
      f" Population Size:{sr_model.population_size},\n"
      f" Maxsize:{sr_model.maxsize},\n"
      f" Parsimony:{sr_model.parsimony}")

try:
    print(
        "Expressão:",
        sr_model.equations_.iloc[best_sr_idx]["equation"]
    )
except Exception:
    pass

# ======================
# Métricas
# ======================
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# ======================
# Salvar métricas
# ======================
with open("metrics.txt", "w", encoding="utf-8") as f:

    for name, model in models.items():

        if name == "SR":
            y_val_pred = model.predict(X_val, index=best_sr_idx)
            y_test_pred = model.predict(X_test, index=best_sr_idx)

            f.write(f"SR - Best Equation #{best_sr_idx + 1}\n")

            try:
                f.write(
                    f"Expression: "
                    f"{model.equations_.iloc[best_sr_idx]['equation']}\n"
                )
                f.write(f"Melhor combinação de hiperparâmetros SR:\n"
                        f" Populations:{sr_model.populations},\n"
                        f" Population Size:{sr_model.population_size},\n"
                        f" Maxsize:{sr_model.maxsize},\n"
                        f" Parsimony:{sr_model.parsimony}")
            except Exception:
                pass

        else:
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)

        val_metrics = compute_metrics(y_val, y_val_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        f.write(f"{name} - Validation:\n")
        f.write(
            f"MSE: {val_metrics[0]}\n"
            f"RMSE: {val_metrics[1]}\n"
            f"MAE: {val_metrics[2]}\n"
            f"R2: {val_metrics[3]}\n"
        )

        f.write(f"{name} - Test:\n")
        f.write(
            f"MSE: {test_metrics[0]}\n"
            f"RMSE: {test_metrics[1]}\n"
            f"MAE: {test_metrics[2]}\n"
            f"R2: {test_metrics[3]}\n\n"
        )

# ======================
# Função de plot
# ======================
def plot_sorted_scatter(ax, y_true, y_pred):
    idx = np.argsort(y_true)

    y_true_sorted = y_true.iloc[idx].values
    y_pred_sorted = np.asarray(y_pred)[idx]

    ax.scatter(
        range(len(y_true_sorted)),
        y_true_sorted,
        label="True"
    )

    ax.scatter(
        range(len(y_pred_sorted)),
        y_pred_sorted,
        label="Predicted"
    )

    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("MDO (m³/h)", fontsize=12)
    ax.legend()

# ======================
# Criar diretórios
# ======================
os.makedirs("plots", exist_ok=True)
os.makedirs("plots/validation", exist_ok=True)
os.makedirs("plots/test", exist_ok=True)

# ======================
# Salvar gráficos de validação
# ======================
for name, model in models.items():

    if name == "SR":
        y_pred = model.predict(X_val, index=best_sr_idx)
        filename = "plots/validation/SR_validation.pdf"
    else:
        y_pred = model.predict(X_val)
        filename = f"plots/validation/{name}_validation.pdf"

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_sorted_scatter(
        ax,
        y_val,
        y_pred,
    )

    plt.tight_layout()
    plt.savefig(
        filename,
        bbox_inches="tight"
    )
    plt.close(fig)

# ======================
# Salvar gráficos de teste
# ======================
for name, model in models.items():

    if name == "SR":
        y_pred = model.predict(X_test, index=best_sr_idx)
        filename = "plots/test/SR_test.pdf"
    else:
        y_pred = model.predict(X_test)
        filename = f"plots/test/{name}_test.pdf"

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_sorted_scatter(
        ax,
        y_test,
        y_pred,
    )

    plt.tight_layout()
    plt.savefig(
        filename,
        bbox_inches="tight"
    )
    plt.close(fig)

print("Relatório gerado.")
print("Métricas: metrics.txt")
print("Gráficos: plots/validation e plots/test")