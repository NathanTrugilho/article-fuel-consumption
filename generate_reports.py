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
    run_directory='sr/best_model/20260415_070252_EbFzWd'
)

ann_model = joblib.load("ann/best_model/model.joblib")
rf_model  = joblib.load("rf/best_model/model.joblib")
svm_model = joblib.load("svm/best_model/model.joblib")

models = {
    "SR": sr_model,
    "ANN": ann_model,
    "RF": rf_model,
    "SVM": svm_model
}

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
with open("metrics.txt", "w") as f:
    for name, model in models.items():

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_metrics = compute_metrics(y_val, y_val_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        f.write(f"{name} - Validation:\n")
        f.write(f"MSE: {val_metrics[0]} RMSE: {val_metrics[1]} MAE: {val_metrics[2]} R2: {val_metrics[3]}\n")

        f.write(f"{name} - Test:\n")
        f.write(f"MSE: {test_metrics[0]} RMSE: {test_metrics[1]} MAE: {test_metrics[2]} R2: {test_metrics[3]}\n\n")

# ======================
# Função de plot (scatter ordenado)
# ======================
def plot_sorted_scatter(ax, y_true, y_pred, title):
    idx = np.argsort(y_true)
    y_true_sorted = y_true.iloc[idx].values
    y_pred_sorted = y_pred[idx]

    ax.scatter(range(len(y_true_sorted)), y_true_sorted, label="Real")
    ax.scatter(range(len(y_pred_sorted)), y_pred_sorted, label="Predito")
    ax.set_title(title)
    ax.set_xlabel("Amostras (ordenadas)")
    ax.set_ylabel("Valor")
    ax.legend()

# ======================
# Figura 1: Validação
# ======================
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_val)
    plot_sorted_scatter(axes[i], y_val, y_pred, f"{name} - Validation")

plt.tight_layout()
plt.show()

# ======================
# Figura 2: Teste
# ======================
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    plot_sorted_scatter(axes[i], y_test, y_pred, f"{name} - Test")

plt.tight_layout()
plt.show()