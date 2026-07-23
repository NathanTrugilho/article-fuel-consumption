import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ======================
# Carregar os datasets
# ======================

train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

# ======================
# Selecionar as variáveis
# ======================

X_train = train_df.iloc[1:, 5:9]
X_val = val_df.iloc[1:, 5:9]
X_test = test_df.iloc[1:, 5:9]

# Renomear as colunas
columns = ["Speed", "DraftBow", "DraftStern", "Beaufort"]

X_train.columns = columns
X_val.columns = columns
X_test.columns = columns

# ======================
# Criar dataset completo
# ======================

X_full = pd.concat(
    [X_train, X_val, X_test],
    ignore_index=True
)

X_full = X_full.astype(float)

# ======================
# Matriz de Correlação
# ======================

correlation_matrix = X_full.corr()

# ======================
# Heatmap
# ======================

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)

plt.tight_layout()

# Salvar figura
plt.savefig(
    "correlation_heatmap.pdf",
    bbox_inches="tight"
)

plt.show()