import pandas as pd

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
columns = ["Speed", "DraftBow", "DraftStern", "BEAUFORT"]

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
# Estatísticas Descritivas
# ======================

statistics = X_full.describe().T

# Adicionando variância
statistics["Variance"] = X_full.var()

# Reorganizando as colunas
statistics = statistics[
    [
        "count",
        "mean",
        "std",
        "Variance",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
]

# Arredondar os valores
statistics = statistics.round(4)

# Mostrar na tela
print("\nEstatísticas Descritivas:\n")
print(statistics)

# Salvar em csv
statistics.to_csv(
    "descriptive_statistics.csv",
    index=True
)