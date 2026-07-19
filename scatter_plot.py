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

columns = ["Speed", "DraftBow", "DraftStern", "BEAUFORT"]

X_train.columns = columns
X_val.columns = columns
X_test.columns = columns

# Dataset completo
X_full = pd.concat([X_train, X_val, X_test],
                   ignore_index=True)

X_full = X_full.astype(float)

# ======================
# Pair Plot
# ======================

sns.pairplot(
    X_full,
    diag_kind="hist",
    plot_kws={"s":20, "alpha":0.7}
)

plt.savefig("pairplot.png", dpi=300, bbox_inches="tight")
plt.show()