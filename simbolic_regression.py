import pandas as pd
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ======================
# Carregar datasets
# ======================
train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

# Features: colunas 6,7,8,9
X_train = train_df.iloc[1:, 5:9]
y_train = train_df.iloc[1:, -1]
#print("X_train ========= \n",X_train)
#print("y_train ========= \n",y_train)

X_val   = val_df.iloc[1:, 5:9]
X_test  = test_df.iloc[1:, 5:9]

# Target: última coluna
y_val   = val_df.iloc[1:, -1]
y_test  = test_df.iloc[1:, -1]

# ======================
# Modelo base (operadores fixos)
# ======================
base_model = PySRRegressor(
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sin", "cos", "exp", "log", "sinh", "cosh", "erf"],
    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
    constraints={'^': (-1, 1)},
    verbosity=True,
    annealing=True,
    turbo=True,
    warm_start=False,
    output_directory="grid_search_models",
    parsimony=1e-4, # Penalização por complexidade
)

# ======================
# Grid de hiperparâmetros
# ======================
param_grid = {
    "niterations": [100, 200, 400],
    "populations": [50, 100, 200],
    "population_size": [50, 100, 200],
    "maxsize": [20, 30, 40],
}

# ======================
# K-Fold no treino
# ======================
cv = KFold(n_splits=5, shuffle=True, random_state=28)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    verbose=0,
    scoring="neg_mean_squared_error"
)

# Treinar grid search (apenas treino)
grid.fit(X_train, y_train)

# Melhor modelo após CV
best_model = grid.best_estimator_
best_model.output_directory="best_model"

# ======================
# Função de métricas
# ======================
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}:")
    print(f"MSE:  {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE:  {mae}")
    print(f"R2:   {r2}")

# ======================
# Avaliação
# ======================
y_val_pred = best_model.predict(X_val)
evaluate(y_val, y_val_pred, "Validação")

y_test_pred = best_model.predict(X_test)
evaluate(y_test, y_test_pred, "Teste")

# ======================
# Expressão simbólica final
# ======================
print("\nEquação encontrada:")
print(best_model)