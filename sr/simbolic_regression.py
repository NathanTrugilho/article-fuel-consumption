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

X_train = train_df.iloc[1:, 5:9]
y_train = train_df.iloc[1:, -1]

X_val   = val_df.iloc[1:, 5:9]
X_test  = test_df.iloc[1:, 5:9]

y_val   = val_df.iloc[1:, -1]
y_test  = test_df.iloc[1:, -1]

# ======================
# Modelo base (grid)
# ======================
base_model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sin", "cos", "exp", "log", "sinh", "cosh", "erf"],
    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
    constraints={'^': (-1, 1)},
    verbosity=False,
    annealing=True,
    turbo=True,
    warm_start=False,
    parallelism='multithreading',
)

# ======================
# Grid
# ======================
param_grid = {
    "populations": [50, 100, 200],
    "population_size": [50, 100, 200],
    "maxsize": [20, 30, 40],
    "parsimony": [1e-5, 1e-4, 1e-3],
}

# ======================
# K-Fold
# ======================
cv = KFold(n_splits=5, shuffle=True, random_state=28)

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv,
    verbose=1,
    scoring="neg_mean_squared_error",
)

# ======================
# Fase 1: Grid
# ======================
grid.fit(X_train, y_train)

best_params = grid.best_params_

print("\nMelhores hiperparâmetros (grid):")
print(best_params)

# ======================
# Fase 2: Modelo final (recriado)
# ======================
final_model = PySRRegressor(
    niterations=100000,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sin", "cos", "exp", "log", "sinh", "cosh", "erf"],
    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
    constraints={'^': (-1, 1)},
    verbosity=True,
    annealing=True,
    turbo=True,
    warm_start=False,
    output_directory="best_model",
    parallelism='multithreading',
    **best_params
)

final_model.fit(X_train, y_train)

# ======================
# Métricas
# ======================
def print_metrics(y_true, y_pred, name):
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
y_val_pred = final_model.predict(X_val)
print_metrics(y_val, y_val_pred, "Validation")

y_test_pred = final_model.predict(X_test)
print_metrics(y_test, y_test_pred, "Test")

# ======================
# Resultado final
# ======================
print("\nEquação encontrada:")
print(final_model)