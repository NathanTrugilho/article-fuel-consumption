import pandas as pd
import numpy as np
import os
import joblib

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ======================
# Carregar datasets
# ======================
train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

# Features
X_train = train_df.iloc[1:, 5:9]
y_train = train_df.iloc[1:, -1]

X_val   = val_df.iloc[1:, 5:9]
X_test  = test_df.iloc[1:, 5:9]

# Target
y_val   = val_df.iloc[1:, -1]
y_test  = test_df.iloc[1:, -1]

# ======================
# Pipeline (Scaler + SVR)
# ======================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR())
])

# ======================
# Grid (moderado)
# ======================
param_grid = {
    "svr__kernel": ["rbf", "linear"],
    "svr__C": [0.1, 1, 10, 100],
    "svr__epsilon": [0.01, 0.1, 0.5],
    "svr__gamma": ["scale", "auto"]
}

# ======================
# K-Fold
# ======================
cv = KFold(n_splits=5, shuffle=True, random_state=28)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    verbose=1,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

# Treinamento
grid.fit(X_train, y_train)

# Melhor modelo
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

# ======================
# Salvar modelo
# ======================
os.makedirs("svm/best_model", exist_ok=True)
joblib.dump(best_model, "svm/best_model/model.joblib")

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
y_val_pred = best_model.predict(X_val)
print_metrics(y_val, y_val_pred, "Validation")

y_test_pred = best_model.predict(X_test)
print_metrics(y_test, y_test_pred, "Test")

# ======================
# Resultados finais
# ======================
print("\nMelhores hiperparâmetros:")
print(grid.best_params_)