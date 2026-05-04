import joblib

model = joblib.load("rf/best_model/model.joblib")

print(model)