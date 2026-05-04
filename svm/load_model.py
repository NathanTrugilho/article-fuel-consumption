import joblib

model = joblib.load("svm/best_model/model.joblib")

print(model)