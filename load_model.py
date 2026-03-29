from pysr import PySRRegressor

model = PySRRegressor.from_file(run_directory="best_model/20260329_114046_lpt0Yb/")

print(model)