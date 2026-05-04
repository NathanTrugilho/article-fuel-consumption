from pysr import PySRRegressor

model = PySRRegressor.from_file(run_directory='sr/best_model/20260415_070252_EbFzWd')

print(model.population_size)
print(model.populations)
print(model.maxsize)
print(model.niterations)