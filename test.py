import numpy as np
from pysr import PySRRegressor

x = np.array([1,2,3,4,5,6]).reshape(-1,1)
y = np.array([2,4,6,8,10,12])

model = PySRRegressor()
model.fit(x, y)

print(model)