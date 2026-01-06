import numpy as np
from MLP import NeuralNetMLP
from MLP import getXy
model = NeuralNetMLP(784, 50, 10)

data = np.load("mlp_mnist_model.npz")
model.weight_h = data["weight_h"]
model.bias_h = data["bias_h"]
model.weight_out = data["weight_out"]
model.bias_out = data["bias_out"]

idx = 46734
X , y = getXy()
x_input = X[idx].reshape(1, -1)
prediction = model.predict(x_input)
print(f"Predicted label: {prediction[0]}, True label: {y[idx]}")
