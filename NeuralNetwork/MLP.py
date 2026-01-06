import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def int_to_one_hot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i in range(y.shape[0]):
        ary[i, y[i]] = 1
    return ary

class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        self.num_features = num_features
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        y_onehot = int_to_one_hot(y, self.num_classes)

        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1.0 - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_loss__dw_out = np.dot(delta_out.T, a_h)
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_loss__a_h = np.dot(delta_out, self.weight_out)
        d_a_h__d_z_h = a_h * (1.0 - a_h)

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, x)
        d_loss__d_b_h = np.sum(d_loss__a_h * d_a_h__d_z_h, axis=0)

        return (d_loss__dw_out , d_loss__db_out , d_loss__d_w_h , d_loss__d_b_h)
    
    def predict(self, x):
     _, a_out = self.forward(x)
     return np.argmax(a_out, axis=1)

def minibatch_generator(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]

def train(model, X, y, epochs=10, lr=0.1, batch_size=100):

    for epoch in range(epochs):

        minibatches = minibatch_generator(X, y, batch_size)

        for X_batch, y_batch in minibatches:

            # 1. forward
            a_h, a_out = model.forward(X_batch)

            # 2. backward (gradients)
            dW_out, db_out, dW_h, db_h = model.backward(
                X_batch, a_h, a_out, y_batch
            )

            # 3. gradient descent (LEARNING)
            model.weight_h -= lr * dW_h
            model.bias_h   -= lr * db_h
            model.weight_out -= lr * dW_out
            model.bias_out   -= lr * db_out

        # ---- Epoch evaluation ----
        _, probas = model.forward(X)
        preds = np.argmax(probas, axis=1)

        loss = mse_loss(y, probas, model.num_classes)
        acc = accuracy(y, preds)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")

def mse_loss(y, probas, num_classes):
    y_onehot = int_to_one_hot(y, num_classes)
    return np.mean((y_onehot - probas) ** 2)

def accuracy(y, preds):
    return np.mean(y == preds)

# 

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X.values / 255.0
y = y.astype(int).values
def getXy():
    return X, y
model = NeuralNetMLP(num_features=784,num_hidden=50,num_classes=10)
train(model, X, y, epochs=10, lr=0.1, batch_size=100)
np.savez(
"mlp_mnist_model.npz",
weight_h=model.weight_h,
bias_h=model.bias_h,
weight_out=model.weight_out,
bias_out=model.bias_out
)

   