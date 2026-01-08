
# Neural Networks


---

## What I Built

- A fully working **Multi-Layer Perceptron (MLP)** from scratch
- No PyTorch / TensorFlow used
- Only NumPy for math
- Trained on the **MNIST digit dataset**
- Achieved ~90% accuracy

---

## Core Concepts I Learned

### 1. How Images Become Numbers
- An image is not “seen” by a neural network
- MNIST images are **28 × 28 grayscale pixels**
- Each pixel is a number (0–255)
- Images are flattened into a **784-length vector**

Dataset shape:
```text
[n_examples, n_features]
→ [number of images, 784]
````

---

### 2. One-Hot Encoding

* Neural networks output **one value per class**
* Labels must match this output shape

Example:

* Possible digits: `0–4`
* True digit: `3`

One-hot encoding:

```text
[0, 0, 0, 1, 0]
```

Why it’s needed:

* Allows loss to be computed neuron-by-neuron
* Aligns target labels with output layer

---

### 3. Forward Propagation

* Input → hidden layer → output layer
* Each neuron computes:
  **weighted sum + bias**

```text
z = (weights · inputs) + bias
```

* Sigmoid activation is applied
* Output layer produces values in `[0, 1]`

This step only **predicts**, it does not learn.

---

### 4. Loss Function

* Used **Mean Squared Error (MSE)** for simplicity
* Measures how far predictions are from true labels
* Loss is averaged over all samples in a batch

Loss answers:

> “How wrong is the model right now?”

---

### 5. Backpropagation

* Computes **how much each weight and bias contributed to the loss**
* Uses the **chain rule**
* Error flows backward:
  output → hidden → input

Produces gradients:

```text
∂Loss / ∂weights
∂Loss / ∂biases
```

Important:

* Backpropagation **does not update weights**
* It only calculates gradients (blame assignment)

---

### 6. Gradient Descent (Actual Learning)

* Uses gradients to update parameters

Update rule:

```text
weight = weight − learning_rate × gradient
```

* Moves weights in the direction that **reduces loss**
* Learning happens **inside the training loop**

---

### 7. Training Loop

For each epoch:

1. Split data into mini-batches
2. Forward pass (prediction)
3. Backward pass (gradient computation)
4. Update weights (gradient descent)

**Epoch** = one full pass over the training dataset.

---

### 8. Mini-Batch Training

* Training is done on small batches (e.g. 100 samples)
* More memory-efficient
* More stable learning
* Matches real-world training setups

---

### 9. Prediction / Inference

* After training, only **forward pass** is used
* No backpropagation during prediction
* Output neuron with highest value = predicted class

---

### 10. Saving and Loading the Model

* Model weights are NumPy arrays
* Saved using `np.savez`
* Allows reuse without retraining

---

## Why This Project Exists

* To deeply understand neural networks
* To remove the “black box” feeling
* To prepare for learning **PyTorch** properly
* To build intuition before moving to LLMs

---

## Next Steps (Planned)

* Reimplement this model in **PyTorch**
* Replace sigmoid + MSE with softmax + cross-entropy
* Learn CNNs for better image understanding
* Move towards LLM internals and systems

---

## Note

This project is intentionally simple and educational.
The focus is on **clarity**, not optimization or production readiness.

```
```
