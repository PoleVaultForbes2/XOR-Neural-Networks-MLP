<!-- Markdeep README -->
<meta charset="utf-8">
<link rel="stylesheet" href="https://casual-effects.com/markdeep/latest/markdeep.min.css">


# XOR Neural Network (Python)

This project implements a **Multilayer Perceptron (MLP)** with **Backpropagation** from scratch in Python to learn the XOR logical operation — a classic problem in neural network theory.

---

## 🧠 Overview

The program builds and trains a **2-2-2 neural network** to solve the XOR truth table using:
- **Feedforward propagation** with the sigmoid activation function  
- **Backpropagation** for both hidden and output layers  
- **Manually adjustable weights and biases**  
- **Error monitoring** over training epochs  

It demonstrates fundamental neural network principles, such as non-linearity, error correction, and gradient-based learning.

---

## 📂 Files

| File | Description |
|------|--------------|
| `xor_network.py` | Main script containing the neuron class, feedforward, and backpropagation logic. |

---

## ⚙️ How to Run

To execute the program and observe training:

~~~~~~~~~~~~~~~~~~~~
python xor_network.py
~~~~~~~~~~~~~~~~~~~~

**Optional adjustments:**
- Modify `epochs` to control training duration.  
- Change `train_data` to experiment with logical input patterns.  

---

## 📊 Expected Behavior

The network is trained on the XOR dataset:

| Input (x1, x2) | Target Output |
|----------------|----------------|
| (0, 0) | 0 |
| (0, 1) | 1 |
| (1, 0) | 1 |
| (1, 1) | 0 |

After training, the **True** and **False** output neurons should converge near:  
- `True ≈ 1` for XOR = True cases  
- `False ≈ 1` for XOR = False cases  

---

## 🧩 Key Concepts

- **Feedforward**: Computes neuron activations layer by layer using weighted sums and the sigmoid function.  
- **Backpropagation**: Adjusts weights and biases by propagating errors backward from the output to the hidden layer.  
- **Learning Rate (η)**: Controls how fast weights are updated.  
- **Error Calculation**: Mean squared error (MSE) used to monitor training progress.

---

## 🧮 Example Output

After training (e.g., 10,000 epochs):

