import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuron:
    def __init__(self, weights=None, bias=0.0):
        self.weights = weights if weights is not None else []
        self.bias = bias

    # Function for feed forward 
    def feed_forward(self, inputs):
        z = self.weights[0] * inputs[0] + self.weights[1] * inputs[1] + self.bias
        return sigmoid(z)

def main():
    # Hard code neurons
    hidden1 = Neuron(weights=[0.2, 0.4], bias=0.1)
    hidden2 = Neuron(weights=[-0.3, 0.3], bias=-0.1)

    output1 = Neuron(weights=[0.3, 0.5], bias=-0.2)
    output2 = Neuron(weights=[-0.2, -0.4], bias=0.3)

    # Print initial weights and biases 
    print("Initial Weights and Biases:")
    print(f"Hidden1 Weights: {hidden1.weights}, Bias: {hidden1.bias}")
    print(f"Hidden2 Weights: {hidden2.weights}, Bias: {hidden2.bias}")
    print(f"Output1 Weights: {output1.weights}, Bias: {output1.bias}")
    print(f"Output2 Weights: {output2.weights}, Bias: {output2.bias}")

    # Now test based on 1 1 input
    inputs = [(1, 1)]
    for x1, x2 in inputs:
        h1_output = hidden1.feed_forward([x1, x2])
        h2_output = hidden2.feed_forward([x1, x2])

        final_output1 = round(output1.feed_forward([h1_output, h2_output]), 2)
        final_output2 = round(output2.feed_forward([h1_output, h2_output]), 2)

        print(f"Input: ({x1}, {x2}) -> Output(true neuron, false neuron): ({final_output1}, {final_output2})")


if __name__ == "__main__":
    main()
