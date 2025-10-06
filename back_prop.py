import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neuron:
    def __init__(self, weights=None, bias=0.0, output=0.0, error=0.0):
        self.weights = weights if weights is not None else []
        self.bias = bias
        self.output = output    # keep track of that neurons output
        self.error = error      # use for back_prop for hidden layer
        self.inputs = []        # store input for backprop

    # Function for feed forward 
    def feed_forward(self, inputs):
        self.inputs = inputs
        z = 0
        for index in range(len(self.weights)):
            z += self.weights[index] * inputs[index]
        z += self.bias
        self.output = sigmoid(z)
        return self.output

    # Function for back propagation on output layer
    def back_prop_output(self, lrate, target):
        self.error = self.output * (1-self.output) * (target-self.output)

        # update bias 
        self.bias += lrate * self.error

        # update weights
        for index in range(len(self.weights)):
            delta = lrate * self.error * self.inputs[index]
            self.weights[index] += delta

        return self.error
    
    # Function for back propagation on hidden layer
    def back_prop_hidden(self, lrate, next_layer, index):
        # summation of the error derativates for the output nodes
        sum_error = 0
        for neuron in next_layer:
            sum_error += neuron.weights[index] * neuron.error

        # get neuron error
        self.error = self.output * (1-self.output) * sum_error

        # update bias
        self.bias += lrate * self.error

        # update weights
        for index in range(len(self.weights)):
            delta = lrate * self.error * self.inputs[index]
            self.weights[index] += delta

        return self.error

def main():
    # Training set 
    train_data = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0)]
    test_data = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 0)]
    learning_rate = 0.1
    epochs = 10000

    # Hard code neurons
    hidden1 = Neuron(weights=[0.2, 0.4], bias=0.1)
    hidden2 = Neuron(weights=[-0.3, 0.3], bias=-0.1)

    true_output = Neuron(weights=[0.3, 0.5], bias=-0.2)
    false_output = Neuron(weights=[-0.2, -0.4], bias=0.3)

    outputs = [true_output, false_output]

    # Print initial weights and biases 
    # print("Initial Weights and Biases:")
    # print(f"Hidden1 Weights: {hidden1.weights}, Bias: {hidden1.bias}")
    # print(f"Hidden2 Weights: {hidden2.weights}, Bias: {hidden2.bias}")
    # print(f"True Neuron Weights: {true_output.weights}, Bias: {true_output.bias}")
    # print(f"False Neuron Weights: {false_output.weights}, Bias: {false_output.bias}")

    # Now test based on data set
    for epoch in range(epochs):
        total_error = 0
        for x1, x2, truth in train_data:
            # forward feed
            h1 = hidden1.feed_forward([x1, x2])
            h2 = hidden2.feed_forward([x1, x2])

            y_true = true_output.feed_forward([h1, h2])
            y_false = false_output.feed_forward([h1, h2])

            # Calculate the error rate
            total_error += (truth - y_true) ** 2
            total_error += ((1 - truth) - y_false) ** 2

            # now back prop
            true_output.back_prop_output(learning_rate, truth)
            false_output.back_prop_output(learning_rate, 1 - truth)

            hidden1.back_prop_hidden(learning_rate, outputs, 0)
            hidden2.back_prop_hidden(learning_rate, outputs, 1)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Error: {total_error:.2f}")

    # Final test
    print("\nFinal Outputs: ")
    for x1, x2, truth in test_data:
        h1 = hidden1.feed_forward([x1, x2])
        h2 = hidden2.feed_forward([x1, x2])
        y_true = true_output.feed_forward([h1, h2])
        y_false = false_output.feed_forward([h1, h2])
        print(f"Input: ({x1},{x2}) -> True: {y_true:.3f}, False: {y_false:.3f}")

    # print("\n Final Weights:")
    # print(f"Hidden1 Weights: {hidden1.weights}, Bias: {hidden1.bias}")
    # print(f"Hidden2 Weights: {hidden2.weights}, Bias: {hidden2.bias}")
    # print(f"True Neuron Weights: {true_output.weights}, Bias: {true_output.bias}")
    # print(f"False Neuron Weights: {false_output.weights}, Bias: {false_output.bias}")


if __name__ == "__main__":
    main()

# MAKE EPOCHS = 1, make training data just (1, 1, 0) and we get the initial hard coded output:
# Hidden1 Weights: [0.1904080217093384, 0.3904080217093384], Bias: 0.090
# Hidden2 Weights: [-0.3256565590948485, 0.2743434409051515], Bias: -0.13
# # True Neuron Weights: [0.20789535177163876, 0.43452196124985154], Bias: -0.34
# False Neuron Weights: [-0.11550089351485172, -0.33992881059520985], Bias: 0.43
# Which is about what we expected 

