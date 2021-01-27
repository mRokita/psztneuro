import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import csv

class MultilayerNeuralNetwork:
    class Layer:
        def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
            self.weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

    def __init__(self, first_layer, second_layer, inputs, outputs):
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def activate(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def deactivate(x):
        return x * (1 - x)

    def fit(self, epochs):
        for i in range(epochs):
            first_layer_output, second_layer_output = self.feed_forward()
            first_layer_adjustment, second_layer_adjustment = self.backprop(first_layer_output, second_layer_output)
            self.first_layer.weights += first_layer_adjustment
            self.second_layer.weights += second_layer_adjustment

    def feed_forward(self):
        first_layer_output = self.activate(np.dot(self.inputs, self.first_layer.weights))
        return first_layer_output, self.activate(np.dot(first_layer_output, self.second_layer.weights))

    def backprop(self, first_layer_output, second_layer_output):
        second_layer_error = self.outputs - second_layer_output
        second_layer_delta = second_layer_error * self.deactivate(second_layer_output)
        first_layer_error = second_layer_delta.dot(self.second_layer.weights.T)
        first_layer_delta = first_layer_error * self.deactivate(first_layer_output)
        return self.inputs.T.dot(first_layer_delta), first_layer_output.T.dot(second_layer_delta)

    def test(self, test_example):
        first_layer_output = self.activate(np.dot(test_example, self.first_layer.weights))
        second_layer_output = self.activate(np.dot(first_layer_output, self.second_layer.weights))
        return second_layer_output


def read_dataset(file_name: str):
    with open(file_name, newline='') as fo:
        reader = csv.reader(fo, delimiter=';')
        next(reader)
        return [[float(cell.replace(',', '.')) for cell in row[2:-2]] for row in reader if any(row) and not '-200' in row]


def main():
    input_indices = [1, 4, 5, 8, 9, 10, 11, 12]
    output_indices = [0, 2, 3, 5, 7]
    number_of_epochs = 1000
    for r in read_dataset('data.csv'):
        print(r)
    dataset = np.array(
        read_dataset('data.csv'))

    dataset = dataset[(dataset != -200).all(axis=1)]

    train, test = train_test_split(dataset, test_size=0.7, random_state=42)

    train_inputs, train_outputs = train[:, input_indices], train[:, output_indices]
    test_inputs, test_outputs = train[:, input_indices], train[:, output_indices]

    norm_train_inputs = train_inputs / train_inputs.max(axis=0)
    norm_train_outputs = train_outputs / train_outputs.max(axis=0)
    norm_test_inputs = test_inputs / test_inputs.max(axis=0)
    norm_test_outputs = test_outputs / test_outputs.max(axis=0)

    np.random.seed(1)

    number_of_inputs = train_inputs[0].size
    number_of_outputs = train_outputs[0].size
    neurons = int(2 / 3 * number_of_inputs + number_of_outputs)

    hidden_layer = MultilayerNeuralNetwork.Layer(neurons, number_of_inputs)
    output_layer = MultilayerNeuralNetwork.Layer(number_of_outputs, neurons)
    neural_network = MultilayerNeuralNetwork(hidden_layer, output_layer, norm_train_inputs,
                                             norm_train_outputs)

    neural_network.fit(number_of_epochs)

    test_results = neural_network.test(norm_test_inputs) * train_outputs.max(axis=0)

    print(metrics.r2_score(test_results, test_outputs))


