from brain.neuron import Neuron
import numpy as np

class Brain:
    def __init__(self, input_size, output_size):
        self.sensory_neurons = [Neuron(input_size) for _ in range(output_size)]
        self.output_size = output_size
        self.last_output = np.zeros(output_size)

    def forward(self, inputs):
        activations = np.array([neuron.activate(inputs) for neuron in self.sensory_neurons])
        self.last_output = activations
        return np.argmax(activations)

    def learn(self, inputs):
        for neuron in self.sensory_neurons:
            neuron.hebbian_update(inputs)

    def get_output_vector(self):
        return self.last_output