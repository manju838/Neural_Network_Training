import numpy as np

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1/(1+np.exp(-inputs))

class Activation_Step:
    def forward(self, inputs):
        if(inputs>=0):
            self.output = 1
        else:
            self.output = 0        

class Activation_Signum:
    def forward(self, inputs):
        if(inputs>0):
            self.output = 1
        elif(inputs == 0):
            self.output = 0.5
        else:
            self.output = 0

class Activation_Linear:
    def forward(self, inputs):
        self.output = inputs

class Activation_Hyperbolic_Tan:
    def forward(self, inputs):
        self.output = np.tanh(inputs)

class Activation_LeakyReLU:
    def forward(self, inputs):
        self.output = np.maximum(0.1*inputs, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        self.output = np.exp(inputs)/np.exp(inputs).sum()

class Activation_Swish:
    def forward(self, inputs):
        self.output = inputs/(1 + np.exp(-inputs))
