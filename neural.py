#This is the feed forward implementation of a three layered perceptron network with dynamic no.of neurons for input,hidden and output layers

import numpy as np
import math
import matplotlib.pyplot as plt

class Network_Initialiser:
    def __init__(self,input_shape,hidden_shape,output_shape,inputs):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.inputs = inputs
        self.weights_between_input_and_hidden = np.zeros(hidden_shape*input_shape, dtype=np.float64).reshape(hidden_shape, input_shape) #Total no.of weights for (m,n) shape network is m*n
        self.weights_between_hidden_and_output = np.zeros(hidden_shape*output_shape, dtype=np.float64).reshape(output_shape, hidden_shape)
        self.bias_hidden = np.ones(hidden_shape, dtype=np.float64) #No.of bias terms would be no.of neurons in bias layer
        self.bias_output = np.ones(output_shape, dtype=np.float64)
        self.summation_hidden = np.zeros(hidden_shape, dtype=np.float64) #This is the neuron output before applying activation fn.
        self.summation_output = np.zeros(output_shape, dtype=np.float64)
        #print(inputs[0])

    def sigmoid(self,x):
        #Not using this for now
        return(1 / (1 + np.exp(-x)) )
    
    def hidden_summation(self,vector1,vector2,bias_vector): #This method computes total f(weight*inputs+bias) where f is activation fn sigmoid
        #vector1 is inputs,vector2 is weight matrix of form (hidden_shape,input_shape),so first element of this matrix gives weights for each input with respect to first neuron of hidden layer
        for j in range(len(vector2)):
            self.summation_hidden[j] = np.dot(vector1,vector2[j]) + bias_vector[j]
        print("input-hidden summations,activations:")
        print(self.summation_hidden)
        print("***********")
        print(1 / (1 + np.exp(- self.summation_hidden)))
        print("---------------")
        #return(self.summation_hidden)
        return(1 / (1 + np.exp(- self.summation_hidden)) )

    def output_summation(self,vector1,vector2,bias_vector):
        for k in range(len(vector2)):
            self.summation_output[k] = np.dot(vector1,vector2[k]) + bias_vector[k]
        print("hidden-output summations,activations:")
        print(self.summation_output)
        print("***********")
        print(1 / (1 + np.exp(- self.summation_output)))
        print("---------------")        
        #return(self.summation_hidden)
        return(1 / (1 + np.exp(- self.summation_output)) )
    
    def get_weights(self):
        print("Weight Matrix:")
        print([self.weights_between_input_and_hidden, self.weights_between_hidden_and_output])
        print("---------------")
        return([self.weights_between_input_and_hidden, self.weights_between_hidden_and_output])
    
    def get_bias(self):
        print("Bias Matrix:")
        print([self.bias_hidden, self.bias_output])
        print("---------------")
        return([self.bias_hidden, self.bias_output])

    def weight_initialiser(self):
        for j in range(hidden_shape):
            for i in range(input_shape):
                self.weights_between_input_and_hidden[j][i] = math.pow(-1,(i+j)) #This order of loops gives weights for each hidden neuron with list of weights from each input neuron.So while multiplying,I can just dot product to get summation value,then add bias and apply activation fn.
        # print("Weights between Input and Hidden Layer with respect to Hidden Layer:")
        # print(self.weights_between_input_and_hidden)
        # print("-----------------")
        for k in range(output_shape):
            for j in range(hidden_shape):
                self.weights_between_hidden_and_output[k][j] = math.pow(-1,(j+k))
        # print("Weights between Hidden Layer and Output Layer with respect to Output Layer:")
        # print(self.weights_between_hidden_and_output)
            
    def bias_initialiser(self):
        #This method is not used in this implementation as I considered bias as 1 for all neurons.But if it is to be generated by fn (-1)^j,for hidden layer and (-1)^k for output neuron,then use this method
        for j in range(hidden_shape):
            self.bias_hidden[j] = math.pow(-1,j)
        for k in range(output_shape):
            self.bias_output[k] = math.pow(-1,k)
        print("Hidden Layer Bias:")
        print(self.bias_hidden)
        print("------------")
        print("Output Layer Bias:")
        print(self.bias_output)

if __name__ == "__main__":
    #Prompting the user for no.of neurons in input,hidden and output layers
    input_shape = int(input("Enter the no.of input neurons: ")) 
    hidden_shape = int(input("Enter the no.of hidden neurons: ")) 
    output_shape = int(input("Enter the no.of output neurons: ")) 
    inputs = np.zeros(input_shape) #initialising input layer with shape of no.of input neurons
    
    for i in range(input_shape):
        inputs[i] = int(input("Enter the neuron value: "))

    NI = Network_Initialiser(input_shape,hidden_shape,output_shape,inputs) #Initialising NI object of Network_Initialiser class
    NI.weight_initialiser()
    #NI.bias_initialiser()
    #print(NI.get_weights()[0],NI.get_bias()[1])
    print("Final Output:")
    print( NI.output_summation( ( NI.hidden_summation( inputs, NI.get_weights()[0], NI.get_bias()[0] )), NI.get_weights()[1], NI.get_bias()[1] ))
    print("---------------")







    
