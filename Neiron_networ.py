# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:12:39 2020

@author: Grik
"""
def signoid(x):
    return 1/(1*np.exp(-x))
def signoid_derivative(x):
    return x*(1-x)
training_inputs=np.array([[0,0,1],
                          [1,1,1],
                          [1,0,1],
                          [0,1,1]])
training_outputs=np.array([[0,1,1,0]]).T
np.random.seed(1)
synaptic_weights=2*np.random.random((3,1))-1
print("Random starting synaptic weights: ")
print(synaptic_weights)
for iteration in range(1):
  input_layer=training_inputs
  outputs=signoid(np.dot(input_layer,synaptic_weights))
  error=training_outputs - outputs
  adjusments=error * signoid_derivative(outputs)
  synaptic_weights=np.dot(input_layer.T,adjusments)
print("Synaptic weights after training: ")
print(synaptic_weights)
print("Outputs after training: ")
print(outputs)
