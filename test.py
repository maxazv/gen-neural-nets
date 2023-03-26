import numpy as np
from nn import NeuralNet
from agent import Agent

shape = (2, 9, 9, 2)
nn = NeuralNet(shape)

nn.feed_forward(np.array([[1], [2]]))

parentA = Agent(shape)
parentB = Agent(shape)

child = Agent.crossover(parentA, parentB)
#print(parentA.brain().biases()[0], '\n\n', parentB.brain().biases()[0], '\n')
#print(child.brain().biases()[0])