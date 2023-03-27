import numpy as np
from nn import NeuralNet

class Agent:
    def __init__(self, shape, mut_rate=.5) -> None:
        self.__brain = NeuralNet(shape)

        self.__score = 0

        self.mut_rate = mut_rate


    @staticmethod
    def crossover(agentA, agentB):
        child = Agent(agentA.__brain.shape())

        weights_a, weights_b = agentA.__brain.weights(), agentB.__brain.weights()
        biases_a, biases_b = agentA.__brain.biases(), agentB.__brain.biases()

        for i, w in enumerate(child.__brain.weights()):
            bound = np.random.uniform(high=w.size)

            ind_arr = np.arange(w.size).reshape(w.shape)
            w = np.where(ind_arr < bound, weights_a[i], weights_b[i])

            child.__brain.set_weights(i, w)

        for i, b in enumerate(child.__brain.biases()):
            bound = int(np.random.uniform(high=b.size))

            b[:bound] = biases_a[i][:bound]
            b[bound:] = biases_b[i][bound:]

            child.__brain.set_biases(i, b)

        return child


    def mutate(self, lr, low=-1, high=1):
        
        for i, w in enumerate(self.__brain.weights()):
            prob = np.random.uniform(size=w.size).reshape(w.shape)
            mut = np.random.uniform(low, high, size=w.size).reshape(w.shape)
            w += lr*np.where(prob < self.mut_rate, mut, 0)

            self.__brain.set_weights(i, w)

        for i, b in enumerate(self.__brain.biases()):
            prob = np.random.uniform(size=b.size).reshape(b.shape)
            mut = np.random.uniform(low, high, size=b.size).reshape(b.shape)
            b += lr*np.where(prob < self.mut_rate, mut, 0)

            self.__brain.set_biases(i, b)


    def act(self, input):
        return self.brain().feed_forward(input)

    def score(self):
        return self.__score
    
    def set_score(self, s):
        self.__score = s
    
    def brain(self):
        return self.__brain
    