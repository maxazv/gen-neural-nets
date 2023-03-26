import numpy as np
from nn import NeuralNet

class Agent:
    def __init__(self, shape) -> None:
        self.__brain = NeuralNet(shape)
        self.__score = 0


    @staticmethod
    def crossover(agentA, agentB, ratio=.5):
        child = Agent(agentA.__brain.shape())

        weights_a, weights_b = agentA.__brain.weights(), agentB.__brain.weights()
        biases_a, biases_b = agentA.__brain.biases(), agentB.__brain.biases()

        for i, w in enumerate(child.__brain.weights()):
            bound = np.random.uniform(high=w.size)
            tmp = np.zeros(w.shape)

            for k in range(w.shape[0]):
                for l in range(w.shape[1]):

                    if k * w.shape[1] + l < bound:
                        tmp[k][l] = weights_a[i][k][l]
                    else:
                        tmp[k][l] = weights_b[i][k][l]

            child.__brain.set_weights(i, tmp)

        for i, b in enumerate(child.__brain.biases()):
            bound = int(np.random.uniform(high=b.size))

            b[:bound] = biases_a[i][:bound]
            b[bound:] = biases_b[i][bound:]

            child.__brain.set_biases(i, b)

        return child


    # TODO implement basic mutation
    def mutate(self, rate, freq):
        for w in self.__brain.weights():
            pass

        for b in self.__brain.biases():
            pass


    def act(self, input):
        return self.brain().feed_forward(input)

    def score(self):
        return self.__score
    
    def set_score(self, s):
        self.__score = s
    
    def brain(self):
        return self.__brain
    