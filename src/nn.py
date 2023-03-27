import numpy as np

class NeuralNet:
    def __init__(self, shape, bias_fact=1) -> None:
        self.__shape = shape
        self.__num_layers = len(self.__shape)
        self.__biases = [np.random.uniform(size=d)*bias_fact for d in self.__shape]
        self.__weights = [np.random.uniform(
                          size=(self.__shape[i], self.__shape[i-1])) for i in range(1, len(self.__shape))
                         ]
        
    @staticmethod
    def sigmoid_arr(arr):
        return 1 / (1 + np.exp(-arr))
    
    @staticmethod
    def reLU(arr):
        return (np.maximum(0, arr))


    def feed_forward(self, input):
        output = input

        for i in range(self.__num_layers-1):
            output = np.matmul(self.__weights[i], output) + self.__biases[i+1][:, None]
            
            output = NeuralNet.reLU(output)

        return output
    

    def weights(self):
        return self.__weights
    
    def set_weights(self, i, w):
        self.__weights[i] = w

    def biases(self):
        return self.__biases
    
    def set_biases(self, i, b):
        self.__biases[i] = b
    
    def size(self):
        return self.__num_layers
    
    def shape(self):
        return self.__shape