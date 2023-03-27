import numpy as np

class Game:
    def __init__(self, low, high, iter=100, res=1/50) -> None:
        self.res = res  # 1 iteration is <res> seconds
        self.target_pos = np.random.randint(low, high, size=2)
        self.ball_vals = np.zeros((2, 2))
        self.iter = iter

    def set_vel(self, v):
        self.ball_vals[:, 1] = v.reshape((2,))

    def iteration(self, f):
        self.ball_vals[:, 1] += f*self.res
        self.ball_vals[:, 0] += self.ball_vals[:, 1]

    def play(self, input, f=np.array([0, -9.81])):
        self.set_vel(input)
        dist = float('inf')

        for _ in range(self.iter):
            self.iteration(f)
            tmp = self.dist()
            dist = tmp if (tmp < dist) else dist

        self.reset()
        return dist

    def dist(self):
        dist = np.power(self.target_pos - self.ball_vals[:, 0], 2)
        return np.sqrt(np.sum(dist))
    
    def reset(self):
        self.ball_vals = np.zeros((2, 2))

    def get_input(self, agent_inf):
        return np.reshape(self.target_pos, (2, 1))
    
    def rand_inst(bound=1000, res=1/50):
        return Game(np.random.randint(-bound, 0), np.random.randint(bound), res)