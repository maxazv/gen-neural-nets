import numpy as np

class Game:
    def __init__(self, low, high, res=1/50) -> None:
        self.res = res  # 1 iteration is <res> seconds
        self.target_pos = np.random.randint(low, high, size=2)
        self.ball_vals = np.zeros((2, 2))

    def set_vel(self, v):
        self.ball_vals[:, 1] = v

    def iteration(self, f=np.array([0, -9.81])):
        self.ball_vals[:, 1] += f*self.res
        self.ball_vals[:, 0] += self.ball_vals[:, 1]

    def dist(self):
        dist = np.power(self.target_pos - self.ball_vals[:, 0], 2)
        return np.sqrt(np.sum(dist))
    
    def reset(self):
        self.ball_vals = np.zeros((2, 2))

    def get_target(self):
        return np.reshape(self.target_pos, (2, 1))