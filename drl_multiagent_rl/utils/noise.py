
import numpy as np
import numpy.random as nr

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state





if __name__ == '__main__':
    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(2, 1)


    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    axarr[0].plot(states)
    axarr[0].set_title(' original')


    ou = OUNoise(3,mu=0,theta=0.5,sigma=0.4)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    axarr[1].plot(states)
    axarr[1].set_title(' after')

    plt.show()
