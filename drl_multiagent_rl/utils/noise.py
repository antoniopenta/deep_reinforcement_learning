
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

    f, axarr = plt.subplots(7, 1)


    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    axarr[0].plot(states)
    axarr[0].set_title(' original')


    ou = OUNoise(3,mu=1,theta=0.15,sigma=0.6)
    states = []
    for i in range(1000):
        states.append(0.7*ou.sample())
    axarr[1].plot(states)
    axarr[1].set_title(' after')



    ou = OUNoise(3, mu=1, theta=0.15, sigma=0.6)
    states = []
    for i in range(1000):
        states.append(0.9 * ou.sample())
    axarr[2].plot(states)
    axarr[2].set_title(' after')


    ou = OUNoise(3, mu=1, theta=0.15, sigma=0.6)
    states = []
    for i in range(1000):
        states.append(0.5 * ou.sample())
    axarr[3].plot(states)
    axarr[3].set_title(' after')



    ou = OUNoise(3, mu=1, theta=0.15, sigma=0.6)
    states = []
    for i in range(1000):
        states.append(0.05 * ou.sample())
    axarr[4].plot(states)
    axarr[4].set_title(' after')


    exploration_range = (1, 0.0)
    v =[]
    for i_episode in range(2000):
        exploration = max(0, 25000 - i_episode) / 25000
        exploration = exploration_range[1] + (exploration_range[0] - exploration_range[
                                                 1]) * exploration
        v.append(exploration)


    axarr[5].plot(v)
    axarr[5].set_title('exploration 1')
    exploration =1
    exploration_decay =0.99
    exploration_min =0.05
    v2=[]
    for i_episode in range(2000):
        exploration = max(exploration * exploration_decay, exploration_min)

        v2.append(exploration)

    axarr[6].plot(v2)
    axarr[6].plot(v)

    axarr[6].set_title('exploration 1')



    plt.show()