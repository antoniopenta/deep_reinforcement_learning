
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


    ou = OUNoise(3,mu=0, theta=0.15, sigma=3)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    axarr[0].plot(states)
    axarr[0].set_title(' original mu=1,theta=0.15,sigma=0.6 ')


    eps = 1
    eps_decay = 0.999
    noise_episode = []
    eps_min=0.05
    for i_episode in range(2000):
        ou = OUNoise(1, mu=0, theta=0.15, sigma=3)
        states = []
        for i in range(1000):
            if np.random.random()>0.5:
                states.append(eps * ou.sample())
            else:
                states.append(-eps * ou.sample())
        noise_episode.append(np.mean(states))
        eps = max(eps_min, eps * eps_decay)
    axarr[1].plot(noise_episode)
    axarr[1].set_title('noise_episode with expnetial decay mu=1,theta=0.15,sigma=0.6')

    plt.show()