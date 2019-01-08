
import numpy as np
import numpy.random as nr
import copy
import random
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, scale=0.1,  mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.scale = scale
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale





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
    eps_min = 0.05
    for i_episode in range(2000):
        ou = OUNoise(1, mu=0, theta=0.15, sigma=6)
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