import numpy as np

class Bandit(object):
    """
    Бандит, в котором сначала по нормальному распределению с заданным средним mu и 
    дисперсией sigma выбираются средние для всех ручек, а потом каждая ручка работает
    по нормальному распределению с выбранным ранее средним, и стандартной дисперсией 1.
    """
    def __init__(self, k, mu=0, sigma=1):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.means = np.random.normal(mu, sigma, k)
        self.optimal = np.argmax(self.means)

    def reset(self):
        """
        Сброс памяти бандита.
        """
        self.means = np.random.normal(self.mu, self.sigma, self.k)
        self.optimal = np.argmax(self.means)
        
    def pull(self, action):
        """
        Возвращает полученную награду и флаг, показывающий, оптимально ли выбранное действие.
        """
        return (np.random.normal(self.means[action]), action == self.optimal)
