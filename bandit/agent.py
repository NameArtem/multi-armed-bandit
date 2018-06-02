import numpy as np

class Agent(object):
    """
    Агент, выбирающий одно из доступных действий на каждом шаге.
    value_estimates = Q
    """
    def __init__(self, bandit, policy):
        self.policy = policy
        self.k = bandit.k
        self.qt = np.zeros(bandit.k)
        self.attempts = np.zeros(bandit.k)
        self.t = 0
        self.last_action = None

    def __str__(self):
        return '{}'.format(str(self.policy))

    def reset(self):
        """
        Сброс памяти агента до начального состояния.
        """
        self.qt[:] = 0
        self.attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        """
        Выбор действия (с помощью стратегии policy). 
        """
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        """
        Получение награды агентом.
        """
        self.attempts[self.last_action] += 1
        
        g = 1 / self.attempts[self.last_action]
        
        q = self.qt[self.last_action]

        self.qt[self.last_action] += g*(reward - q)
        self.t += 1


