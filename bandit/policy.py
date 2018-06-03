import numpy as np


class Policy(object):
    """
    Стратегия выбора действия.
    """
    def __str__(self):
        """
        Лейбл (для графиков).
        """
        return 'generic policy'
    
    def actionFromDist(self, pi):
        """
        Выбрать случайное действие из множества
        с заданным распределением вероятностей.
        """
        cs = np.cumsum(pi)
        s = np.random.random()
        return np.where(s < cs)[0][0]
    
    def actionFromBest(self, agent):
        """
        Выбрать случайное действие равномерно
        из множетства действий с максимальным средним.
        """
        qt = agent.qt
        action = np.argmax(qt)
        check = np.where(qt == action)[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)

    def choose(self, agent):
        """
        Функция выбора. Возвращает индекс выбраного действия.
        """
        return 0


class RandomPolicy(Policy):
    """
    Чисто исследовательская стратегия.
    Каждый раз действие выбирается случайно и равномерно.
    """
    def __str__(self):
        return 'random'
    
    def choose(self, agent):
        return np.random.choice(len(agent.qt))


class GreedyPolicy(Policy):
    """
    Жадная стратегия. Получается, если в E-жадной стратегии выбрать
    параметр E = 0.
    """
    def __str__(self):
        return 'greedy'
    
    def choose(self, agent):
        return self.actionFromBest(agent)


class EpsilonGreedyPolicy(Policy):
    """
    E-жадная стратегия (параметр - epsilon).
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        qt = agent.qt
        if np.random.random() < self.epsilon:
            return np.random.choice(len(qt))
        else:
            return self.actionFromBest(agent)


class SoftmaxPolicy(Policy):
    """
    Стратегия softmax.
    """
    def __init__(self, tau = 1):
        self.tau = tau
    
    def __str__(self):
        return 'Softmax (tau={})'.format(self.tau)

    def choose(self, agent):
        qt = agent.qt
        pi = np.exp(qt / self.tau) / np.sum(np.exp(qt / self.tau))
        return self.actionFromDist(pi)
