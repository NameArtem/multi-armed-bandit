import matplotlib.pyplot as plt
import numpy as np


class Environment(object):
    """
    Класс среды, моделирующий взаимодействие бандита и агента.
    """
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        """
        Сброс модели до начального состояния.
        """
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=100, experiments=1):
        """
        Запуск модели.
        """
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    scores[t, i] += reward
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments
    
    def run_with_change(self, trials_before_change = 100, trials_after_change=100, experiments=1):
        """
        Запуск модели с изменением распределений бандита после
        некоторого числа шагов.
        """
        scores = np.zeros((trials_before_change + trials_after_change, len(self.agents)))
        optimal = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for t1 in range(trials_before_change):
                for i1, agent1 in enumerate(self.agents):
                    action1 = agent1.choose()
                    reward1, is_optimal1 = self.bandit.pull(action1)
                    agent1.observe(reward1)

                    scores[t1, i1] += reward1
                    if is_optimal1:
                        optimal[t1, i1] += 1
            
            self.bandit.reset()
            for t2 in range(trials_after_change):
                for i2, agent2 in enumerate(self.agents):
                    action2 = agent2.choose()
                    reward2, is_optimal2 = self.bandit.pull(action2)
                    agent2.observe(reward2)

                    scores[trials_before_change + t2, i2] += reward2
                    if is_optimal2:
                        optimal[trials_before_change + t2, i2] += 1

        return scores / experiments, optimal / experiments

    def plot(self, scores, optimal):
        """
        Построение графика.
        """
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(scores)
        plt.ylabel('Average reward')
        plt.legend(self.agents, loc=4)
        plt.subplot(2, 1, 2)
        plt.plot(optimal * 100)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal')
        plt.xlabel('Time step')
        plt.legend(self.agents, loc=4)
        plt.show()
