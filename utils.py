import numpy as np
from scipy import spatial

class CostFunctions(object):
    maxValue = None
    maxX = None
    def __init__(self, maxvalue,optimal_arms, toMaximize=True, sd=1, maxiter=10 ):
        self._maxiter = maxiter
        self.toMaximize = toMaximize
        self.sd = sd
        self._nit = 0
        self.requested_arm = []
        self.reward = []
        self.optimal_arms = optimal_arms
        if self.toMaximize:
            self.MaxMinValue = maxvalue
        else:
            self.MaxMinValue = - maxvalue

    #Needs to be implemented the function without noise
    def func(self,x):
        return

    def funcNoNoise(self,x):
        value = self.func(x)
        if self.toMaximize:
            return value
        else:
            return -value


    def eval(self,x):
        self.requested_arm.append(x)
        value = self.func(x)
        if self.toMaximize:
            reward = np.random.normal(loc=value, scale=self.sd, size=1)[0]
            self.reward.append(reward)
            return reward
        else:
            reward = np.random.normal(loc=-value, scale=self.sd, size=1)[0]
            self.reward.append(reward)
            return reward


    def __call__(self, x):
        if self._nit >= self._maxiter:

            raise ValueError("Max iterations allowed is over")

        self._nit = 1 + self._nit
        return self.eval(x)

    def regret(self):
        return abs(np.ones(np.size(self.reward)) * self.MaxMinValue - self.reward)

    def cum_regret(self):
        regret = self.regret()
        return np.cumsum(regret)

    def generate_info(self, best_arm, timetocomplete, algorithm):
        dist = []
        for arm in self.optimal_arms:
            d = spatial.distance.euclidean(arm,best_arm)
            dist.append(d)
            # print('Best arm: ', best_arm, ' Optimal arm: ', arm, ' Distance: ', d)
        smallestdist = np.min(dist)
        # print('Smallest distance:', smallestdist)
        info = {
            'best_arm': best_arm,
            'algorithm': algorithm,
            'function': self.__class__.__name__,
            'euclidean_distance':smallestdist,
            'true_reward_difference': abs(self.MaxMinValue - self.funcNoNoise(best_arm)),
            'cumulative_regret': np.sum(self.regret()),
            'timetocomple': timetocomplete
                }
        return info