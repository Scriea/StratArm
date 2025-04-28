import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars


class MABAlgorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon

    def give_pull(self):
        raise NotImplementedError

    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(MABAlgorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# class Eps_Greedy(Algorithm):
class UCB(MABAlgorithm):
    def __init__(self, num_arms, horizon, c =2):
        super().__init__(num_arms, horizon)
        self.time = 1                
        self.counts = np.zeros(num_arms)     
        self.values = np.ones(num_arms)
        self.ucb = np.ones(num_arms)
        self.c = c

    def give_pull(self):
        # Initially pulling in Round Robin Fashion till each arm is pulled atleast once
        if 0 in self.counts:
            for arm, count in enumerate(self.counts):
                if count==0:
                    return arm
                
        return np.argmax(self.ucb)

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        self.time += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = ((n - 1) * value + reward)/n
        self.ucb = [self._ucb_solver(self.time, self.values[i], self.counts[i]) 
                    for i in range(self.num_arms)]
        
    def _ucb_solver(self, time, action, count):
        if count==0:
            return 1
        return min(action + math.sqrt(self.c*math.log(time)/count), 1)

class KL_UCB(MABAlgorithm):
    def __init__(self, num_arms, horizon, c=2):
        super().__init__(num_arms, horizon)
        self.time = 1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        self.c = c

    

    def give_pull(self):
        if 0 in self.counts:
                for arm, count in enumerate(self.counts):
                    if count==0:
                        return arm
        return np.argmax(self.ucb)


    def get_reward(self, arm_index, reward):
        self.time += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = ((n - 1) * value + reward)/n
        self.ucb = [self._klucb_solver(self.time, self.values[index], self.counts[index])
                    for index in range(self.num_arms)]

    def _klucb_solver(self, time, action, count):
        if count==0:
            return 1
        bound = (math.log(time)+self.c*math.log(math.log(time)))/count
        low, high = action, 1
        while high-low>=1e-4:
            mid = (low + high) / 2
            div = self.KL(action, mid)
            if div<=bound and bound-div<=1e-4:
                return mid
            elif div > bound:
                high = mid
            else:
                low = mid
        return low


    def KL(self, p, q):
        if p==q:
            return 0
        if q==0 or q==1:
            return float("inf")
        if p==1:
            return -math.log(q)
        if p==0:
            return -math.log(1-q)
        return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))



class Thompson_Sampling(MABAlgorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.success = np.zeros(num_arms)
        self.failure = np.zeros(num_arms)

    def give_pull(self):
        return np.argmax([self._belief(self.success[i], self.failure[i]) 
                          for i in range(self.num_arms)])

    def get_reward(self, arm_index, reward):
        if (reward == 1):
            self.success[arm_index] += 1
        else:
            self.failure[arm_index] += 1

    def _belief(success, failure):
        return np.random.beta(success+1, failure+1)

