import numpy as np
import math
from collections import deque
import scipy.stats # For Gaussian TS

# Helper function for scaling rewards to [0, 1]
def _min_max_scale(reward, min_r, max_r):
    """Scales a reward to [0, 1] using min-max normalization."""
    if min_r is None or max_r is None or max_r == min_r:
        # print("Warning: Invalid min/max reward for scaling. Clipping reward instead.")
        return np.clip(reward, 0.0, 1.0)
    scaled = (reward - min_r) / (max_r - min_r)
    return np.clip(scaled, 0.0, 1.0)

# --- Base Class ---
class MABAlgorithm:
    """Base class for Multi-Armed Bandit Algorithms."""
    def __init__(self, num_arms, horizon=None): # Horizon is optional info
        if num_arms <= 0:
            raise ValueError("Number of arms must be positive.")
        self.num_arms = num_arms
        self.horizon = horizon
        self.time = 0 # Internal time step counter

    def give_pull(self):
        """Returns the index of the arm to pull."""
        raise NotImplementedError

    def get_reward(self, arm_index, reward):
        """Updates the algorithm's state with the observed reward."""
        # Specific update logic implemented in subclasses first
        # THEN increment time step after processing reward for the current step
        self.time += 1

    def reset(self):
        """Resets the algorithm's state."""
        self.time = 0
        # Subclasses should implement specific state resets (counts, values, etc.)

# --- Epsilon Greedy Algorithm ---
class Eps_Greedy(MABAlgorithm):
    """
    Implements Epsilon-Greedy with optional annealing epsilon and
    optional non-stationarity handling (discounting or sliding window).
    """
    def __init__(self, num_arms, horizon=None,
                 epsilon=0.1, epsilon_decay=None, min_epsilon=0.01, # Annealing params
                 discount_factor=None, window_size=None): # Non-stationarity params
        super().__init__(num_arms, horizon)
        if not 0 <= epsilon <= 1:
            raise ValueError("Initial Epsilon must be between 0 and 1.")
        if discount_factor is not None and window_size is not None:
            raise ValueError("Cannot use both discount factor and sliding window.")
        if discount_factor is not None and not (0 < discount_factor < 1):
             raise ValueError("Discount factor must be between 0 and 1.")
        if window_size is not None and window_size <= 0:
             raise ValueError("Window size must be positive.")

        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay # Can be None, 'annealing', or a float factor
        self.min_epsilon = min_epsilon
        self.discount_factor = discount_factor
        self.window_size = window_size
        self.reset() # Initialize state

    @property
    def current_epsilon(self):
        """Calculates epsilon based on decay schedule."""
        if self.epsilon_decay is None: return self.initial_epsilon
        t = self.time # Use current time for decay calculation
        if self.epsilon_decay == 'annealing':
            return max(self.min_epsilon, self.initial_epsilon / (1 + t * 0.01))
        elif isinstance(self.epsilon_decay, float) and 0 < self.epsilon_decay < 1:
            return max(self.min_epsilon, self.initial_epsilon * (self.epsilon_decay ** t))
        else: return self.initial_epsilon

    def give_pull(self):
        if np.random.random() < self.current_epsilon:
            return np.random.randint(self.num_arms)
        else:
            if not hasattr(self, 'values') or not self.values.size: return np.random.randint(self.num_arms)
            best_arms = np.where(self.values == np.max(self.values))[0]
            return np.random.choice(best_arms) if len(best_arms) > 0 else np.random.randint(self.num_arms)


    def get_reward(self, arm_index, reward):
        # Update values based on mode BEFORE calling super().get_reward()
        if self.window_size is not None:
            self.recent_rewards[arm_index].append(reward)
            if self.recent_rewards[arm_index]:
                 self.values[arm_index] = np.mean(self.recent_rewards[arm_index])
                 self.counts[arm_index] = len(self.recent_rewards[arm_index])
            else:
                 self.values[arm_index], self.counts[arm_index] = 0.0, 0
        elif self.discount_factor is not None:
             old_count = self.counts[arm_index]
             self.counts[arm_index] = self.discount_factor * old_count + 1
             # Update EWMA style: value = lambda * old_value + (1-lambda)*reward
             alpha = 1.0 / self.counts[arm_index] if self.counts[arm_index] > 0 else 1.0 # Approximation of changing alpha
             # Or maybe simpler: EWMA value update
             self.values[arm_index] = self.discount_factor * self.values[arm_index] + (1-self.discount_factor)*reward

        else: # Standard Stationary Update
            self.counts[arm_index] += 1
            n = self.counts[arm_index]
            self.values[arm_index] = ((n - 1) / n) * self.values[arm_index] + (1 / n) * reward

        super().get_reward(arm_index, reward) # Let base class increment time

    def reset(self):
        super().reset()
        self.counts = np.zeros(self.num_arms, dtype=float if self.discount_factor or self.window_size else int)
        self.values = np.zeros(self.num_arms, dtype=float)
        if self.window_size is not None:
            self.recent_rewards = [deque(maxlen=self.window_size) for _ in range(self.num_arms)]
            # Need counts even for window mode if give_pull checks it
            self.counts = np.zeros(self.num_arms, dtype=int)
        else: self.recent_rewards = None

# --- UCB (Upper Confidence Bound) Algorithm ---
class UCB(MABAlgorithm):
    """ Standard UCB1 with scaling and optional non-stationarity """
    def __init__(self, num_arms, horizon=None, c=2.0, min_reward=None, max_reward=None, discount_factor=None, window_size=None):
        super().__init__(num_arms, horizon)
        if c < 0: raise ValueError("c cannot be negative.")
        if discount_factor is not None and window_size is not None: raise ValueError("Cannot use both discount factor and sliding window.")
        if discount_factor is not None and not (0 < discount_factor < 1): raise ValueError("Discount factor must be between 0 and 1.")
        if window_size is not None and window_size <= 0: raise ValueError("Window size must be positive.")
        self.c = c
        self.min_reward, self.max_reward = min_reward, max_reward
        self.discount_factor, self.window_size = discount_factor, window_size
        self.reset()

    def give_pull(self):
        total_time = self.time + 1
        # Prioritize unpulled arms
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0: return np.random.choice(unpulled)

        ucb_scores = np.zeros(self.num_arms)
        log_total_time = math.log(max(1, total_time))

        for arm in range(self.num_arms):
            current_count = max(1, self.counts[arm]) # Avoid division by zero
            bonus = self.c * math.sqrt(log_total_time / current_count)
            ucb_scores[arm] = self.values[arm] + bonus

        best_arms = np.where(ucb_scores == np.max(ucb_scores))[0]
        return np.random.choice(best_arms) if len(best_arms) > 0 else np.random.randint(self.num_arms)

    def get_reward(self, arm_index, reward):
        scaled_reward = _min_max_scale(reward, self.min_reward, self.max_reward)

        if self.window_size is not None:
            self.recent_rewards[arm_index].append(scaled_reward)
            if self.recent_rewards[arm_index]:
                self.values[arm_index] = np.mean(self.recent_rewards[arm_index])
                self.counts[arm_index] = len(self.recent_rewards[arm_index])
            else: self.values[arm_index], self.counts[arm_index] = 0.0, 0
        elif self.discount_factor is not None:
             self.counts[arm_index] = self.discount_factor * self.counts[arm_index] + 1
             # Discounted value update (EWMA style)
             self.values[arm_index] = self.discount_factor * self.values[arm_index] + (1 - self.discount_factor) * scaled_reward
        else: # Standard Stationary Update
            self.counts[arm_index] += 1
            n = self.counts[arm_index]
            self.values[arm_index] = ((n - 1) / n) * self.values[arm_index] + (1 / n) * scaled_reward

        super().get_reward(arm_index, reward) # Let base class increment time

    def reset(self):
        super().reset()
        self.counts = np.zeros(self.num_arms, dtype=float if self.discount_factor else int)
        self.values = np.zeros(self.num_arms, dtype=float)
        if self.window_size is not None:
            self.recent_rewards = [deque(maxlen=self.window_size) for _ in range(self.num_arms)]
            self.counts = np.zeros(self.num_arms, dtype=int) # Use int count for window
        else: self.recent_rewards = None

# --- KL-UCB Algorithm ---
class KL_UCB(MABAlgorithm):
    """ KL-UCB requires rewards scaled to [0, 1]. """
    def __init__(self, num_arms, horizon=None, c=1.0, min_reward=None, max_reward=None, discount_factor=None, window_size=None, eps=1e-9):
        super().__init__(num_arms, horizon)
        if discount_factor is not None and window_size is not None: raise ValueError("Cannot use both discount factor and sliding window.")
        if discount_factor is not None and not (0 < discount_factor < 1): raise ValueError("Discount factor must be between 0 and 1.")
        if window_size is not None and window_size <= 0: raise ValueError("Window size must be positive.")
        if min_reward is None or max_reward is None: print("Warning: KL-UCB requires scaling. Ensure min/max_reward are set.")
        self.c, self.min_reward, self.max_reward = c, min_reward, max_reward
        self.discount_factor, self.window_size, self.eps = discount_factor, window_size, eps
        self.reset()

    def _kl_divergence_bernoulli(self, p, q):
        p, q = np.clip(p, self.eps, 1 - self.eps), np.clip(q, self.eps, 1 - self.eps)
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def _solve_klucb_upper_bound(self, emp_mean, count, time):
        if count == 0: return 1.0
        effective_time = max(time, 1)
        log_term = math.log(effective_time)
        if self.c > 0 and log_term > 0: target_bound = (log_term + self.c * math.log(log_term)) / count
        elif log_term >= 0: target_bound = log_term / count
        else: target_bound = float('inf')

        low, high = emp_mean, 1.0
        for _ in range(16): # Binary search iterations
            mid = (low + high) / 2.0
            kl_div = self._kl_divergence_bernoulli(emp_mean, mid)
            if kl_div <= target_bound: low = mid
            else: high = mid
        return high

    def give_pull(self):
        total_time = self.time + 1
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0: return np.random.choice(unpulled)

        kl_ucb_scores = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            current_count = max(1, self.counts[arm])
            current_sum = self.sum_rewards[arm]
            emp_mean = np.clip(current_sum / current_count, 0.0, 1.0)
            kl_ucb_scores[arm] = self._solve_klucb_upper_bound(emp_mean, current_count, total_time)

        best_arms = np.where(kl_ucb_scores == np.max(kl_ucb_scores))[0]
        return np.random.choice(best_arms) if len(best_arms) > 0 else np.random.randint(self.num_arms)


    def get_reward(self, arm_index, reward):
        scaled_reward = _min_max_scale(reward, self.min_reward, self.max_reward)

        if self.window_size is not None:
            self.recent_rewards[arm_index].append(scaled_reward)
            if self.recent_rewards[arm_index]:
                self.sum_rewards[arm_index] = np.sum(self.recent_rewards[arm_index])
                self.counts[arm_index] = len(self.recent_rewards[arm_index])
            else: self.sum_rewards[arm_index], self.counts[arm_index] = 0.0, 0
        elif self.discount_factor is not None:
            self.counts[arm_index] = self.discount_factor * self.counts[arm_index] + 1
            self.sum_rewards[arm_index] = self.discount_factor * self.sum_rewards[arm_index] + scaled_reward
        else: # Standard Stationary Update
            self.counts[arm_index] += 1
            self.sum_rewards[arm_index] += scaled_reward

        super().get_reward(arm_index, reward) # Let base class increment time

    def reset(self):
        super().reset()
        self.sum_rewards = np.zeros(self.num_arms, dtype=float)
        self.counts = np.zeros(self.num_arms, dtype=float if self.discount_factor else int)
        if self.window_size is not None:
             self.recent_rewards = [deque(maxlen=self.window_size) for _ in range(self.num_arms)]
             self.counts = np.zeros(self.num_arms, dtype=int) # Use int count for window
        else: self.recent_rewards = None

# --- Gaussian Thompson Sampling ---
class GaussianThompsonSampling(MABAlgorithm):
    """ Gaussian TS assuming normal rewards. Optional non-stationarity handling. """
    def __init__(self, num_arms, horizon=None, discount_factor=None, window_size=None, prior_mean=0.0, prior_var=1e6, min_var=1e-6):
        super().__init__(num_arms, horizon)
        if discount_factor is not None and window_size is not None: raise ValueError("Cannot use both discount factor and sliding window.")
        if discount_factor is not None and not (0 < discount_factor < 1): raise ValueError("Discount factor must be between 0 and 1.")
        if window_size is not None and window_size <= 0: raise ValueError("Window size must be positive.")
        self.discount_factor, self.window_size = discount_factor, window_size
        self.prior_mean, self.prior_var, self.min_var = prior_mean, prior_var, min_var
        self.reset()

    def give_pull(self):
        sampled_theta = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            count = self.counts[arm]
            sum_r, sum_r_sq = self.sum_rewards[arm], self.sum_rewards_sq[arm]

            if count < 2: # Use prior or pull if not enough data
                sampled_theta[arm] = np.random.normal(self.prior_mean, math.sqrt(self.prior_var))
            else: # Estimate mean and variance, then sample
                mean_est = sum_r / count
                variance_est = max(self.min_var, (sum_r_sq / count) - (mean_est ** 2)) # MLE variance
                # Sample from N(mean_est, variance_est / count) - approximate posterior
                post_std_dev = math.sqrt(variance_est / count)
                sampled_theta[arm] = np.random.normal(mean_est, post_std_dev)

        best_arms = np.where(sampled_theta == np.max(sampled_theta))[0]
        return np.random.choice(best_arms) if len(best_arms) > 0 else np.random.randint(self.num_arms)


    def get_reward(self, arm_index, reward):
        # Update sums/counts based on mode BEFORE calling super().get_reward()
        if self.window_size is not None:
            if len(self.recent_rewards[arm_index]) == self.window_size: # Remove oldest contribution
                old_reward = self.recent_rewards[arm_index][0]
                self.sum_rewards[arm_index] -= old_reward
                self.sum_rewards_sq[arm_index] -= old_reward**2
            self.recent_rewards[arm_index].append(reward) # Add new reward
            self.sum_rewards[arm_index] += reward
            self.sum_rewards_sq[arm_index] += reward**2
            self.counts[arm_index] = len(self.recent_rewards[arm_index])
        elif self.discount_factor is not None: # Discounted Update
            self.counts[arm_index] = self.discount_factor * self.counts[arm_index] + 1
            self.sum_rewards[arm_index] = self.discount_factor * self.sum_rewards[arm_index] + reward
            self.sum_rewards_sq[arm_index] = self.discount_factor * self.sum_rewards_sq[arm_index] + reward**2
        else: # Standard Stationary Update
            self.counts[arm_index] += 1
            self.sum_rewards[arm_index] += reward
            self.sum_rewards_sq[arm_index] += reward**2

        super().get_reward(arm_index, reward) # Let base class increment time

    def reset(self):
        super().reset()
        self.counts = np.zeros(self.num_arms, dtype=float if self.discount_factor else int)
        self.sum_rewards = np.zeros(self.num_arms, dtype=float)
        self.sum_rewards_sq = np.zeros(self.num_arms, dtype=float)
        if self.window_size is not None:
             self.recent_rewards = [deque(maxlen=self.window_size) for _ in range(self.num_arms)]
             # Counts needed for window mode give_pull variance calc
             self.counts = np.zeros(self.num_arms, dtype=int)
        else: self.recent_rewards = None


# --- Sliding Window UCB (SW-UCB) ---  <-- CORRECTED CLASS DEFINITION -->
class SW_UCB(MABAlgorithm):
    """
    Implements the Sliding Window UCB (SW-UCB) algorithm.
    Focuses on rewards within a fixed-size window W.
    Uses the formula: SW_UCB = mean_W + c * sqrt(2*log(t) / n_W)
    """
    def __init__(self, num_arms, horizon=None, window_size=100, c=2.0,
                 min_reward=None, max_reward=None): # Scaling params
        super().__init__(num_arms, horizon)
        if window_size <= 0: raise ValueError("Window size W must be positive.")
        if c < 0: raise ValueError("Exploration parameter c cannot be negative.")
        self.window_size = window_size
        self.c = c
        self.min_reward, self.max_reward = min_reward, max_reward
        self.reset()

    def give_pull(self):
        total_time = self.time + 1
        ucb_scores = np.full(self.num_arms, float('inf')) # Default to infinity for unpulled

        for arm in range(self.num_arms):
            arm_deque = self.recent_rewards[arm]
            n_W = len(arm_deque)

            if n_W > 0: # Only calculate score if data exists in window
                mean_W = np.mean(arm_deque)
                log_term = math.log(max(1, total_time))
                bonus = self.c * math.sqrt((2 * log_term) / n_W)
                ucb_scores[arm] = mean_W + bonus
            # Else: score remains infinity, prioritizing pull

        # Handle case where all scores are infinity (very start)
        if np.all(np.isinf(ucb_scores)):
             return np.random.randint(self.num_arms)

        best_arms = np.where(ucb_scores == np.nanmax(ucb_scores[np.isfinite(ucb_scores)]))[0] # Find max among finite scores
        return np.random.choice(best_arms) if len(best_arms) > 0 else np.random.randint(self.num_arms)


    def get_reward(self, arm_index, reward):
        # --- Scale Reward ---
        scaled_reward = _min_max_scale(reward, self.min_reward, self.max_reward)
        # --- Sliding Window Update ---
        self.recent_rewards[arm_index].append(scaled_reward)
        # --- Call superclass get_reward ---
        super().get_reward(arm_index, reward) # Let base class increment time


    def reset(self):
        super().reset()
        self.recent_rewards = [deque(maxlen=self.window_size) for _ in range(self.num_arms)]

# --- Exponentially Weighted Moving Average UCB (EWMA-UCB) --- <-- CORRECTED CLASS DEFINITION -->
class EWMA_UCB(MABAlgorithm):
    """
    Implements EWMA-UCB using discounted sums and counts.
    Formula: EWMA_UCB = ewma_mean + c * sqrt(2*log(t) / discounted_n)
    """
    def __init__(self, num_arms, horizon=None, lambda_decay=0.99, c=2.0,
                 min_reward=None, max_reward=None):
        super().__init__(num_arms, horizon)
        if not (0 < lambda_decay < 1): raise ValueError("lambda_decay must be between 0 and 1.")
        if c < 0: raise ValueError("c cannot be negative.")
        self.lambda_decay = lambda_decay
        self.c = c
        self.min_reward, self.max_reward = min_reward, max_reward
        self.reset()

    def give_pull(self):
        total_time = self.time + 1
        ewma_ucb_scores = np.full(self.num_arms, float('inf')) # Default to infinity

        for arm in range(self.num_arms):
            discounted_n = self.discounted_counts[arm]

            if discounted_n > 1e-9: # Check if count is effectively non-zero
                ewma_mean = self.discounted_sums[arm] / discounted_n
                log_term = math.log(max(1, total_time))
                bonus = self.c * math.sqrt((2 * log_term) / discounted_n)
                ewma_ucb_scores[arm] = ewma_mean + bonus
            # Else: score remains infinity

        if np.all(np.isinf(ewma_ucb_scores)):
             return np.random.randint(self.num_arms)

        best_arms = np.where(ewma_ucb_scores == np.nanmax(ewma_ucb_scores[np.isfinite(ewma_ucb_scores)]))[0]
        return np.random.choice(best_arms) if len(best_arms) > 0 else np.random.randint(self.num_arms)


    def get_reward(self, arm_index, reward):
        # --- Scale Reward ---
        scaled_reward = _min_max_scale(reward, self.min_reward, self.max_reward)
        # --- Discounted Update ---
        self.discounted_sums[arm_index] = self.lambda_decay * self.discounted_sums[arm_index] + scaled_reward
        self.discounted_counts[arm_index] = self.lambda_decay * self.discounted_counts[arm_index] + 1
        # --- Call superclass get_reward ---
        super().get_reward(arm_index, reward) # Let base class increment time

    def reset(self):
        super().reset()
        self.discounted_sums = np.zeros(self.num_arms, dtype=float)
        self.discounted_counts = np.zeros(self.num_arms, dtype=float)

# --- Example Usage (Including corrected classes) ---
if __name__ == '__main__':
    NUM_ARMS = 3 # Match user's simulation context (SMA, RSI, BB)
    HORIZON = 1000

    # --- Non-Stationary Reward Example ---
    print("--- USING NON-STATIONARY REWARDS ---")
    arm_means_phase1 = np.array([0.02, -0.01, 0.01]) # Example: SMA good initially
    arm_means_phase2 = np.array([-0.01, 0.03, 0.01]) # Example: RSI becomes better
    arm_stds = np.ones(NUM_ARMS) * 0.05 # Example noise level

    def generate_reward_nonstationary(arm_index, current_time):
        means = arm_means_phase1 if current_time < HORIZON / 2 else arm_means_phase2
        return np.random.normal(means[arm_index], arm_stds[arm_index])

    generate_reward = generate_reward_nonstationary # Use this generator

    print(f"Phase 1 Means: {arm_means_phase1}")
    print(f"Phase 2 Means: {arm_means_phase2}")
    best_arm_phase1 = np.argmax(arm_means_phase1)
    best_arm_phase2 = np.argmax(arm_means_phase2)
    print(f"Best Arm Phase 1: {best_arm_phase1}, Phase 2: {best_arm_phase2}")

    # Assume estimated reward range for scaling:
    EST_MIN_R = -0.15 # Adjust based on your strategy returns
    EST_MAX_R = 0.15 # Adjust based on your strategy returns

    # --- Select algorithm ---
    # algo = SW_UCB(NUM_ARMS, HORIZON, window_size=100, c=2.0, min_reward=EST_MIN_R, max_reward=EST_MAX_R)
    algo = EWMA_UCB(NUM_ARMS, HORIZON, lambda_decay=0.99, c=2.0, min_reward=EST_MIN_R, max_reward=EST_MAX_R)
    # algo = UCB(NUM_ARMS, HORIZON, c=2.0, min_reward=EST_MIN_R, max_reward=EST_MAX_R) # Stationary UCB for comparison
    # algo = GaussianThompsonSampling(NUM_ARMS, HORIZON, discount_factor=0.995)

    print(f"\nRunning Simulation with: {algo.__class__.__name__}")

    # --- Simulation Loop ---
    algo.reset() # Ensure clean state
    cumulative_reward = 0
    chosen_arms = []
    rewards_over_time = []

    for t in range(HORIZON):
        try:
            # 1. Choose arm
            arm_to_pull = algo.give_pull()
            chosen_arms.append(arm_to_pull)

            # 2. Get reward (using selected generator)
            # Pass the algorithm's current time
            reward = generate_reward(arm_to_pull, algo.time)

            # 3. Update algorithm
            algo.get_reward(arm_to_pull, reward) # This now increments time internally via super()

            # 4. Track cumulative reward
            cumulative_reward += reward
            rewards_over_time.append(reward)
        except Exception as e:
            print(f"Error during simulation loop at step {t+1} for arm {arm_to_pull}: {e}")
            # Optionally break or add more detailed debugging
            # import traceback
            # traceback.print_exc()
            break


    # --- Results ---
    print(f"\n--- Results for {algo.__class__.__name__} ---")
    # ... (rest of the results printing code from previous example) ...

    print(f"Total Cumulative Reward: {cumulative_reward:.4f}")
    (unique, counts) = np.unique(chosen_arms, return_counts=True)
    frequencies = np.zeros(NUM_ARMS, dtype=int)
    if unique.size > 0: frequencies[unique] = counts
    print("Arm Pull Frequencies:")
    for i in range(NUM_ARMS):
        print(f"  Arm {i}: {frequencies[i]} pulls ({(frequencies[i]/max(1,HORIZON))*100:.1f}%)")

    # Plotting (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(np.cumsum(rewards_over_time))
        plt.title(f"Cumulative Reward Over Time ({algo.__class__.__name__})")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Reward")
        if generate_reward == generate_reward_nonstationary:
            plt.axvline(HORIZON / 2, color='r', linestyle='--', label='Change Point')
            plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\nInstall matplotlib to see the cumulative reward plot (`pip install matplotlib`)")
    except NameError: # Handle if generate_reward_nonstationary not defined
         pass