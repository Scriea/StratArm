import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from .strategy import TradingStrategy
from .bandit_algorithms import MABAlgorithm

class Simulator:
    def __init__(self, df: pd.DataFrame, strategies: List[TradingStrategy], mab: MABAlgorithm):
        self.df = df
        self.strategies = strategies
        self.mab = mab
        self.results = {
            'cumulative_returns': [],
            'cumulative_regret': [],
            'chosen_arms': [],
            'returns': []
        }

    def run(self) -> dict:
        """Execute the bandit simulation with correct reward calculation"""
        price_returns = self.df['Price'].pct_change().fillna(0).values
        
        for t in range(1, len(self.df)):
            # Calculate potential rewards for all strategies
            current_return = price_returns[t]
            strategy_rewards = {}
            
            for i, strat in enumerate(self.strategies):
                signal = strat.generate_signals(self.df).iat[t]
                
                # Calculate reward based on signal
                if signal == 1:  # Buy
                    reward = current_return
                elif signal == -1:  # Sell
                    reward = -current_return
                else:  # Hold
                    reward = 0
                
                strategy_rewards[i] = reward
            
            # Store all potential rewards for regret calculation
            self.results['all_rewards'].append(strategy_rewards)
            optimal_reward = max(strategy_rewards.values())
            self.results['optimal_rewards'].append(optimal_reward)
            
            # Select arm and get actual reward
            arm = self.mab.give_pull()
            actual_reward = strategy_rewards[arm]
            self.mab.get_reward(arm, actual_reward)
            
            # Track metrics
            self.results['chosen_arms'].append(arm)
            self.results['cumulative_regret'].append(optimal_reward - actual_reward)
            
            # Update cumulative returns
            if t == 1:
                cumulative = actual_reward
            else:
                cumulative = self.results['cumulative_returns'][-1] + actual_reward
            self.results['cumulative_returns'].append(cumulative)
            
        return self.results

    def plot_results(self, figsize=(15, 10)):
        """Visualize simulation results using three mandatory plots"""
        plt.figure(figsize=figsize)
        
        # 1. Regret Plot
        plt.subplot(3, 1, 1)
        plt.plot(np.cumsum(self.results['cumulative_regret']), color='red')
        plt.title('Cumulative Regret')
        plt.xlabel('Time Steps')
        plt.ylabel('Regret')
        plt.grid(True)
        
        # 2. Arm Selection Histogram
        plt.subplot(3, 1, 2)
        arm_counts = pd.Series(self.results['chosen_arms']).value_counts().sort_index()
        arm_counts.plot(kind='bar', color='blue', alpha=0.7)
        plt.title('Arm Selection Frequency')
        plt.xlabel('Strategy Index')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.grid(True)
        
        # 3. Returns Comparison
        plt.subplot(3, 1, 3)
        plt.plot(self.results['cumulative_returns'], color='green')
        plt.title('Cumulative Returns')
        plt.xlabel('Time Steps')
        plt.ylabel('Returns (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()