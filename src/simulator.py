# /Users/mohdtahaabbas/StratArm/src/simulator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from .strategy import TradingStrategy
from .bandit_algorithms import MABAlgorithm

class Simulator:
    def __init__(self, df: pd.DataFrame, strategies: List[TradingStrategy], mab: MABAlgorithm):
        """
        Initializes the simulator.

        Args:
            df: Preprocessed DataFrame with price data and necessary features.
                Assumes df is output from data_utils.calculate_features (no leading NaNs).
            strategies: A list of TradingStrategy objects.
            mab: An initialized MABAlgorithm object.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("DataFrame df cannot be empty.")

        if df.isnull().values.any():
             print("Warning: Initial DataFrame df contains NaNs. Ensure this is handled (e.g., by feature calculation or signal generation).")

        self.df = df.reset_index(drop=True) # Ensure index is 0-based sequential
        self.strategies = strategies
        self.mab = mab
        self.num_strategies = len(strategies)

        if self.num_strategies != self.mab.num_arms:
            raise ValueError(f"Number of strategies ({self.num_strategies}) must match MAB algorithm's num_arms ({self.mab.num_arms}).")

        # Initialize results dictionary correctly
        self.results = {
            'cumulative_returns': [], # MAB policy's cumulative returns
            'cumulative_regret': [],  # Stores PER-STEP regret (plot uses cumsum)
            'chosen_arms': [],        # <<< Index of the arm chosen at each step >>>
            'actual_rewards': [],     # Reward obtained by the chosen arm at each step
            'optimal_rewards': [],    # Best possible reward at each step
            'all_strategy_rewards': [] # List of dicts {arm_index: reward} per step
        }

    def run(self) -> dict:
        """Execute the bandit simulation with pre-calculated signals and correct reward/regret."""
        print("Starting simulation...")

        # --- Calculate percentage returns (handle potential inf values) ---
        if 'Price' not in self.df.columns:
             raise ValueError("DataFrame must contain a 'Price' column.")
        price_returns = self.df['Price'].pct_change()
        # Replace inf/-inf which can occur if price was 0 then non-zero, with 0
        price_returns.replace([np.inf, -np.inf], 0, inplace=True)
        # Fill initial NaN (at index 0) with 0
        price_returns = price_returns.fillna(0)

        # --- Pre-calculate all signals for all strategies ---
        print("Pre-calculating strategy signals...")
        all_signals = pd.DataFrame(index=self.df.index)
        for i, strat in enumerate(self.strategies):
            try:
                # Assume generate_signals returns a Series aligned with df.index
                # The signal Series should already incorporate the necessary shift(1)
                print(f"  Generating signals for strategy {i}: {strat.name}...")
                all_signals[i] = strat.generate_signals(self.df)
                # Verify the output type and index
                if not isinstance(all_signals[i], pd.Series):
                     raise TypeError(f"Strategy {strat.name} did not return a pandas Series.")
                if not all_signals[i].index.equals(self.df.index):
                     print(f"Warning: Index mismatch for strategy {strat.name}. Attempting reindex.")
                     all_signals[i] = all_signals[i].reindex(self.df.index)

            except Exception as e:
                 print(f"Error generating signals for strategy {i} ({strat.name}): {e}")
                 # Consider logging the full traceback
                 import traceback
                 traceback.print_exc()
                 raise # Re-raise the exception to stop simulation if signals fail

        # Check for NaNs in signals AFTER the intended shift(1) warmup period
        # Signals at index 0 will be NaN due to shift(1)
        if all_signals.iloc[1:].isnull().values.any():
              print("Warning: NaNs detected in signals after the first time step. Strategies might have different warmup periods or issues.")


        print("Running simulation loop...")
        # Horizon: Number of time steps where decisions are made
        # Starts from t=1 because returns and shifted signals depend on previous data
        horizon = len(self.df) - 1
        cumulative_return_value = 0.0 # Initialize cumulative return

        # Simulation loop starts from t=1 (second row, index 1) because:
        # - price_returns[t] is the return from t-1 to t
        # - all_signals[i].iloc[t] is the signal generated using data up to t-1
        for t in range(1, len(self.df)):
            current_return = price_returns.iloc[t] # Use iloc for 0-based integer indexing

            # --- Calculate potential rewards for all strategies at time t ---
            strategy_rewards = {}
            for i in range(self.num_strategies):
                # Get pre-calculated signal for strategy i at time t
                # This signal was generated based on data up to t-1
                signal = all_signals[i].iloc[t]

                # Handle potential NaN signals (e.g., during strategy warmup)
                if pd.isna(signal):
                    reward = 0.0 # Assign neutral reward for NaN signal
                # Calculate reward based on signal and market return at time t
                elif signal == 1:    # Buy signal generated at t-1
                    reward = current_return # Profit/loss from price move at t
                elif signal == -1: # Sell signal generated at t-1
                    reward = -current_return # Profit/loss from price move at t (inverted)
                else:              # Hold signal (signal == 0 or other invalid values treated as 0)
                    reward = 0.0

                strategy_rewards[i] = reward

            # --- Determine optimal reward and calculate regret ---
            if not strategy_rewards: # Should not happen if num_strategies > 0
                 optimal_reward = 0.0
            else:
                 # Filter out NaN rewards before taking max, if any occurred
                 valid_rewards = [r for r in strategy_rewards.values() if pd.notna(r)]
                 # If all rewards are NaN or invalid, optimal is 0
                 optimal_reward = max(valid_rewards) if valid_rewards else 0.0

            # --- MAB Algorithm pulls an arm ---
            try:
                 arm = self.mab.give_pull()
                 if not isinstance(arm, (int, np.integer)) or arm < 0 or arm >= self.num_strategies:
                     print(f"Warning: MAB algorithm returned invalid arm index {arm} (type: {type(arm)}) at step {t}. Choosing arm 0.")
                     arm = 0
            except Exception as e:
                 print(f"Error during MAB give_pull() at step {t}: {e}")
                 # Decide how to handle: stop simulation, default arm?
                 print("Defaulting to arm 0 due to MAB error.")
                 arm = 0


            actual_reward = strategy_rewards.get(arm, 0.0) # Get reward for chosen arm, default to 0.0 if arm invalid


            # --- Append results for this time step ---
            self.results['optimal_rewards'].append(optimal_reward)
            self.results['actual_rewards'].append(actual_reward)
            self.results['chosen_arms'].append(arm) # <<< THIS LINE STORES THE CHOSEN ARM >>>


            # --- Calculate and store step regret ---
            step_regret = optimal_reward - actual_reward

            # ---- START DEBUG PRINTS (Commented Out By Default) ----
            # Uncomment these lines to debug negative regret issues
            # if step_regret < -1e-9: # Allow for tiny float inaccuracies
            #     print(f"\n--- NEGATIVE REGRET DETECTED at t={t} ---")
            #     print(f"  Time index in DataFrame: {self.df.index[t]}") # Use df.index if meaningful dates exist
            #     print(f"  Current Return: {current_return:.6f}")
            #     # Fetch signals for this step for context
            #     signals_this_step = {i: all_signals[i].iloc[t] for i in range(self.num_strategies)}
            #     print(f"  Signals this Step: {signals_this_step}")
            #     print(f"  Strategy Rewards: {strategy_rewards}")
            #     print(f"  Optimal Reward: {optimal_reward:.6f}")
            #     print(f"  Chosen Arm: {arm}")
            #     print(f"  Actual Reward: {actual_reward:.6f}")
            #     print(f"  Step Regret: {step_regret:.6f} (SHOULD BE >= 0)")
            #     print(f"------------------------------------------")
                # Optional: stop execution
                # raise ValueError(f"Negative regret detected at step {t}")
            # ---- END DEBUG PRINTS ----

            # Append the calculated step regret (must be non-negative)
            self.results['cumulative_regret'].append(max(0.0, step_regret)) # Ensure non-negative due to potential float issues

            # --- Update MAB Algorithm ---
            try:
                # IMPORTANT: Consider if 'actual_reward' needs transformation/scaling
                # before passing to certain MAB algorithms (UCB, KL-UCB, Thompson).
                # scaled_reward = self.scale_reward(actual_reward) # Implement scaling if needed
                # self.mab.get_reward(arm, scaled_reward)
                self.mab.get_reward(arm, actual_reward) # Passing raw reward for now
            except Exception as e:
                 print(f"Error during MAB get_reward() for arm {arm} at step {t}: {e}")
                 # Decide how to handle: stop simulation, skip update?


            # --- Update cumulative returns for the policy ---
            cumulative_return_value += actual_reward
            self.results['cumulative_returns'].append(cumulative_return_value)

            # --- Progress Update ---
            if (t + 1) % 100 == 0 or t == horizon: # Print every 100 steps or at the end
                 # Adding +1 to t for 1-based step counting in printout
                 print(f"  Step {t+1}/{horizon+1} completed. Current Cumulative Return: {cumulative_return_value:.4f}")


        print("Simulation finished.")
        return self.results

    def plot_results(self, figsize=(15, 12)):
        """Visualize simulation results: Cumulative Regret, Arm Selection, Cumulative Returns."""
        # Check if results exist and have data
        if not self.results['cumulative_returns'] or len(self.results['cumulative_returns']) == 0:
             print("No results to plot. Run the simulation first.")
             return

        print("Plotting results...")
        plt.figure(figsize=figsize)
        num_plots = 3
        plot_index = 1

        # --- 1. Cumulative Regret Plot ---
        plt.subplot(num_plots, 1, plot_index)
        # Calculate cumulative sum of per-step regret for plotting
        cumulative_regret_plot = np.cumsum(self.results['cumulative_regret'])
        plt.plot(cumulative_regret_plot, color='red', label='Cumulative Regret')
        plt.title('Cumulative Regret Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Regret')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plot_index += 1

        # --- 2. Arm Selection Histogram ---
        plt.subplot(num_plots, 1, plot_index)
        if self.results['chosen_arms']: # Check if list is not empty
            arm_counts = pd.Series(self.results['chosen_arms']).value_counts().sort_index()
            # Ensure all arms are represented, even if count is 0
            arm_counts = arm_counts.reindex(range(self.num_strategies), fill_value=0)
            # Create labels with strategy names
            strategy_names = []
            for i, s in enumerate(self.strategies):
                 # Shorten long composite names if necessary
                 name = s.name if len(s.name) < 30 else s.name[:27] + '...'
                 strategy_names.append(f"S{i}: {name}")
            # Ensure strategy_names has the same length as the number of strategies
            if len(strategy_names) == len(arm_counts):
                arm_counts.index = strategy_names
            else:
                 print("Warning: Mismatch between strategy names and arm counts length.")

            arm_counts.plot(kind='bar', color='steelblue', alpha=0.9)
            plt.title('Arm (Strategy) Selection Frequency')
            plt.xlabel('Strategy')
            plt.ylabel('Number of Times Chosen')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
             plt.text(0.5, 0.5, 'No arms were chosen (chosen_arms list is empty).', ha='center', va='center')
             plt.title('Arm (Strategy) Selection Frequency')
             plt.xlabel('Strategy')
             plt.ylabel('Number of Times Chosen')
        plot_index += 1

        # --- 3. Cumulative Returns Comparison ---
        plt.subplot(num_plots, 1, plot_index)
        



        time_steps = range(len(self.results['cumulative_returns'])) # X-axis for plots

        # Plot MAB policy returns
        plt.plot(time_steps, self.results['cumulative_returns'], color='green', label='MAB Strategy Returns')

        # Optional: Plot cumulative returns of the best single strategy in hindsight
        if self.results['optimal_rewards']:
             optimal_rewards_cumsum = np.cumsum(self.results['optimal_rewards'])
             plt.plot(time_steps, optimal_rewards_cumsum, color='orange', linestyle='--', label='Optimal Fixed Strategy (Hindsight)')

        # Optional: Plot buy-and-hold benchmark
        try:
            # Ensure benchmark calculation aligns with simulation length
            sim_length = len(self.results['cumulative_returns'])
            buy_hold_pct = self.df['Price'].iloc[1:sim_length+1].pct_change().fillna(0) # Match length and period
            buy_hold_returns = buy_hold_pct.cumsum()
            if len(buy_hold_returns) == sim_length:
                 plt.plot(time_steps, buy_hold_returns.values, color='purple', linestyle=':', label='Buy & Hold')
            else:
                 print(f"Warning: Buy & Hold length ({len(buy_hold_returns)}) doesn't match simulation length ({sim_length}). Skipping plot.")
        except Exception as e:
            print(f"Could not plot Buy & Hold benchmark: {e}")


        plt.title('Cumulative Returns Over Time')
        plt.xlabel('Time Steps (since start of simulation)')
        plt.ylabel('Cumulative Return')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plot_index += 1

        plt.tight_layout() # Adjust layout to prevent overlap
        plt.show()
        print("Plotting finished.")