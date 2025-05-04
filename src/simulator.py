# /Users/mohdtahaabbas/StratArm/src/simulator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from .strategy import TradingStrategy
# --- Import MAB algorithm classes for type checking ---
from .bandit_algorithms import MABAlgorithm, UCB, KL_UCB, SW_UCB, EWMA_UCB

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

        # Check for NaNs *after* potential feature calculation drops
        if df.isnull().values.any():
             print("Warning: DataFrame df contains NaNs AFTER feature calculation/dropna. This might indicate issues upstream.")
             # Consider adding more robust handling or raising an error depending on severity

        self.df = df.reset_index(drop=True) # Ensure index is 0-based sequential
        self.strategies = strategies
        self.mab = mab
        self.num_strategies = len(strategies)

        if self.num_strategies != self.mab.num_arms:
            raise ValueError(f"Number of strategies ({self.num_strategies}) must match MAB algorithm's num_arms ({self.mab.num_arms}).")

        # Initialize results dictionary correctly
        self.results = {
            'cumulative_returns': [], # MAB policy's cumulative returns (using RAW rewards)
            'cumulative_regret': [],  # Stores PER-STEP regret (plot uses cumsum)
            'chosen_arms': [],        # Index of the arm chosen at each step
            'actual_rewards': [],     # RAW reward obtained by the chosen arm at each step
            'optimal_rewards': [],    # Best possible RAW reward at each step
            'all_strategy_rewards': [] # List of dicts {arm_index: RAW reward} per step (optional to store)
        }

    @staticmethod
    def _min_max_scale(reward, min_r, max_r):
        """
        Scales a reward to [0, 1] using min-max normalization.
        Static method accessible within the class.
        """
        if min_r is None or max_r is None:
            print("Warning: min_r or max_r is None for scaling. Clipping reward to [0, 1].")
            return np.clip(reward, 0.0, 1.0)
        if max_r == min_r:
            # Handle case where min and max are the same (e.g., zero volatility)
            # Return 0.5 as a neutral value, or clip based on position relative to min/max
            print(f"Warning: min_r ({min_r}) == max_r ({max_r}) for scaling. Returning 0.5.")
            return 0.5 # Return midpoint as neutral scaled value
            # Alternative: return 0.0 if reward <= min_r else 1.0
            # return 0.0 if reward <= min_r else 1.0
        try:
            scaled = (reward - min_r) / (max_r - min_r)
            return np.clip(scaled, 0.0, 1.0) # Clip to ensure result is strictly within [0, 1]
        except Exception as e: # Catch potential numerical issues broader than ZeroDivisionError
            print(f"Error during scaling calculation: {e}. Clipping reward to [0, 1].")
            return np.clip(reward, 0.0, 1.0)


    def run(self) -> dict:
        """Execute the bandit simulation with pre-calculated signals and correct reward/regret."""
        print("Starting simulation...")

        # --- Calculate percentage returns (handle potential inf values) ---
        if 'Price' not in self.df.columns:
             raise ValueError("DataFrame must contain a 'Price' column.")
        price_returns = self.df['Price'].pct_change()
        price_returns.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf/-inf
        price_returns = price_returns.fillna(0) # Fill initial NaN

        # --- Pre-calculate all signals for all strategies ---
        print("Pre-calculating strategy signals...")
        all_signals = pd.DataFrame(index=self.df.index)
        for i, strat in enumerate(self.strategies):
            try:
                print(f"  Generating signals for strategy {i}: {strat.name}...")
                signals = strat.generate_signals(self.df) # Assumes shifted signal Series
                # Verify the output type and index
                if not isinstance(signals, pd.Series):
                     raise TypeError(f"Strategy {strat.name} did not return a pandas Series.")
                if not signals.index.equals(self.df.index):
                     print(f"Warning: Index mismatch for strategy {strat.name}. Reindexing.")
                     signals = signals.reindex(self.df.index) # Ensure alignment

                all_signals[i] = signals # Assign to DataFrame column i

            except Exception as e:
                 print(f"Error generating signals for strategy {i} ({strat.name}): {e}")
                 import traceback
                 traceback.print_exc()
                 raise

        # Check for NaNs in signals *after* the first row (which is expected NaN due to shift)
        if all_signals.iloc[1:].isnull().values.any():
              print("Warning: NaNs detected in signals after the first time step. Check strategy logic or warmup periods.")


        print("Running simulation loop...")
        horizon = len(self.df) - 1 # Number of decision steps
        cumulative_return_value = 0.0

        # Loop starts from t=1 (second row, index 1)
        for t in range(1, len(self.df)):
            current_return = price_returns.iloc[t] # Return from t-1 to t

            # --- Calculate potential RAW rewards for all strategies at time t ---
            strategy_rewards = {}
            for i in range(self.num_strategies):
                signal = all_signals[i].iloc[t] # Signal decided based on data up to t-1

                if pd.isna(signal):
                    reward = 0.0 # Neutral reward for NaN signal
                elif signal == 1:    # Buy signal from t-1 -> hold until t
                    reward = current_return
                elif signal == -1: # Sell signal from t-1 -> hold short until t
                    reward = -current_return
                else:              # Hold signal (signal == 0)
                    reward = 0.0

                strategy_rewards[i] = reward

            # --- Determine optimal RAW reward and calculate regret ---
            if not strategy_rewards:
                 optimal_reward = 0.0
            else:
                 # Filter out potential NaN rewards (shouldn't happen with current logic but safe)
                 valid_rewards = [r for r in strategy_rewards.values() if pd.notna(r)]
                 optimal_reward = max(valid_rewards) if valid_rewards else 0.0

            # --- MAB Algorithm pulls an arm ---
            try:
                 arm = self.mab.give_pull()
                 if not isinstance(arm, (int, np.integer)) or arm < 0 or arm >= self.num_strategies:
                     print(f"Warning: MAB returned invalid arm {arm}. Defaulting to arm 0.")
                     arm = 0
            except Exception as e:
                 print(f"Error during MAB give_pull() at step {t}: {e}. Defaulting to arm 0.")
                 arm = 0

            # --- Get the RAW reward for the chosen arm ---
            actual_reward = strategy_rewards.get(arm, 0.0) # Default to 0.0 if arm somehow invalid

            # --- Append RAW results for this time step ---
            self.results['optimal_rewards'].append(optimal_reward)
            self.results['actual_rewards'].append(actual_reward) # Store raw reward for performance
            self.results['chosen_arms'].append(arm)
            # Optionally store all strategy rewards for analysis (can consume memory)
            # self.results['all_strategy_rewards'].append(strategy_rewards.copy())

            # --- Calculate and store step regret (based on RAW rewards) ---
            step_regret = optimal_reward - actual_reward
            # Ensure non-negative due to potential float issues before cumsum
            self.results['cumulative_regret'].append(max(0.0, step_regret))

            # ---- START DEBUG PRINTS for negative regret ----
            # if step_regret < -1e-9: # Allow for tiny float inaccuracies
            #     print(f"\n--- NEGATIVE REGRET DETECTED at t={t} ---")
            #     print(f"  Current Return: {current_return:.6f}")
            #     signals_this_step = {i: all_signals[i].iloc[t] for i in range(self.num_strategies)}
            #     print(f"  Signals this Step: {signals_this_step}")
            #     print(f"  Strategy Rewards: {strategy_rewards}")
            #     print(f"  Optimal Reward: {optimal_reward:.6f}")
            #     print(f"  Chosen Arm: {arm}")
            #     print(f"  Actual Reward: {actual_reward:.6f}")
            #     print(f"  Step Regret: {step_regret:.6f} (SHOULD BE >= 0)")
            #     print(f"------------------------------------------")
            # ---- END DEBUG PRINTS ----


            # --- V V V --- REWARD SCALING FOR MAB UPDATE --- V V V ---
            # Determine the reward to pass to the MAB's get_reward method
            scaled_reward_for_mab = actual_reward # Default to raw reward

            # Check if the current MAB algorithm is one that requires scaling
            if isinstance(self.mab, (UCB, KL_UCB, SW_UCB, EWMA_UCB)):
                # Check if the necessary scaling parameters exist on the MAB instance
                if hasattr(self.mab, 'min_reward') and hasattr(self.mab, 'max_reward') and \
                   self.mab.min_reward is not None and self.mab.max_reward is not None:
                    try:
                        # Use the static scaling method defined in this class
                        scaled_reward_for_mab = Simulator._min_max_scale(
                            actual_reward, self.mab.min_reward, self.mab.max_reward
                        )
                        # --- Optional Debug Print for Scaling ---
                        # if t % 100 == 0: # Print scaling periodically
                        #      print(f"t={t}: Scaling for {self.mab.__class__.__name__}. Raw={actual_reward:.4f}, Scaled={scaled_reward_for_mab:.4f} (Min={self.mab.min_reward}, Max={self.mab.max_reward})")
                        # --- End Optional Debug ---
                    except Exception as e:
                        print(f"Error during reward scaling at step {t}: {e}. Using raw reward for MAB update.")
                        # Fallback to raw reward if scaling fails unexpectedly
                        scaled_reward_for_mab = actual_reward
                else:
                    # Warn if scaling parameters are missing for an algorithm that needs them
                    print(f"Warning at step {t}: MAB {self.mab.__class__.__name__} likely requires scaling, "
                          "but min/max_reward not properly set during MAB initialization. "
                          "Using raw reward ({actual_reward:.4f}) for MAB update.")
                    scaled_reward_for_mab = actual_reward # Use raw as fallback

            # --- ^ ^ ^ --- END REWARD SCALING --- ^ ^ ^ ---


            # --- Update MAB Algorithm with the correctly scaled (or raw) reward ---
            try:
                # Pass the potentially scaled reward to the MAB
                self.mab.get_reward(arm, scaled_reward_for_mab)
            except Exception as e:
                 print(f"Error during MAB get_reward() for arm {arm} at step {t}: {e}")
                 # Decide how to handle: stop simulation, skip MAB update for this step?

            # --- Update cumulative returns for the policy ---
            # IMPORTANT: Use the ORIGINAL, UNSCALED actual_reward for performance tracking
            cumulative_return_value += actual_reward
            self.results['cumulative_returns'].append(cumulative_return_value)

            # --- Progress Update ---
            if (t + 1) % 100 == 0 or t == horizon: # Print every 100 steps or at the end
                 print(f"  Step {t+1}/{horizon+1} completed. Current Cumulative Return: {cumulative_return_value:.4f}")


        print("Simulation finished.")
        return self.results

    def plot_results(self, figsize=(15, 12)):
        """Visualize simulation results: Cumulative Regret, Arm Selection, Cumulative Returns."""
        if not self.results or not self.results.get('cumulative_returns'):
             print("No results to plot. Run the simulation first.")
             return

        print("Plotting results...")
        num_steps = len(self.results['cumulative_returns'])
        if num_steps == 0:
             print("Results dictionary is present but contains no data points. Cannot plot.")
             return

        plt.figure(figsize=figsize)
        num_plots = 3
        plot_index = 1
        time_steps = range(num_steps) # X-axis based on actual simulation length

        # --- 1. Cumulative Regret Plot ---
        plt.subplot(num_plots, 1, plot_index)
        # Calculate cumulative sum of per-step regret for plotting
        if self.results['cumulative_regret']:
            cumulative_regret_plot = np.cumsum(self.results['cumulative_regret'])
            if len(cumulative_regret_plot) == num_steps:
                plt.plot(time_steps, cumulative_regret_plot, color='red', label='Cumulative Regret')
                plt.title('Cumulative Regret Over Time')
                plt.xlabel('Time Steps')
                plt.ylabel('Cumulative Regret')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
            else:
                 plt.text(0.5, 0.5, f'Length mismatch: Regret ({len(cumulative_regret_plot)}) vs Steps ({num_steps})', ha='center', va='center')
                 plt.title('Cumulative Regret Over Time - Data Error')
        else:
             plt.text(0.5, 0.5, 'No regret data available.', ha='center', va='center')
             plt.title('Cumulative Regret Over Time')
        plot_index += 1

        # --- 2. Arm Selection Histogram ---
        plt.subplot(num_plots, 1, plot_index)
        if self.results['chosen_arms']:
            arm_counts = pd.Series(self.results['chosen_arms']).value_counts().sort_index()
            arm_counts = arm_counts.reindex(range(self.num_strategies), fill_value=0)
            strategy_names = [f"S{i}: {s.name[:30]}" for i, s in enumerate(self.strategies)] # Truncate long names
            if len(strategy_names) == len(arm_counts):
                arm_counts.index = strategy_names
            else: print("Warning: Mismatch between strategy names and arm counts.")

            arm_counts.plot(kind='bar', color='steelblue', alpha=0.9)
            plt.title('Arm (Strategy) Selection Frequency')
            plt.xlabel('Strategy')
            plt.ylabel('Number of Times Chosen')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
             plt.text(0.5, 0.5, 'No arms were chosen.', ha='center', va='center')
             plt.title('Arm (Strategy) Selection Frequency')
             plt.xlabel('Strategy')
             plt.ylabel('Number of Times Chosen')
        plot_index += 1

        # --- 3. Cumulative Returns Comparison ---
        plt.subplot(num_plots, 1, plot_index)
        # Plot MAB policy returns
        plt.plot(time_steps, self.results['cumulative_returns'], color='green', label='MAB Strategy Returns')

        # Optional: Plot cumulative returns of the best single strategy in hindsight
        if self.results['optimal_rewards'] and len(self.results['optimal_rewards']) == num_steps:
             optimal_rewards_cumsum = np.cumsum(self.results['optimal_rewards'])
             plt.plot(time_steps, optimal_rewards_cumsum, color='orange', linestyle='--', label='Optimal Fixed Strategy (Hindsight)')

        # Optional: Plot buy-and-hold benchmark
        # Ensure benchmark calculation aligns with the simulation period length (num_steps)
        # The simulation starts at index 1 of the original df for returns calc.
        # So, the benchmark should cover the same period.
        try:
            # Calculate B&H return over the *simulated* period
            # Price at start of sim (end of df index 0) to price at end of sim (df index num_steps)
            start_price = self.df['Price'].iloc[0] # Price at the time of the first decision's *basis*
            end_price = self.df['Price'].iloc[num_steps] # Price at the end of the last simulated step
            if start_price != 0 and pd.notna(start_price) and pd.notna(end_price):
                # Simple B&H: Total return (not compounded daily for this plot)
                # More comparable: calculate cumulative daily pct_change over the sim period
                # price_returns starts at index 1, corresponding to time_steps[0]
                buy_hold_daily_returns = price_returns.iloc[1:num_steps+1] # Match length
                if len(buy_hold_daily_returns) == num_steps:
                     buy_hold_cumulative = buy_hold_daily_returns.cumsum()
                     plt.plot(time_steps, buy_hold_cumulative.values, color='purple', linestyle=':', label='Buy & Hold (Daily Cumsum)')
                else:
                     print(f"Warning: Buy & Hold length mismatch. Skipping plot.")
            else:
                print("Warning: Cannot calculate Buy & Hold benchmark due to zero or NaN start/end price.")
        except IndexError:
            print(f"Warning: DataFrame too short to calculate Buy & Hold benchmark over {num_steps} steps.")
        except Exception as e:
            print(f"Could not plot Buy & Hold benchmark: {e}")

        plt.title('Cumulative Returns Over Time')
        plt.xlabel(f'Time Steps (Simulation ran for {num_steps} steps)')
        plt.ylabel('Cumulative Return')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plot_index += 1

        plt.tight_layout() # Adjust layout
        plt.show()
        print("Plotting finished.")