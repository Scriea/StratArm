# --- compare_mab_algorithms_final_v2.py --- # Added version marker

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time # To time the simulations
import traceback # For detailed error printing
import os # For path joining

# --- Import your existing classes ---
# Ensure the 'src' directory is accessible from where you run this script
# Option 1: Run script from the parent directory of 'src'
# Option 2: Add the parent directory to PYTHONPATH
# Option 3: Adjust the import path below (e.g., if script is inside 'src')
try:
    # Assuming these classes are defined in the specified structure
    # If not, replace with dummy classes or comment out relevant parts
    # Example Dummy Class:
    # class TradingStrategy:
    #     def __init__(self, name="Dummy"): self.name = name
    #     def generate_signals(self, df): return pd.Series(0, index=df.index) # Returns neutral signal

    # class SMACrossoverStrategy(TradingStrategy):
    #     def __init__(self, short_window=10, long_window=20): super().__init__(f"SMA({short_window},{long_window})")
    # class RSIStrategy(TradingStrategy):
    #     def __init__(self, rsi_window=14, low=30, high=70): super().__init__(f"RSI({rsi_window},{low},{high})")
    # class BollingerStrategy(TradingStrategy):
    #     def __init__(self, window=20, num_std=2): super().__init__(f"BB({window},{num_std})")
    # class CompositeStrategy(TradingStrategy):
    #     def __init__(self, strategies): super().__init__("Composite")
    #         # Basic composite logic: Majority vote or similar
    #         # This needs a proper implementation matching your actual class
    #         self.strategies = strategies
    #     def generate_signals(self, df):
    #         all_signals = pd.DataFrame({s.name: s.generate_signals(df) for s in self.strategies})
    #         # Example: take the mode (most frequent signal) across strategies for each step
    #         return all_signals.mode(axis=1)[0].fillna(0) # Fill NaNs if no majority

    # class BaseBanditAlgorithm:
    #     def __init__(self, num_arms): self.num_arms = num_arms; self.reset()
    #     def choose_arm(self): return np.random.randint(self.num_arms)
    #     def update(self, chosen_arm, reward): pass
    #     def reset(self): pass

    # class Eps_Greedy(BaseBanditAlgorithm):
    #     def __init__(self, num_arms, epsilon=0.1): super().__init__(num_arms); self.epsilon=epsilon
    # class UCB(BaseBanditAlgorithm):
    #     def __init__(self, num_arms, c=2.0, min_reward=0, max_reward=1): super().__init__(num_arms); self.c=c
    # class SW_UCB(BaseBanditAlgorithm):
    #      def __init__(self, num_arms, window_size=100, c=2.0, min_reward=0, max_reward=1): super().__init__(num_arms); self.window_size=window_size; self.c=c
    # class EWMA_UCB(BaseBanditAlgorithm):
    #     def __init__(self, num_arms, lambda_decay=0.99, c=2.0, min_reward=0, max_reward=1): super().__init__(num_arms); self.lambda_decay=lambda_decay; self.c=c
    # class GaussianThompsonSampling(BaseBanditAlgorithm): pass

    # class Simulator:
    #     def __init__(self, df, strategies, mab):
    #         self.df = df
    #         self.strategies = strategies
    #         self.mab = mab
    #         self.num_arms = len(strategies)
    #         # Precompute signals if needed by your simulator structure
    #         self.signals = {s.name: s.generate_signals(df) for s in strategies}
    #         self.price_returns = df['Price'].pct_change().fillna(0)

    #     def run(self):
    #         sim_length = len(self.df) - 1
    #         cumulative_returns = [0.0] * sim_length
    #         cumulative_regret = [0.0] * sim_length
    #         chosen_arms = [0] * sim_length
    #         # Dummy implementation - replace with your actual logic
    #         optimal_returns = self.price_returns.abs()[1:sim_length+1] # Example: Perfect foresight return
    #         current_cumulative_return = 0.0
    #         current_cumulative_regret = 0.0

    #         for t in range(1, sim_length + 1):
    #             chosen_arm_index = self.mab.choose_arm()
    #             chosen_strategy = self.strategies[chosen_arm_index]
    #             signal = self.signals[chosen_strategy.name].iloc[t]
    #             market_return = self.price_returns.iloc[t]

    #             step_return = 0.0
    #             if pd.notna(signal):
    #                 if signal == 1: step_return = market_return
    #                 elif signal == -1: step_return = -market_return

    #             # Reward scaling (example)
    #             reward = np.clip((step_return - ESTIMATED_MIN_REWARD) / (ESTIMATED_MAX_REWARD - ESTIMATED_MIN_REWARD), 0, 1)
    #             self.mab.update(chosen_arm_index, reward) # Use scaled reward for MAB update

    #             current_cumulative_return += step_return
    #             cumulative_returns[t-1] = current_cumulative_return
    #             chosen_arms[t-1] = chosen_arm_index

    #             # Regret Calculation (Example: vs perfect foresight)
    #             optimal_reward = optimal_returns.iloc[t-1] # Indexing adjusted
    #             regret = optimal_reward - step_return
    #             current_cumulative_regret += regret
    #             cumulative_regret[t-1] = current_cumulative_regret

    #         return {
    #             'cumulative_returns': cumulative_returns,
    #             'cumulative_regret': cumulative_regret,
    #             'chosen_arms': chosen_arms
    #         }

    # def preprocess_data(df):
    #     # Replace with your actual preprocessing
    #     df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    #     df.set_index('Date', inplace=True)
    #     df['Price'] = pd.to_numeric(df['Price'].str.replace(',', ''), errors='coerce')
    #     df = df[['Price']].dropna()
    #     return df

    # def calculate_features(df):
    #      # Replace with your actual feature calculation (SMA, RSI, Bollinger Bands etc.)
    #      # Ensure columns needed by strategies (e.g., 'SMA_short', 'SMA_long', 'RSI', 'BB_upper', 'BB_lower') exist
    #      df['SMA_15'] = df['Price'].rolling(15).mean()
    #      df['SMA_30'] = df['Price'].rolling(30).mean()
    #      delta = df['Price'].diff()
    #      gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
    #      loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
    #      rs = gain / loss
    #      df['RSI_9'] = 100 - (100 / (1 + rs))
    #      df['BB_Middle'] = df['Price'].rolling(window=10).mean()
    #      df['BB_Std'] = df['Price'].rolling(window=10).std()
    #      df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 1.5)
    #      df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 1.5)
    #      return df.dropna() # Drop NaNs created by rolling windows

    # --- If using actual imports, use this block ---
    from src.simulator import Simulator
    from src.strategy import TradingStrategy, SMACrossoverStrategy, RSIStrategy, BollingerStrategy, CompositeStrategy
    from src.bandit_algorithms import Eps_Greedy, UCB, KL_UCB, GaussianThompsonSampling, SW_UCB, EWMA_UCB
    from src.data_utils import preprocess_data, calculate_features
    # -----------------------------------------------

except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the 'src' directory/classes are correctly structured and accessible, or use dummy classes for testing.")
    exit()

# --- Configuration ---
# Use os.path.join for better cross-platform compatibility
DATA_DIR = 'data' # Specify directory containing data
DATA_FILENAME = 'BSE_Sensex_30_Historical_Data_2021_2024.csv' # Specify filename
DATA_FILE = os.path.join(DATA_DIR, DATA_FILENAME)

N_RUNS_MAB = 10 # Number of simulation runs per MAB algorithm
HORIZON = None # Set to a specific number of steps if needed

# MAB Reward Scaling Configuration
ESTIMATED_MIN_REWARD = -0.05 # <<< --- Adjust: Estimated minimum single-step return
ESTIMATED_MAX_REWARD = 0.05  # <<< --- Adjust: Estimated maximum single-step return

# --- Define Base Trading Strategies ---
base_strategies: list[TradingStrategy] = [
    SMACrossoverStrategy(short_window=15, long_window=30),
    RSIStrategy(rsi_window=9, low=40, high=90),
    BollingerStrategy(window=10, num_std=1.5)
]

# --- Create Composite Strategy ---
composite_strategy = CompositeStrategy(strategies=base_strategies)

# --- Full List of Strategies to Simulate as Pure ---
strategies_to_simulate_pure: list[TradingStrategy] = base_strategies + [composite_strategy]
num_pure_strategies = len(strategies_to_simulate_pure)
pure_strategy_names = [s.name for s in strategies_to_simulate_pure]

# --- Strategies for the MAB to Choose From ---
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! MODIFICATION HERE: Include Composite Strategy !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mab_chooses_from_strategies: list[TradingStrategy] = base_strategies + [composite_strategy]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

num_mab_arms = len(mab_chooses_from_strategies) # This will now correctly reflect the number of arms (e.g., 4)
mab_arm_strategy_names = [s.name for s in mab_chooses_from_strategies] # Names for plot legend (will include Composite)

# --- Define MAB Algorithms to Compare ---
# Ensure keys are unique and descriptive
# The num_arms argument will now use the updated count including the composite strategy
algorithms_to_test = {
    "Eps-Greedy(0.1)": Eps_Greedy(num_arms=num_mab_arms, epsilon=0.1),
    "UCB(c=2)": UCB(num_arms=num_mab_arms, c=2.0,
                    min_reward=ESTIMATED_MIN_REWARD, max_reward=ESTIMATED_MAX_REWARD),
    "SW-UCB(W=100)": SW_UCB(num_arms=num_mab_arms, window_size=100, c=2.0,
                           min_reward=ESTIMATED_MIN_REWARD, max_reward=ESTIMATED_MAX_REWARD),
    "EWMA-UCB(L=0.99)": EWMA_UCB(num_arms=num_mab_arms, lambda_decay=0.99, c=2.0,
                                min_reward=ESTIMATED_MIN_REWARD, max_reward=ESTIMATED_MAX_REWARD),
    "GaussTS": GaussianThompsonSampling(num_arms=num_mab_arms) # Shorter name
}
mab_algo_names = list(algorithms_to_test.keys())

# --- Load and Prepare Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
    # Check if file exists before reading
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(f"Specified path is not a file: {DATA_FILE}")

    raw_df = pd.read_csv(DATA_FILE)
    df_processed = preprocess_data(raw_df.copy())
    df_features = calculate_features(df_processed.copy()) # This drops leading NaNs

    if df_features.empty:
        raise ValueError("DataFrame is empty after preprocessing and feature calculation. Check data and functions.")
    if 'Price' not in df_features.columns:
         raise ValueError("'Price' column not found after feature calculation. Ensure it's preserved or recalculated.")

    market_prices = df_features['Price'].copy()
    # Handle potential DatetimeIndex vs. 'Date' column
    if isinstance(df_features.index, pd.DatetimeIndex):
        market_dates = df_features.index
    elif 'Date' in df_features.columns:
        # Ensure 'Date' is datetime if it exists and is not the index
        if not pd.api.types.is_datetime64_any_dtype(df_features['Date']):
            try:
                df_features['Date'] = pd.to_datetime(df_features['Date'])
            except Exception as date_err:
                print(f"Warning: Could not convert 'Date' column to datetime: {date_err}. Using index for plotting.")
                market_dates = df_features.index # Fallback to index
        market_dates = df_features['Date']
    else:
        # Fallback if no date info available (will affect market plot x-axis)
        print("Warning: No 'Date' column or DatetimeIndex found. Using integer index for market plot.")
        market_dates = df_features.index

    # Determine simulation length AFTER dropping NaNs in calculate_features
    sim_length = len(df_features) - 1 # -1 because returns/signals often start from the second row
    if sim_length <= 0:
        raise ValueError(f"Simulation length ({sim_length}) is not positive. Check data processing and NaN handling.")
    print(f"Data prepared. Simulation horizon: {sim_length} steps.")

    # Calculate percentage returns needed for simulation
    # Ensure this is calculated on the final df_features used for simulation
    price_returns = df_features['Price'].pct_change()
    price_returns.replace([np.inf, -np.inf], 0, inplace=True)
    price_returns = price_returns.fillna(0) # Fill first NaN (pct_change introduces one)

    # --- Input Data Validation ---
    if not isinstance(price_returns, pd.Series):
        raise TypeError("price_returns is not a Pandas Series after calculation.")
    if price_returns.isnull().any():
        print("Warning: NaNs found in price_returns after fillna(0).")
    if len(price_returns) != len(df_features):
         raise ValueError(f"Length mismatch: price_returns ({len(price_returns)}) vs df_features ({len(df_features)})")


except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except ValueError as e:
     print(f"ValueError during data preparation: {e}")
     traceback.print_exc()
     exit()
except TypeError as e:
    print(f"TypeError during data preparation: {e}")
    traceback.print_exc()
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data preparation: {e}")
    traceback.print_exc()
    exit()

# --- Simulate Pure Strategies ---
print("\nSimulating Pure Strategies (Fixed Arm)...")
pure_strategy_results = {} # Store cumulative return series {name: array}
for strategy in strategies_to_simulate_pure:
    print(f"  Simulating: {strategy.name}...")
    try:
        # Ensure generate_signals uses the same dataframe as the simulation
        signals = strategy.generate_signals(df_features.copy())

        # Validate signals
        if not isinstance(signals, pd.Series):
             raise TypeError(f"generate_signals for {strategy.name} did not return a Pandas Series.")
        if signals.isnull().any():
             print(f"    Warning: NaNs generated by generate_signals for {strategy.name}. Filling with 0.")
             signals = signals.fillna(0) # Handle potential NaNs from strategy logic

        # Align signals index with df_features index if necessary (e.g., if strategy drops extra rows)
        if not signals.index.equals(df_features.index):
             print(f"    Warning: Index mismatch for {strategy.name} signals. Reindexing...")
             signals = signals.reindex(df_features.index, fill_value=0) # Reindex and fill missing signals

        cumulative_return = 0.0
        returns_over_time = np.zeros(sim_length)

        # Start loop from 1 as index 0 has no prior data for return/signals
        for t in range(1, sim_length + 1):
            # Use .iloc for positional access corresponding to simulation step t
            signal_t = signals.iloc[t]
            market_return_t = price_returns.iloc[t]

            step_return = 0.0
            # Ensure signal is valid before calculating return
            if pd.notna(signal_t):
                if signal_t == 1: step_return = market_return_t
                elif signal_t == -1: step_return = -market_return_t
                #elif signal_t == 0: step_return = 0.0 # Hold
            else:
                 # Should not happen if NaNs were filled, but good to have a failsafe
                 print(f"    Warning: NaN signal encountered at step {t} for {strategy.name} despite checks.")

            # Check for non-finite returns
            if not np.isfinite(step_return):
                print(f"    Warning: Non-finite step_return ({step_return}) at step {t} for {strategy.name}. Setting to 0.")
                step_return = 0.0

            cumulative_return += step_return
            # Store in array index corresponding to step number (0 to sim_length-1)
            returns_over_time[t-1] = cumulative_return

        pure_strategy_results[strategy.name] = returns_over_time
        print(f"    Finished. Final Cumulative Return: {cumulative_return:.4f}")
    except TypeError as e:
        print(f"    TypeError during pure strategy simulation for {strategy.name}: {e}")
        traceback.print_exc()
        pure_strategy_results[strategy.name] = np.full(sim_length, np.nan)
    except IndexError as e:
        print(f"    IndexError during pure strategy simulation for {strategy.name}: {e}")
        print(f"    Check alignment of signals/returns with sim_length ({sim_length}) and DataFrame length ({len(df_features)}).")
        traceback.print_exc()
        pure_strategy_results[strategy.name] = np.full(sim_length, np.nan)
    except Exception as e:
        print(f"    ERROR during pure strategy simulation for {strategy.name}: {e}")
        traceback.print_exc()
        pure_strategy_results[strategy.name] = np.full(sim_length, np.nan)


# --- Run MAB Simulations ---
mab_results_storage = {name: {'regrets': [], 'returns_final': [], 'choices': [], 'returns_over_time': []} for name in mab_algo_names}
start_time = time.time()
print(f"\nStarting {N_RUNS_MAB} MAB simulation runs for each of {len(mab_algo_names)} algorithms...")
print(f"MABs will choose from {num_mab_arms} strategies: {mab_arm_strategy_names}") # Verify the correct arms are used

for run in range(N_RUNS_MAB):
    print(f"\n--- MAB Run {run + 1}/{N_RUNS_MAB} ---")
    for algo_name, mab_agent in algorithms_to_test.items():
        # print(f"  Simulating MAB: {algo_name}...") # Optional: Reduce verbosity inside run loop
        mab_agent.reset() # Reset agent state for each run

        # Pass the list of strategies the MAB should choose from
        simulator = Simulator(df=df_features.copy(), strategies=mab_chooses_from_strategies, mab=mab_agent)
        try:
            # Ensure the simulator's run method handles the simulation length correctly
            run_results = simulator.run() # run() should return a dict with results

            # Validate the structure and content of run_results
            if not isinstance(run_results, dict):
                 print(f"    ERROR: Simulator.run() for {algo_name} did not return a dictionary.")
                 raise TypeError("Invalid return type from simulator.")

            returns_cum = run_results.get('cumulative_returns')
            regret_cum = run_results.get('cumulative_regret') # Assuming regret is cumulative in results
            choices = run_results.get('chosen_arms')

            # Convert to numpy arrays and basic validation
            if returns_cum is None or choices is None or regret_cum is None:
                 print(f"    ERROR: Missing keys in results dict for {algo_name} on run {run + 1}. Keys: {run_results.keys()}")
                 raise ValueError("Missing results data.")

            returns_cum_np = np.array(returns_cum)
            regret_cum_np = np.array(regret_cum)
            choices_np = np.array(choices)

            # Rigorous length check against sim_length
            if len(returns_cum_np) == sim_length and \
               len(regret_cum_np) == sim_length and \
               len(choices_np) == sim_length:

                # Check for NaNs or Infs in critical results before storing
                if np.isnan(returns_cum_np).any() or not np.isfinite(returns_cum_np[-1]):
                     print(f"    WARNING: NaNs or non-finite values found in cumulative returns for {algo_name} on run {run + 1}. Skipping storage.")
                     raise ValueError("Invalid return values.")
                if np.isnan(regret_cum_np).any() or not np.isfinite(regret_cum_np[-1]):
                    print(f"    WARNING: NaNs or non-finite values found in cumulative regret for {algo_name} on run {run + 1}. Skipping storage.")
                    raise ValueError("Invalid regret values.")

                # Store validated results
                mab_results_storage[algo_name]['returns_over_time'].append(returns_cum_np)
                mab_results_storage[algo_name]['returns_final'].append(returns_cum_np[-1])
                mab_results_storage[algo_name]['regrets'].append(regret_cum_np) # Store cumulative regret directly
                mab_results_storage[algo_name]['choices'].append(choices_np)

            else:
                print(f"    WARNING: Inconsistent results length for {algo_name} on run {run + 1}.")
                print(f"      Expected length: {sim_length}")
                print(f"      Actual lengths: returns={len(returns_cum_np)}, regret={len(regret_cum_np)}, choices={len(choices_np)}")
                # Append NaNs/empty to maintain structure for aggregation checks, but log the error
                mab_results_storage[algo_name]['returns_over_time'].append(np.full(sim_length, np.nan))
                mab_results_storage[algo_name]['returns_final'].append(np.nan)
                mab_results_storage[algo_name]['regrets'].append(np.full(sim_length, np.nan))
                mab_results_storage[algo_name]['choices'].append(np.array([]))


        except (TypeError, ValueError, IndexError) as sim_err: # Catch specific simulation errors
            print(f"    ERROR during MAB simulation for {algo_name} on run {run + 1}: {sim_err}")
            # traceback.print_exc() # Optional: more detail
            # Append NaNs/empty on error
            mab_results_storage[algo_name]['returns_over_time'].append(np.full(sim_length, np.nan))
            mab_results_storage[algo_name]['returns_final'].append(np.nan)
            mab_results_storage[algo_name]['regrets'].append(np.full(sim_length, np.nan))
            mab_results_storage[algo_name]['choices'].append(np.array([]))
        except Exception as e: # Catch any other unexpected errors
            print(f"    UNEXPECTED ERROR during MAB simulation for {algo_name} on run {run + 1}: {e}")
            traceback.print_exc()
            # Append NaNs/empty on error
            mab_results_storage[algo_name]['returns_over_time'].append(np.full(sim_length, np.nan))
            mab_results_storage[algo_name]['returns_final'].append(np.nan)
            mab_results_storage[algo_name]['regrets'].append(np.full(sim_length, np.nan))
            mab_results_storage[algo_name]['choices'].append(np.array([]))


end_time = time.time()
print(f"\nMAB simulations completed in {end_time - start_time:.2f} seconds.")

# --- Process and Aggregate MAB Results ---
print("Aggregating MAB results...")
processed_mab_results = {}
for algo_name in mab_algo_names:
    processed_mab_results[algo_name] = {} # Initialize dict for this algo

    # --- Filter valid runs based on final return AND expected length ---
    valid_run_indices = []
    for i in range(len(mab_results_storage[algo_name]['returns_final'])):
        # Check if final return is valid
        final_return_valid = pd.notna(mab_results_storage[algo_name]['returns_final'][i])
        # Check if the corresponding returns_over_time array has the correct length
        returns_over_time_valid = len(mab_results_storage[algo_name]['returns_over_time'][i]) == sim_length
        # Check if the corresponding regrets array has the correct length
        regrets_valid = len(mab_results_storage[algo_name]['regrets'][i]) == sim_length
        # Check if the corresponding choices array has the correct length
        choices_valid = len(mab_results_storage[algo_name]['choices'][i]) == sim_length

        if final_return_valid and returns_over_time_valid and regrets_valid and choices_valid:
             valid_run_indices.append(i)
        # else:
        #      print(f"Debug: Run {i+1} for {algo_name} excluded. FinalReturnValid={final_return_valid}, ReturnsLenValid={returns_over_time_valid}, RegretLenValid={regrets_valid}, ChoicesLenValid={choices_valid}")


    num_valid_runs = len(valid_run_indices)
    print(f"  {algo_name}: Found {num_valid_runs} valid runs out of {N_RUNS_MAB}.")

    if num_valid_runs == 0:
        print(f"  Warning: No valid runs found for {algo_name}. Cannot aggregate results.")
        # Fill with defaults/NaNs of correct expected shape
        processed_mab_results[algo_name] = {
            'mean_cum_regret': np.full(sim_length, np.nan), 'std_cum_regret': np.full(sim_length, np.nan),
            'mean_cum_return': np.full(sim_length, np.nan), 'std_cum_return': np.full(sim_length, np.nan),
            'mean_final_return': np.nan, 'std_final_return': np.nan,
            'choice_counts': np.zeros(num_mab_arms, dtype=int), # Ensure integer counts
            'all_final_returns': [],
            'num_valid_runs': 0
        }
        continue # Skip to next algorithm

    # --- Aggregate Cumulative Returns Over Time ---
    valid_returns_over_time = [mab_results_storage[algo_name]['returns_over_time'][i] for i in valid_run_indices]
    returns_stack = np.stack(valid_returns_over_time, axis=0) # Shape: (num_valid_runs, sim_length)
    processed_mab_results[algo_name]['mean_cum_return'] = np.mean(returns_stack, axis=0)
    processed_mab_results[algo_name]['std_cum_return'] = np.std(returns_stack, axis=0)

    # --- Aggregate Cumulative Regret Over Time ---
    valid_regrets = [mab_results_storage[algo_name]['regrets'][i] for i in valid_run_indices]
    regret_stack = np.stack(valid_regrets, axis=0) # Shape: (num_valid_runs, sim_length)
    # Assuming 'regrets' stored cumulative regret per run
    processed_mab_results[algo_name]['mean_cum_regret'] = np.mean(regret_stack, axis=0)
    processed_mab_results[algo_name]['std_cum_regret'] = np.std(regret_stack, axis=0)

    # --- Aggregate Final Returns ---
    valid_final_returns = [mab_results_storage[algo_name]['returns_final'][i] for i in valid_run_indices]
    processed_mab_results[algo_name]['mean_final_return'] = np.mean(valid_final_returns)
    processed_mab_results[algo_name]['std_final_return'] = np.std(valid_final_returns)
    processed_mab_results[algo_name]['all_final_returns'] = valid_final_returns # For box plot

    # --- Aggregate Choices ---
    valid_choices = [mab_results_storage[algo_name]['choices'][i] for i in valid_run_indices]
    # Concatenate choices from all valid runs, then count occurrences of each arm index
    if valid_choices: # Ensure there's something to concatenate
        all_choices_flat = np.concatenate(valid_choices)
        # Ensure choices are within the expected range [0, num_mab_arms-1]
        if all_choices_flat.max() >= num_mab_arms or all_choices_flat.min() < 0:
             print(f"    ERROR: Invalid arm index found in choices for {algo_name}. Max: {all_choices_flat.max()}, Min: {all_choices_flat.min()}, Num Arms: {num_mab_arms}")
             # Handle error appropriately, e.g., skip choice counting or filter invalid choices
             choice_counts = np.zeros(num_mab_arms, dtype=int) # Default to zeros on error
        else:
             choice_counts = np.bincount(all_choices_flat, minlength=num_mab_arms)
             # Ensure the output has the correct length
             if len(choice_counts) > num_mab_arms:
                 print(f"    Warning: bincount for {algo_name} produced more counts ({len(choice_counts)}) than arms ({num_mab_arms}). Truncating.")
                 choice_counts = choice_counts[:num_mab_arms]

    else:
        choice_counts = np.zeros(num_mab_arms, dtype=int) # No valid choices found

    processed_mab_results[algo_name]['choice_counts'] = choice_counts.astype(int) # Ensure integer type

    # Store number of valid runs used for aggregation
    processed_mab_results[algo_name]['num_valid_runs'] = num_valid_runs


# ============================================================================
# --- Plotting Code Starts Here ---
# ============================================================================
print("\nGenerating and saving individual plots...")

# --- Define Colors ---
# Ensure enough colors if many algos/strategies exist
mab_colors = plt.cm.viridis(np.linspace(0, 1, len(mab_algo_names)))
# Use tab10 or another colormap suitable for categorical data
# Ensure enough distinct colors for ALL strategies (pure + MAB arms) if needed elsewhere,
# Here specifically, strategy_colors are used for pure strategies and MAB arm choices.
# Make sure num_mab_arms matches the length needed for strategy colors in Plot 4.
num_colors_needed = max(num_pure_strategies, num_mab_arms)
if num_colors_needed <= 10:
    strategy_colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_colors_needed]
elif num_colors_needed <= 20:
     strategy_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_colors_needed]
else: # Fallback for many strategies
    strategy_colors = plt.cm.viridis(np.linspace(0, 0.9, num_colors_needed)) # Avoid yellow


# --- Plot 1: Cumulative Regret Comparison (MABs) ---
try:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    time_steps = np.arange(sim_length)
    plot_successful = False
    for i, algo_name in enumerate(mab_algo_names):
        # Check if results exist, are not all NaN, and have the correct length
        results = processed_mab_results.get(algo_name, {})
        mean_r = results.get('mean_cum_regret')
        std_r = results.get('std_cum_regret')

        if mean_r is not None and std_r is not None and \
           len(mean_r) == sim_length and len(std_r) == sim_length and \
           not np.isnan(mean_r).all():

            label_with_runs = f"{algo_name} (N={results.get('num_valid_runs', 0)})" # Add run count to label
            ax.plot(time_steps, mean_r, label=label_with_runs, color=mab_colors[i])
            ax.fill_between(time_steps, mean_r - std_r, mean_r + std_r, color=mab_colors[i], alpha=0.2)
            plot_successful = True
        elif algo_name in processed_mab_results: # Log if data exists but is invalid
             print(f"Plotting Warning (Regret): Invalid data or length mismatch for {algo_name}. Mean length: {len(mean_r) if mean_r is not None else 'None'}, Std length: {len(std_r) if std_r is not None else 'None'}, Expected: {sim_length}")


    if plot_successful:
        ax.set_title('MAB Cumulative Regret (Mean +/- StdDev across Valid Runs)')
        ax.set_xlabel(f'Time Steps ({sim_length} total)')
        ax.set_ylabel('Cumulative Regret')
        ax.legend(fontsize='medium', title="Algorithm (Valid Runs)")
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('mab_cumulative_regret.png', bbox_inches='tight')
        print("Saved mab_cumulative_regret.png")
    else:
        print("Skipping saving regret plot - no valid data plotted.")
    plt.close() # Close figure to free memory

except Exception as e:
    print(f"Error generating/saving Regret Plot: {e}")
    traceback.print_exc()
    plt.close() # Ensure plot is closed on error


# --- Plot 2: Market Prices ---
try:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Ensure market_prices and market_dates are valid and aligned
    if market_prices is None or market_dates is None or len(market_prices) != len(market_dates):
         raise ValueError("Market price data or dates are invalid or misaligned.")

    # Check if market_dates is suitable for plotting (e.g., DatetimeIndex)
    if pd.api.types.is_datetime64_any_dtype(market_dates):
        plot_dates = market_dates
        ax.set_xlabel('Date')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right') # Rotate labels only for datetime
    else: # Use integer index if dates aren't datetime objects
        plot_dates = np.arange(len(market_prices))
        ax.set_xlabel('Time Steps (Index)') # Adjust label if using index

    ax.plot(plot_dates, market_prices.values) # Ensure plotting numpy array from Series
    ax.set_title(f'Market Prices ({DATA_FILENAME})')
    ax.set_ylabel('Price')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout
    plt.savefig('market_prices.png', bbox_inches='tight')
    print("Saved market_prices.png")
    plt.close()
except ValueError as e:
     print(f"Error generating/saving Market Price Plot: {e}")
     plt.close()
except Exception as e:
    print(f"Unexpected error generating/saving Market Price Plot: {e}")
    traceback.print_exc()
    plt.close()


# --- Plot 3: Final Cumulative Return Comparison (MABs + Pure) ---
try:
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    plot_successful = False

    all_plot_elements = [] # Store tuples of (x_pos, y_val, label, color, marker, yerr)

    # --- MAB Data Processing ---
    current_x = 0
    for i, algo_name in enumerate(mab_algo_names):
         results = processed_mab_results.get(algo_name, {})
         mean_ret = results.get('mean_final_return')
         std_ret = results.get('std_final_return')
         num_runs = results.get('num_valid_runs', 0)

         if pd.notna(mean_ret) and num_runs > 0:
              # Ensure std is non-negative, replace NaN std with 0 for plotting if mean is valid
              plot_std = std_ret if pd.notna(std_ret) else 0
              plot_std = max(0, plot_std) # Ensure non-negative error bar

              label = f"{algo_name} (N={num_runs})"
              all_plot_elements.append((current_x, mean_ret, label, mab_colors[i], 'o', plot_std))
              current_x += 1
              plot_successful = True

    # --- Pure Strategy Data Processing ---
    for i, name in enumerate(pure_strategy_names):
         returns = pure_strategy_results.get(name)
         # Check if returns exist, have correct length, and final value is valid
         if returns is not None and len(returns) == sim_length and pd.notna(returns[-1]):
             final_return = returns[-1]
             label = f"{name} (Pure)"
             # Assign color based on the pure strategy index
             color_idx = i % len(strategy_colors) # Cycle through strategy colors
             all_plot_elements.append((current_x, final_return, label, strategy_colors[color_idx], '*', 0)) # No error bars for pure
             current_x += 1
             plot_successful = True
         elif name in pure_strategy_results: # Log if data exists but is invalid
             print(f"Plotting Warning (Final Returns): Invalid data or length mismatch for pure strategy {name}. Length: {len(returns) if returns is not None else 'None'}, Expected: {sim_length}")


    # --- Plotting ---
    if plot_successful:
        all_plot_elements.sort(key=lambda item: item[0]) # Ensure plotting in order of x_pos

        x_positions = [item[0] for item in all_plot_elements]
        y_values = [item[1] for item in all_plot_elements]
        labels = [item[2] for item in all_plot_elements]
        colors = [item[3] for item in all_plot_elements]
        markers = [item[4] for item in all_plot_elements]
        y_errors = [item[5] for item in all_plot_elements] # 0 for pure strategies

        # Plot error bars first (only for MABs where y_errors > 0)
        mab_x = [p[0] for p in all_plot_elements if p[5] > 0]
        mab_y = [p[1] for p in all_plot_elements if p[5] > 0]
        mab_err = [p[5] for p in all_plot_elements if p[5] > 0]
        if mab_x:
            ax.errorbar(mab_x, mab_y, yerr=mab_err, fmt='none', capsize=5, color='gray', ecolor='lightgray', elinewidth=1, capthick=1, label='_nolegend_')

        # Plot individual points with markers and colors
        for x, y, lbl, clr, mrk in zip(x_positions, y_values, labels, colors, markers):
            ax.plot(x, y, marker=mrk, color=clr, linestyle='None', markersize=10 if mrk=='o' else 12, label=lbl)


        ax.set_title('Final Cumulative Return Comparison (MAB vs Pure)')
        ax.set_xlabel('Algorithm / Strategy (N = Valid Runs for MABs)')
        ax.set_ylabel('Final Cumulative Return')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        # Place legend outside plot area
        ax.legend(title="Legend", bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.savefig('final_return_comparison.png', bbox_inches='tight')
        print("Saved final_return_comparison.png")
    else:
        print("Skipping saving Final Return plot - no valid data plotted.")

    plt.close() # Close figure

except Exception as e:
    print(f"Error generating/saving Final Return Plot: {e}")
    traceback.print_exc()
    plt.close()


# --- Plot 4: Cumulative Strategy Choices (MABs) ---
try:
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Filter only algos with valid choice counts AND more than 0 valid runs
    valid_choice_algos_data = {}
    valid_algo_names_ordered = [] # Keep track of order for plotting
    for name in mab_algo_names: # Iterate in defined order
         results = processed_mab_results.get(name, {})
         counts = results.get('choice_counts')
         num_runs = results.get('num_valid_runs', 0)

         # Check if counts exist, match expected arm number, and sum > 0 (meaning choices were made)
         # Also ensure there was at least one valid run for this algo
         if counts is not None and len(counts) == num_mab_arms and counts.sum() > 0 and num_runs > 0:
             valid_choice_algos_data[name] = counts
             valid_algo_names_ordered.append(name)
         elif name in processed_mab_results: # Log if data exists but is invalid
             print(f"Plotting Warning (Choices): Invalid or zero choice counts for {name}. Counts: {counts}, NumRuns: {num_runs}, Expected Arms: {num_mab_arms}")


    if not valid_choice_algos_data:
         raise ValueError("No valid choice data found for any MAB algorithm with successful runs.")

    # Create DataFrame using the filtered data and ordered names
    choice_data = pd.DataFrame(valid_choice_algos_data, index=mab_arm_strategy_names).T
    choice_data = choice_data.loc[valid_algo_names_ordered] # Ensure rows match algo order

    # Ensure column names (strategy names) are correct
    if list(choice_data.columns) != mab_arm_strategy_names:
        print("Warning: Column names mismatch in choice data. Attempting to fix.")
        if len(choice_data.columns) == len(mab_arm_strategy_names):
            choice_data.columns = mab_arm_strategy_names
        else:
             raise ValueError(f"Cannot plot choices: Number of columns ({len(choice_data.columns)}) != Number of MAB arms ({len(mab_arm_strategy_names)}).")


    # Assign colors based on the strategy names (columns)
    plot_colors = [strategy_colors[mab_arm_strategy_names.index(name)] for name in choice_data.columns]

    choice_data.plot(kind='bar', stacked=True, ax=ax, color=plot_colors, width=0.8)

    # Add number of valid runs to x-axis labels
    run_counts = [processed_mab_results[name]['num_valid_runs'] for name in valid_algo_names_ordered]
    x_labels_with_runs = [f"{name}\n(N={n})" for name, n in zip(valid_algo_names_ordered, run_counts)]
    ax.set_xticklabels(x_labels_with_runs, rotation=0, ha='center') # Adjust rotation if needed

    ax.set_title('MAB Strategy Choices (Total Across Valid Runs)')
    ax.set_xlabel('MAB Algorithm (N = Valid Runs)')
    ax.set_ylabel('Total Times Chosen')
    ax.legend(title='Chosen Strategy', bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig('mab_strategy_choices.png', bbox_inches='tight')
    print("Saved mab_strategy_choices.png")
    plt.close()

except ValueError as e: # Catch specific error for no/inconsistent data
     print(f"Skipping saving MAB Choices plot - {e}")
     if plt.gcf().get_axes(): # Close figure only if it was created
         plt.close()
except Exception as e:
    print(f"Error generating/saving MAB Choices Plot: {e}")
    traceback.print_exc()
    if plt.gcf().get_axes(): # Close figure only if it was created
         plt.close()


# --- Plot 5: Cumulative Returns of All Strategies (MABs + Pure) ---
try:
    plt.figure(figsize=(12, 7)) # Wider figure
    ax = plt.gca()
    time_steps = np.arange(sim_length)
    plot_successful = False

    # Plot MAB average cumulative returns
    for i, algo_name in enumerate(mab_algo_names):
        results = processed_mab_results.get(algo_name, {})
        mean_ret = results.get('mean_cum_return')
        num_runs = results.get('num_valid_runs', 0)

        if mean_ret is not None and len(mean_ret) == sim_length and not np.isnan(mean_ret).all() and num_runs > 0:
            label = f"{algo_name} (MAB Avg, N={num_runs})"
            ax.plot(time_steps, mean_ret, label=label, color=mab_colors[i], linestyle='-', linewidth=1.5)
            plot_successful = True
        elif algo_name in processed_mab_results:
             print(f"Plotting Warning (All Returns - MAB): Invalid data or length mismatch for {algo_name}. Length: {len(mean_ret) if mean_ret is not None else 'None'}, Expected: {sim_length}")


    # Plot Pure Strategy cumulative returns
    for i, name in enumerate(pure_strategy_names):
        returns = pure_strategy_results.get(name)
        if returns is not None and len(returns) == sim_length and not np.isnan(returns).all():
            label = f"{name} (Pure)"
            color_idx = i % len(strategy_colors) # Use strategy colors
            ax.plot(time_steps, returns, label=label, color=strategy_colors[color_idx], linestyle='--', linewidth=1.5, alpha=0.9)
            plot_successful = True
        elif name in pure_strategy_results:
             print(f"Plotting Warning (All Returns - Pure): Invalid data or length mismatch for {name}. Length: {len(returns) if returns is not None else 'None'}, Expected: {sim_length}")


    if plot_successful:
        ax.set_title('Cumulative Returns: MAB Averages vs. Pure Strategies')
        ax.set_xlabel(f'Time Steps ({sim_length} total)')
        ax.set_ylabel('Cumulative Return')
        # Place legend outside plot area
        ax.legend(title="Strategy / Algorithm", fontsize='small', bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plt.savefig('all_cumulative_returns.png', bbox_inches='tight')
        print("Saved all_cumulative_returns.png")
    else:
        print("Skipping saving All Cumulative Returns plot - no valid data plotted.")
    plt.close()

except Exception as e:
    print(f"Error generating/saving All Cumulative Returns Plot: {e}")
    traceback.print_exc()
    plt.close()


# --- Plot 6: Distribution of Final MAB Returns (Box Plot) ---
try:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Prepare data for boxplot: list of non-empty, valid return arrays
    valid_boxplot_data = []
    valid_labels_with_runs = []
    valid_algo_indices = [] # Store original index for coloring

    for i, algo_name in enumerate(mab_algo_names):
         results = processed_mab_results.get(algo_name, {})
         final_returns = results.get('all_final_returns')
         num_runs = results.get('num_valid_runs', 0)

         # Check if final_returns is a list/array with actual numbers and runs > 0
         if isinstance(final_returns, (list, np.ndarray)) and len(final_returns) > 0 and num_runs > 0:
             # Further filter out potential NaNs within the list if any slipped through
             valid_returns_for_algo = [r for r in final_returns if pd.notna(r)]
             if valid_returns_for_algo: # Ensure list is not empty after filtering NaNs
                 valid_boxplot_data.append(valid_returns_for_algo)
                 valid_labels_with_runs.append(f"{algo_name}\n(N={num_runs})")
                 valid_algo_indices.append(i) # Store original index

    if valid_boxplot_data:
        bp = ax.boxplot(valid_boxplot_data, patch_artist=True, showfliers=False, labels=valid_labels_with_runs)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center') # Adjust rotation if needed

        # Apply colors using the stored original indices
        for i, patch in enumerate(bp['boxes']):
             original_index = valid_algo_indices[i] # Get original index for this box
             patch.set_facecolor(mab_colors[original_index])
             patch.set_alpha(0.7) # Slightly more opaque

        ax.set_title('Distribution of Final MAB Returns (Across Valid Runs)')
        ax.set_ylabel('Final Cumulative Return')
        ax.set_xlabel('MAB Algorithm (N = Valid Runs)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('mab_final_return_distribution.png', bbox_inches='tight')
        print("Saved mab_final_return_distribution.png")

    else:
        print("Skipping saving MAB Return Distribution plot - no valid data for boxplot.")
    plt.close() # Close figure

except Exception as e:
    print(f"Error generating/saving MAB Return Distribution Plot: {e}")
    traceback.print_exc()
    plt.close()


print("\nComparison script finished.")