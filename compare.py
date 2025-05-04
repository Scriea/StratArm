# --- compare_mab_algorithms.py ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time # To time the simulations

# --- Import your existing classes ---
# Make sure these files are in the same directory or accessible via PYTHONPATH
from src.simulator import Simulator
from src.strategy import SMACrossoverStrategy, RSIStrategy, BollingerStrategy # Add others if used
from src.bandit_algorithms import Eps_Greedy, UCB, KL_UCB, GaussianThompsonSampling, SW_UCB, EWMA_UCB # Import the MABs you want to test
from src.data_utils import preprocess_data, calculate_features

# --- Configuration ---
DATA_FILE = './data/BSE_Sensex_30_Historical_Data_2021_2024.csv' # <<< --- Specify your data file path here
N_RUNS = 10 # <<< --- Number of simulation runs per algorithm (increase for smoother means, e.g., 30, 50)
HORIZON = None # Set to a specific number of steps if needed, otherwise uses full data length

# --- Define Trading Strategies ---
# Use the same strategies you used in your single simulation
strategies = [
    SMACrossoverStrategy(short_window=20, long_window=50),
    RSIStrategy(rsi_window=14, low=30, high=70),
    BollingerStrategy(window=20, num_std=2)
    # Add other strategies if you have them
]
num_strategies = len(strategies)
strategy_names = [s.name for s in strategies] # Get names for plotting

# --- Define MAB Algorithms to Compare ---
# IMPORTANT: Initialize with appropriate parameters, including scaling range!
ESTIMATED_MIN_REWARD = -0.05 # <<< --- Adjust based on your expected single-step returns
ESTIMATED_MAX_REWARD = 0.05  # <<< --- Adjust based on your expected single-step returns

algorithms_to_test = {
    "Epsilon-Greedy (0.1)": Eps_Greedy(num_arms=num_strategies, epsilon=0.1),
    "UCB (c=2)": UCB(num_arms=num_strategies, c=2.0,
                     min_reward=ESTIMATED_MIN_REWARD, max_reward=ESTIMATED_MAX_REWARD),
    "SW-UCB (W=100)": SW_UCB(num_arms=num_strategies, window_size=100, c=2.0,
                            min_reward=ESTIMATED_MIN_REWARD, max_reward=ESTIMATED_MAX_REWARD),
    "EWMA-UCB (L=0.99)": EWMA_UCB(num_arms=num_strategies, lambda_decay=0.99, c=2.0,
                                 min_reward=ESTIMATED_MIN_REWARD, max_reward=ESTIMATED_MAX_REWARD),
    "Thompson Sampling": GaussianThompsonSampling(num_arms=num_strategies) # Assumes Gaussian rewards
    # Add/remove algorithms as needed
}
algo_names = list(algorithms_to_test.keys())

# --- Load and Prepare Data (Do this ONCE outside the loop) ---
print(f"Loading data from {DATA_FILE}...")
try:
    raw_df = pd.read_csv(DATA_FILE)
    # Make a copy for preprocessing to avoid modifying the original raw_df
    df_processed = preprocess_data(raw_df.copy())
    df_features = calculate_features(df_processed.copy()) # df_features has NaNs dropped
    # Store the price data for the market plot - use the one AFTER feature calculation dropna
    market_prices = df_features['Price'].copy()
    market_dates = df_features['Date'].copy() if 'Date' in df_features else pd.to_datetime(df_features.index) # Assuming Date index if column missing
    print(f"Data prepared. Simulation horizon based on features: {len(df_features)-1} steps.")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    exit()
except Exception as e:
    print(f"Error during data preparation: {e}")
    exit()

# --- Store Results ---
# Dictionary to hold results for each algorithm
# Structure: results[algo_name] = {'regrets': [run1_regret_array, run2_regret_array,...],
#                                'returns': [run1_final_return, run2_final_return,...],
#                                'choices': [run1_choices_array, run2_choices_array,...]}
results = {name: {'regrets': [], 'returns': [], 'choices': []} for name in algo_names}

# --- Run Simulations ---
start_time = time.time()
print(f"\nStarting {N_RUNS} simulation runs for each of {len(algo_names)} algorithms...")

for run in range(N_RUNS):
    print(f"\n--- Run {run + 1}/{N_RUNS} ---")
    for algo_name, mab_agent in algorithms_to_test.items():
        print(f"  Simulating: {algo_name}...")
        # Reset the MAB agent's state before each run
        mab_agent.reset()

        # Create a new simulator instance for each run
        # Pass the feature DataFrame directly
        simulator = Simulator(df=df_features.copy(), strategies=strategies, mab=mab_agent)

        try:
            # Run the simulation
            run_results = simulator.run() # run() returns the results dict

            # Store the results for this run and algorithm
            # Ensure results are numpy arrays for easier processing later
            results[algo_name]['regrets'].append(np.array(run_results['cumulative_regret']))
            # Store the FINAL cumulative return
            results[algo_name]['returns'].append(run_results['cumulative_returns'][-1] if run_results['cumulative_returns'] else 0)
            results[algo_name]['choices'].append(np.array(run_results['chosen_arms']))

        except Exception as e:
            print(f"    ERROR during simulation for {algo_name} on run {run + 1}: {e}")
            # Optionally add placeholder results (e.g., NaNs) or skip this run for this algo
            results[algo_name]['regrets'].append(np.array([np.nan])) # Placeholder
            results[algo_name]['returns'].append(np.nan)
            results[algo_name]['choices'].append(np.array([]))


end_time = time.time()
print(f"\nSimulations completed in {end_time - start_time:.2f} seconds.")

# --- Process and Aggregate Results ---
print("Aggregating results...")
processed_results = {}
sim_length = 0 # Determine simulation length from first successful run

for algo_name in algo_names:
    # --- Regret ---
    # Stack regrets from all runs for this algo
    all_regrets = results[algo_name]['regrets']
    # Filter out potential placeholder NaNs from failed runs before stacking
    valid_regrets = [r for r in all_regrets if r.ndim > 0 and not np.isnan(r).any()]
    if not valid_regrets:
        print(f"Warning: No valid regret data for {algo_name}. Skipping regret plot for this algo.")
        mean_regret, std_regret = np.array([]), np.array([])
    else:
        # Pad shorter arrays if necessary (e.g., due to errors) - assumes most runs have same length
        max_len = max(len(r) for r in valid_regrets)
        if sim_length == 0: sim_length = max_len # Set consistent sim length for plots
        padded_regrets = [np.pad(r, (0, max_len - len(r)), 'edge') for r in valid_regrets] # Pad with last value
        regret_stack = np.stack(padded_regrets, axis=0)
        # Calculate mean and std dev across runs (axis=0)
        mean_regret = np.cumsum(np.mean(regret_stack, axis=0)) # Plot CUMULATIVE mean regret
        std_regret = np.std(np.cumsum(regret_stack, axis=0), axis=0) # Std dev of CUMULATIVE regret

    # --- Returns ---
    valid_returns = [r for r in results[algo_name]['returns'] if not np.isnan(r)]
    if not valid_returns:
         print(f"Warning: No valid return data for {algo_name}.")
         mean_return, std_return = np.nan, np.nan
    else:
        mean_return = np.mean(valid_returns)
        std_return = np.std(valid_returns)

    # --- Choices ---
    valid_choices = [c for c in results[algo_name]['choices'] if c.size > 0]
    if not valid_choices:
        print(f"Warning: No valid choice data for {algo_name}.")
        choice_counts = np.zeros(num_strategies)
    else:
        all_choices_flat = np.concatenate(valid_choices)
        counts = np.bincount(all_choices_flat, minlength=num_strategies)
        # Normalize counts by the number of valid runs to get average counts per run if desired
        # Or just sum total choices across all runs
        choice_counts = counts # Total counts across all runs

    processed_results[algo_name] = {
        'mean_regret': mean_regret,
        'std_regret': std_regret,
        'mean_return': mean_return,
        'std_return': std_return,
        'choice_counts': choice_counts
    }

# --- Plotting ---
print("Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # Create 2x2 grid
fig.suptitle('Multi-Armed Bandit Algorithm Comparison', fontsize=16)
colors = plt.cm.viridis(np.linspace(0, 1, len(algo_names))) # Color map for algorithms

# --- Plot 1: Cumulative Regret Comparison ---
ax = axes[0, 0]
time_steps = np.arange(sim_length) # X-axis for regret plot

for i, algo_name in enumerate(algo_names):
    if algo_name in processed_results and processed_results[algo_name]['mean_regret'].size > 0:
        mean_r = processed_results[algo_name]['mean_regret']
        std_r = processed_results[algo_name]['std_regret']
        # Ensure lengths match time_steps before plotting
        if len(mean_r) == sim_length and len(std_r) == sim_length:
            ax.plot(time_steps, mean_r, label=f"{algo_name} Mean", color=colors[i])
            ax.fill_between(time_steps, mean_r - std_r, mean_r + std_r, color=colors[i], alpha=0.2, label=f"{algo_name} Std Dev")
        else:
             print(f"Plotting Warning: Length mismatch for {algo_name} regret ({len(mean_r)}) vs sim_length ({sim_length}). Skipping.")


ax.set_title('Cumulative Regret Comparison Across Algorithms')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Cumulative Regret')
ax.legend(fontsize='small')
ax.grid(True, linestyle='--', alpha=0.6)

# --- Plot 2: Simulated Market Prices ---
ax = axes[0, 1]
# Ensure market_dates and market_prices have the same length
min_len = min(len(market_dates), len(market_prices))
ax.plot(market_dates[:min_len], market_prices[:min_len])
ax.set_title('Simulated Market Prices') # Or 'Actual Market Prices'
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.grid(True, linestyle='--', alpha=0.6)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right') # Rotate date labels

# --- Plot 3: Total Mean Profit (Return) Comparison ---
ax = axes[1, 0]
means = [processed_results[name]['mean_return'] for name in algo_names]
stds = [processed_results[name]['std_return'] for name in algo_names]
x_pos = np.arange(len(algo_names))

# Plot as error bars
ax.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, markersize=8, color='black')
# Add colored points for better distinction
for i in range(len(algo_names)):
    ax.plot(x_pos[i], means[i], 'o', color=colors[i], markersize=8, label=algo_names[i])


ax.set_title('Mean Final Cumulative Return with Std Dev')
ax.set_xlabel('Algorithms')
ax.set_ylabel('Mean Final Cumulative Return')
ax.set_xticks(x_pos)
ax.set_xticklabels(algo_names, rotation=45, ha='right')
ax.legend(fontsize='small')
ax.grid(True, axis='y', linestyle='--', alpha=0.6)


# --- Plot 4: Cumulative Strategy Choices ---
ax = axes[1, 1]
choice_data = pd.DataFrame({
    algo_name: processed_results[algo_name]['choice_counts']
    for algo_name in algo_names if algo_name in processed_results # Ensure data exists
}).T # Transpose to have algorithms as rows

if not choice_data.empty:
    choice_data.columns = strategy_names # Set column names to strategy names
    # Create stacked bar chart
    choice_data.plot(kind='bar', stacked=True, ax=ax, color=plt.cm.tab10.colors) # Use a different colormap

    ax.set_title('Cumulative Strategy Choices by MAB Algorithm')
    ax.set_xlabel('MAB Algorithm')
    ax.set_ylabel('Total Times Chosen (Across All Runs)')
    ax.legend(title='Strategies', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
else:
    ax.text(0.5, 0.5, 'No strategy choice data available.', horizontalalignment='center', verticalalignment='center')


# --- Final Adjustments ---
plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
plt.show()

print("Plotting complete.")