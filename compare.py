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
    from src.simulator import Simulator
    from src.strategy import TradingStrategy, SMACrossoverStrategy, RSIStrategy, BollingerStrategy, CompositeStrategy
    from src.bandit_algorithms import Eps_Greedy, UCB, KL_UCB, GaussianThompsonSampling, SW_UCB, EWMA_UCB
    from src.data_utils import preprocess_data, calculate_features
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the 'src' directory is correctly structured and accessible.")
    print("You might need to adjust PYTHONPATH or run this script from the correct directory.")
    exit()

# --- Configuration ---
# Use os.path.join for better cross-platform compatibility
DATA_DIR = 'data' # Specify directory containing data
DATA_FILENAME = 'Download Data - INDEX_US_S&P US_SPX.csv' # Specify filename
DATA_FILE = os.path.join(DATA_DIR, DATA_FILENAME)

N_RUNS_MAB = 10 # Number of simulation runs per MAB algorithm
HORIZON = None # Set to a specific number of steps if needed

# MAB Reward Scaling Configuration
ESTIMATED_MIN_REWARD = -0.05 # <<< --- Adjust: Estimated minimum single-step return
ESTIMATED_MAX_REWARD = 0.05  # <<< --- Adjust: Estimated maximum single-step return

# --- Define Base Trading Strategies ---
base_strategies: list[TradingStrategy] = [
    SMACrossoverStrategy(short_window=20, long_window=50),
    RSIStrategy(rsi_window=14, low=30, high=70),
    BollingerStrategy(window=20, num_std=2)
]

# --- Create Composite Strategy ---
composite_strategy = CompositeStrategy(strategies=base_strategies)

# --- Full List of Strategies to Simulate as Pure ---
strategies_to_simulate_pure: list[TradingStrategy] = base_strategies + [composite_strategy]
num_pure_strategies = len(strategies_to_simulate_pure)
pure_strategy_names = [s.name for s in strategies_to_simulate_pure]

# --- Strategies for the MAB to Choose From ---
mab_chooses_from_strategies: list[TradingStrategy] = base_strategies
num_mab_arms = len(mab_chooses_from_strategies)
mab_arm_strategy_names = [s.name for s in mab_chooses_from_strategies] # Names for plot legend

# --- Define MAB Algorithms to Compare ---
# Ensure keys are unique and descriptive
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
    raw_df = pd.read_csv(DATA_FILE)
    df_processed = preprocess_data(raw_df.copy())
    df_features = calculate_features(df_processed.copy()) # This drops leading NaNs

    if df_features.empty:
        raise ValueError("DataFrame is empty after preprocessing and feature calculation.")

    market_prices = df_features['Price'].copy()
    # Handle potential DatetimeIndex vs. 'Date' column
    if isinstance(df_features.index, pd.DatetimeIndex):
        market_dates = df_features.index
    elif 'Date' in df_features.columns:
        market_dates = pd.to_datetime(df_features['Date'])
    else:
        # Fallback if no date info available (will affect market plot x-axis)
        print("Warning: No 'Date' column or DatetimeIndex found. Using integer index for market plot.")
        market_dates = df_features.index

    sim_length = len(df_features) - 1
    if sim_length <= 0:
        raise ValueError(f"Simulation length ({sim_length}) is not positive. Check data processing.")
    print(f"Data prepared. Simulation horizon: {sim_length} steps.")

    # Calculate percentage returns needed for simulation
    price_returns = df_features['Price'].pct_change()
    price_returns.replace([np.inf, -np.inf], 0, inplace=True)
    price_returns = price_returns.fillna(0) # Fill first NaN

except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except ValueError as e:
     print(f"Error: {e}")
     exit()
except Exception as e:
    print(f"Error during data preparation: {e}")
    traceback.print_exc()
    exit()

# --- Simulate Pure Strategies ---
print("\nSimulating Pure Strategies (Fixed Arm)...")
pure_strategy_results = {} # Store cumulative return series {name: array}
for strategy in strategies_to_simulate_pure:
    print(f"  Simulating: {strategy.name}...")
    try:
        signals = strategy.generate_signals(df_features)
        if not signals.index.equals(df_features.index):
             signals = signals.reindex(df_features.index)
        if signals.iloc[1:sim_length+1].isnull().values.any(): # Check within sim period
             print(f"    Warning: NaNs detected in signals for pure strategy {strategy.name} during simulation period.")

        cumulative_return = 0.0
        returns_over_time = np.zeros(sim_length)
        for t in range(1, sim_length + 1):
            signal = signals.iloc[t]
            market_return = price_returns.iloc[t]
            step_return = 0.0
            if pd.notna(signal):
                if signal == 1: step_return = market_return
                elif signal == -1: step_return = -market_return
            cumulative_return += step_return
            returns_over_time[t-1] = cumulative_return

        pure_strategy_results[strategy.name] = returns_over_time
        print(f"    Finished. Final Cumulative Return: {cumulative_return:.4f}")
    except Exception as e:
        print(f"    ERROR during pure strategy simulation for {strategy.name}: {e}")
        traceback.print_exc()
        pure_strategy_results[strategy.name] = np.full(sim_length, np.nan)

# --- Run MAB Simulations ---
mab_results_storage = {name: {'regrets': [], 'returns_final': [], 'choices': [], 'returns_over_time': []} for name in mab_algo_names}
start_time = time.time()
print(f"\nStarting {N_RUNS_MAB} MAB simulation runs for each of {len(mab_algo_names)} algorithms...")
for run in range(N_RUNS_MAB):
    print(f"\n--- MAB Run {run + 1}/{N_RUNS_MAB} ---")
    for algo_name, mab_agent in algorithms_to_test.items():
        # print(f"  Simulating MAB: {algo_name}...") # Reduce verbosity inside run loop
        mab_agent.reset()
        simulator = Simulator(df=df_features.copy(), strategies=mab_chooses_from_strategies, mab=mab_agent)
        try:
            run_results = simulator.run() # run() returns the results dict
            returns_cum = np.array(run_results['cumulative_returns'])
            # Ensure results have expected length before storing
            if len(run_results.get('cumulative_regret', [])) == sim_length and \
               len(run_results.get('chosen_arms', [])) == sim_length and \
               len(returns_cum) == sim_length:
                mab_results_storage[algo_name]['regrets'].append(np.array(run_results['cumulative_regret']))
                mab_results_storage[algo_name]['returns_final'].append(returns_cum[-1])
                mab_results_storage[algo_name]['choices'].append(np.array(run_results['chosen_arms']))
                mab_results_storage[algo_name]['returns_over_time'].append(returns_cum)
            else:
                print(f"    WARNING: Inconsistent results length for {algo_name} on run {run + 1}. Skipping run storage.")
                # Append NaNs/empty to maintain structure for aggregation checks
                mab_results_storage[algo_name]['regrets'].append(np.full(sim_length, np.nan))
                mab_results_storage[algo_name]['returns_final'].append(np.nan)
                mab_results_storage[algo_name]['choices'].append(np.array([]))
                mab_results_storage[algo_name]['returns_over_time'].append(np.full(sim_length, np.nan))

        except Exception as e:
            print(f"    ERROR during MAB simulation for {algo_name} on run {run + 1}: {e}")
            traceback.print_exc()
            # Append NaNs/empty on error
            mab_results_storage[algo_name]['regrets'].append(np.full(sim_length, np.nan))
            mab_results_storage[algo_name]['returns_final'].append(np.nan)
            mab_results_storage[algo_name]['choices'].append(np.array([]))
            mab_results_storage[algo_name]['returns_over_time'].append(np.full(sim_length, np.nan))

end_time = time.time()
print(f"\nMAB simulations completed in {end_time - start_time:.2f} seconds.")

# --- Process and Aggregate MAB Results ---
print("Aggregating MAB results...")
processed_mab_results = {}
for algo_name in mab_algo_names:
    processed_mab_results[algo_name] = {} # Initialize dict for this algo

    # Filter valid runs based on final return (as a proxy for successful completion)
    valid_run_indices = [i for i, r in enumerate(mab_results_storage[algo_name]['returns_final']) if pd.notna(r)]
    num_valid_runs = len(valid_run_indices)

    if num_valid_runs == 0:
        print(f"  Warning: No valid runs found for {algo_name}. Cannot aggregate results.")
        processed_mab_results[algo_name] = { # Fill with defaults/NaNs
            'mean_cum_regret': np.full(sim_length, np.nan), 'std_cum_regret': np.full(sim_length, np.nan),
            'mean_final_return': np.nan, 'std_final_return': np.nan,
            'choice_counts': np.zeros(num_mab_arms), 'all_final_returns': []
        }
        continue # Skip to next algorithm

    # Aggregate Regret
    valid_regrets = [mab_results_storage[algo_name]['regrets'][i] for i in valid_run_indices]
    regret_stack = np.stack(valid_regrets, axis=0)
    mean_cum_regret = np.cumsum(np.mean(regret_stack, axis=0))
    std_cum_regret = np.std(np.cumsum(regret_stack, axis=0), axis=0)
    processed_mab_results[algo_name]['mean_cum_regret'] = mean_cum_regret
    processed_mab_results[algo_name]['std_cum_regret'] = std_cum_regret

    # Aggregate Final Returns
    valid_final_returns = [mab_results_storage[algo_name]['returns_final'][i] for i in valid_run_indices]
    processed_mab_results[algo_name]['mean_final_return'] = np.mean(valid_final_returns)
    processed_mab_results[algo_name]['std_final_return'] = np.std(valid_final_returns)
    processed_mab_results[algo_name]['all_final_returns'] = valid_final_returns # For box plot

    # Aggregate Choices
    valid_choices = [mab_results_storage[algo_name]['choices'][i] for i in valid_run_indices]
    choice_counts = np.bincount(np.concatenate(valid_choices), minlength=num_mab_arms)
    processed_mab_results[algo_name]['choice_counts'] = choice_counts


# ============================================================================
# --- Plotting Code Starts Here ---
# ============================================================================
print("\nGenerating and saving individual plots...")

# --- Define Colors ---
# Ensure enough colors if many algos/strategies exist
mab_colors = plt.cm.viridis(np.linspace(0, 1, len(mab_algo_names)))
strategy_colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_pure_strategies))) # Use tab10, ensure enough colors

# --- Plot 1: Cumulative Regret Comparison (MABs) ---
try:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    time_steps = np.arange(sim_length)
    plot_successful = False
    for i, algo_name in enumerate(mab_algo_names):
        # Check if results exist and are not all NaN
        if algo_name in processed_mab_results and \
           'mean_cum_regret' in processed_mab_results[algo_name] and \
           not np.isnan(processed_mab_results[algo_name]['mean_cum_regret']).all():

            mean_r = processed_mab_results[algo_name]['mean_cum_regret']
            std_r = processed_mab_results[algo_name]['std_cum_regret']
            # Plot only if lengths are correct (sim_length determined earlier)
            if len(mean_r) == sim_length and len(std_r) == sim_length:
                ax.plot(time_steps, mean_r, label=f"{algo_name}", color=mab_colors[i])
                ax.fill_between(time_steps, mean_r - std_r, mean_r + std_r, color=mab_colors[i], alpha=0.2)
                plot_successful = True
            else:
                 print(f"Plotting Warning (Regret): Length mismatch for {algo_name} ({len(mean_r)}) vs sim_length ({sim_length}).")

    if plot_successful:
        ax.set_title('MAB Cumulative Regret (Mean +/- StdDev)')
        ax.set_xlabel(f'Time Steps ({sim_length} total)')
        ax.set_ylabel('Cumulative Regret')
        ax.legend(fontsize='medium')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('mab_cumulative_regret.png', bbox_inches='tight')
        print("Saved mab_cumulative_regret.png")
    else:
        print("Skipping saving regret plot - no valid data plotted.")
    plt.close()

except Exception as e:
    print(f"Error generating/saving Regret Plot: {e}")
    traceback.print_exc()
    plt.close()

# --- Plot 2: Market Prices ---
try:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # Check if market_dates is suitable for plotting (e.g., DatetimeIndex)
    if pd.api.types.is_datetime64_any_dtype(market_dates):
        plot_dates = market_dates
    else: # Use integer index if dates aren't datetime objects
        plot_dates = np.arange(len(market_prices))
        ax.set_xlabel('Time Steps (Index)') # Adjust label if using index

    ax.plot(plot_dates, market_prices)
    ax.set_title('Market Prices')
    if pd.api.types.is_datetime64_any_dtype(market_dates):
         ax.set_xlabel('Date')
         plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Price')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('market_prices.png', bbox_inches='tight')
    print("Saved market_prices.png")
    plt.close()
except Exception as e:
    print(f"Error generating/saving Market Price Plot: {e}")
    traceback.print_exc()
    plt.close()

# --- Plot 3: Final Cumulative Return Comparison (MABs + Pure) ---
try:
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    plot_successful = False

    # MAB Data
    mab_means = []
    mab_stds = []
    valid_mab_labels = []
    valid_mab_x_pos = []
    current_mab_x = 0
    for i, algo_name in enumerate(mab_algo_names):
         if algo_name in processed_mab_results and pd.notna(processed_mab_results[algo_name]['mean_final_return']):
              mab_means.append(processed_mab_results[algo_name]['mean_final_return'])
              mab_stds.append(processed_mab_results[algo_name]['std_final_return']) # Will be NaN if mean is NaN, handle below
              valid_mab_labels.append(f"{algo_name} (MAB)")
              valid_mab_x_pos.append(current_mab_x)
              current_mab_x += 1
              plot_successful = True # Mark plot as having some data

    # Plot MABs if any valid data exists
    if valid_mab_labels:
        valid_mab_stds = [s if pd.notna(s) else 0 for s in mab_stds] # Replace NaN std with 0 for plotting
        ax.errorbar(valid_mab_x_pos, mab_means, yerr=valid_mab_stds, fmt='none', capsize=5, color='black', ecolor='gray', label='_nolegend_')
        for i in range(len(valid_mab_labels)):
            ax.plot(valid_mab_x_pos[i], mab_means[i], 'o', color=mab_colors[mab_algo_names.index(valid_mab_labels[i].split(' ')[0])], # Match original color index
                    markersize=10, label=valid_mab_labels[i])


    # Pure Strategy Data
    pure_returns_final = []
    valid_pure_labels = []
    valid_pure_x_pos = []
    current_pure_x = current_mab_x # Start pure strategies after MABs
    for i, name in enumerate(pure_strategy_names):
         final_return = pure_strategy_results.get(name, np.array([np.nan]))[-1] # Get last value, default NaN
         if pd.notna(final_return):
             pure_returns_final.append(final_return)
             valid_pure_labels.append(f"{name} (Pure)")
             valid_pure_x_pos.append(current_pure_x)
             current_pure_x += 1
             plot_successful = True # Mark plot as having some data

    # Plot Pure Strategies if any valid data exists
    if valid_pure_labels:
        for i in range(len(valid_pure_labels)):
             ax.plot(valid_pure_x_pos[i], pure_returns_final[i], '*', color=strategy_colors[i % len(strategy_colors)],
                    markersize=14, label=valid_pure_labels[i])


    if plot_successful:
        ax.set_title('Final Cumulative Return Comparison')
        ax.set_xlabel('Algorithm / Strategy')
        ax.set_ylabel('Final Cumulative Return')
        # Combine valid labels and positions for x-axis
        all_valid_labels = valid_mab_labels + valid_pure_labels
        all_valid_x_pos = valid_mab_x_pos + valid_pure_x_pos
        ax.set_xticks(all_valid_x_pos)
        ax.set_xticklabels(all_valid_labels, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.legend(title="Legend", bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
        plt.savefig('final_return_comparison.png', bbox_inches='tight')
        print("Saved final_return_comparison.png")
    else:
        print("Skipping saving Final Return plot - no valid data plotted.")

    plt.close()
except Exception as e:
    print(f"Error generating/saving Final Return Plot: {e}")
    traceback.print_exc()
    plt.close()


# --- Plot 4: Cumulative Strategy Choices (MABs) ---
try:
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Filter only algos with valid choice counts
    valid_choice_algos = {
        name: data['choice_counts'] for name, data in processed_mab_results.items()
        if data['choice_counts'].sum() > 0 # Check if any choices were made
    }
    if not valid_choice_algos:
         raise ValueError("No valid choice data found for any MAB algorithm.")

    choice_data = pd.DataFrame(valid_choice_algos).T # Algos as rows

    if choice_data.shape[1] != len(mab_arm_strategy_names):
        raise ValueError(f"Choice data columns ({choice_data.shape[1]}) != MAB arms ({len(mab_arm_strategy_names)}).")

    choice_data.columns = mab_arm_strategy_names
    choice_data.plot(kind='bar', stacked=True, ax=ax, color=strategy_colors, width=0.8)

    ax.set_title('MAB Strategy Choices (Total Across Valid Runs)')
    ax.set_xlabel('MAB Algorithm')
    ax.set_ylabel('Total Times Chosen')
    ax.legend(title='Base Strategies Chosen', bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.savefig('mab_strategy_choices.png', bbox_inches='tight')
    print("Saved mab_strategy_choices.png")
    plt.close()

except ValueError as e: # Catch specific error for no/inconsistent data
     print(f"Skipping saving MAB Choices plot - {e}")
     plt.close()
except Exception as e:
    print(f"Error generating/saving MAB Choices Plot: {e}")
    traceback.print_exc()
    plt.close()


# --- Plot 5: Cumulative Returns of Pure Strategies ---
try:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    time_steps = np.arange(sim_length) # Define time steps for x-axis
    plot_successful = False
    for i, name in enumerate(pure_strategy_names):
        returns = pure_strategy_results.get(name, np.array([]))
        # Plot only if results exist and are not all NaN
        if returns.ndim > 0 and returns.size == sim_length and not np.isnan(returns).all():
            ax.plot(time_steps, returns, label=name, color=strategy_colors[i % len(strategy_colors)], alpha=0.8)
            plot_successful = True
        elif returns.size != sim_length:
             print(f"Plotting Warning (Pure Returns): Length mismatch for {name} ({returns.size}) vs sim_length ({sim_length}).")


    if plot_successful:
        ax.set_title('Pure Strategy Cumulative Returns')
        ax.set_xlabel(f'Time Steps ({sim_length} total)')
        ax.set_ylabel('Cumulative Return')
        ax.legend(fontsize='medium')
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('pure_strategy_returns.png', bbox_inches='tight')
        print("Saved pure_strategy_returns.png")
    else:
        print("Skipping saving Pure Returns plot - no valid data plotted.")
    plt.close()

except Exception as e:
    print(f"Error generating/saving Pure Returns Plot: {e}")
    traceback.print_exc()
    plt.close()


# --- Plot 6: Distribution of Final MAB Returns (Box Plot) ---
try:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Prepare data for boxplot: list of non-empty arrays, one for each MAB
    valid_boxplot_data = []
    valid_labels = []
    for algo_name in mab_algo_names:
         if algo_name in processed_mab_results:
             final_returns = processed_mab_results[algo_name]['all_final_returns']
             # Ensure it's a list/array with actual numbers
             if isinstance(final_returns, (list, np.ndarray)) and len(final_returns) > 0:
                 valid_boxplot_data.append(final_returns)
                 valid_labels.append(algo_name)

    if valid_boxplot_data:
        bp = ax.boxplot(valid_boxplot_data, patch_artist=True, showfliers=False, labels=valid_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Apply colors matching the order of valid_labels/valid_boxplot_data
        for i, patch in enumerate(bp['boxes']):
             original_algo_name = valid_labels[i]
             original_index = mab_algo_names.index(original_algo_name)
             patch.set_facecolor(mab_colors[original_index])
             patch.set_alpha(0.6)

        ax.set_title('Distribution of Final MAB Returns (Across Runs)')
        ax.set_ylabel('Final Cumulative Return')
        ax.set_xlabel('MAB Algorithm')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.savefig('mab_final_return_distribution.png', bbox_inches='tight')
        print("Saved mab_final_return_distribution.png")

    else:
        print("Skipping saving MAB Return Distribution plot - no valid data.")
        # No need to save an empty frame
    plt.close()

except Exception as e:
    print(f"Error generating/saving MAB Return Distribution Plot: {e}")
    traceback.print_exc()
    plt.close()


print("\nComparison script finished.")