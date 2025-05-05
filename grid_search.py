# --- grid_search_optimizer.py ---

import pandas as pd
import numpy as np
import itertools
import traceback
import os
import time

# --- Import necessary components from your project structure ---
# Adjust the path (`from src...`) if this file is placed differently relative to 'src'
try:
    from src.strategy import TradingStrategy, SMACrossoverStrategy, RSIStrategy, BollingerStrategy
    from src.data_utils import preprocess_data, calculate_features # Assuming these are needed directly
except ImportError as e:
    print(f"Import Error in grid_search_optimizer.py: {e}")
    print("Please ensure the 'src' directory is correctly structured and accessible.")
    print("You might need to adjust PYTHONPATH or the import path.")
    exit()

def simulate_strategy_return(strategy: TradingStrategy, df_features: pd.DataFrame, price_returns: pd.Series) -> float:
    """
    Simulates a single strategy and returns its final cumulative return.
    """
    sim_length = len(df_features) - 1
    if sim_length <= 0:
        return np.nan

    try:
        signals = strategy.generate_signals(df_features)

        if not signals.index.equals(df_features.index):
             signals = signals.reindex(df_features.index)

        signal_slice = signals.iloc[1:sim_length+1]
        if signal_slice.isnull().values.any():
             # Depending on strategy, NaNs might be expected initially.
             # The simulation loop below handles step-by-step NaN signals correctly.
             # print(f"    Debug: NaNs found in signals for {strategy.name} in sim period.")
             pass

        cumulative_return = 0.0
        aligned_returns = price_returns.reindex(df_features.index).fillna(0)

        for t in range(1, sim_length + 1):
            signal = signals.iloc[t]
            market_return = aligned_returns.iloc[t]
            step_return = 0.0
            if pd.notna(signal): # Only trade if signal is not NaN
                if signal == 1: step_return = market_return
                elif signal == -1: step_return = -market_return
            cumulative_return += step_return

        return cumulative_return

    except Exception as e:
        print(f"    ERROR during single strategy simulation for {strategy.name} with params {getattr(strategy, '__dict__', {})}: {e}")
        # traceback.print_exc() # Uncomment for detailed stack trace during debug
        return np.nan


def perform_grid_search(
    strategy_class: type[TradingStrategy],
    param_grid: dict,
    df_features: pd.DataFrame,
    price_returns: pd.Series
) -> tuple[dict | None, float]:
    """
    Performs a grid search for a given strategy class to maximize final cumulative return.

    Args:
        strategy_class: The class of the strategy (e.g., SMACrossoverStrategy).
        param_grid: A dictionary where keys are parameter names (strings)
                    and values are lists of parameter values to test.
        df_features: DataFrame with features needed by the strategy.
        price_returns: Series of market price percentage returns.

    Returns:
        A tuple containing:
        - dict: The best parameter combination found.
        - float: The highest final cumulative return achieved.
        Returns (None, -np.inf) if no valid combinations are found or simulation fails.
    """
    best_params = None
    best_return = -np.inf
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    param_combinations = list(itertools.product(*param_values))
    total_combinations = len(param_combinations)

    print(f"\n--- Starting Grid Search for {strategy_class.__name__} ---")
    print(f"Testing {total_combinations} parameter combinations...")

    start_time = time.time()
    processed_count = 0
    valid_combinations_tested = 0

    for combo in param_combinations:
        params = dict(zip(param_names, combo))
        valid_combo = True

        # --- Strategy-Specific Constraints ---
        if strategy_class == SMACrossoverStrategy:
            if params.get('short_window', 0) >= params.get('long_window', 1):
                valid_combo = False
        elif strategy_class == RSIStrategy:
             if params.get('low', 0) >= params.get('high', 1):
                 valid_combo = False
        # Add other constraints if needed

        if not valid_combo:
            processed_count += 1
            continue # Skip this invalid combination

        try:
            strategy_instance = strategy_class(**params)
            # Ensure features required by these params are calculable (e.g., enough data for window)
            min_required_data = 0
            if hasattr(strategy_instance, 'long_window'):
                min_required_data = max(min_required_data, strategy_instance.long_window)
            if hasattr(strategy_instance, 'rsi_window'):
                min_required_data = max(min_required_data, strategy_instance.rsi_window + 1) # RSI needs +1 period
            if hasattr(strategy_instance, 'window'): # For Bollinger/SMA
                min_required_data = max(min_required_data, strategy_instance.window)

            # Simple check: if required data exceeds available data after initial NaN drop in feature calc
            # Note: calculate_features already drops NaNs, len(df_features) is after that.
            # A more robust check might involve trying feature calculation itself.
            if min_required_data >= len(df_features):
                 # print(f"  Skipping params {params}: requires {min_required_data} periods, only {len(df_features)} available after feature NaNs.")
                 processed_count += 1
                 continue


            final_return = simulate_strategy_return(strategy_instance, df_features, price_returns)
            valid_combinations_tested += 1

            if pd.notna(final_return) and final_return > best_return:
                best_return = final_return
                best_params = params
                # print(f"  New best found: Params={best_params}, Return={best_return:.4f}") # Verbose logging

        except Exception as e:
             print(f"  ERROR Instantiating/Simulating {strategy_class.__name__} with params {params}: {e}")

        processed_count += 1
        if processed_count % 100 == 0 or processed_count == total_combinations: # Print progress update
             elapsed = time.time() - start_time
             print(f"  Processed {processed_count}/{total_combinations} combinations... [Valid tested: {valid_combinations_tested}, Time: {elapsed:.2f}s]")


    end_time = time.time()
    print(f"--- Grid Search Complete for {strategy_class.__name__} ({end_time - start_time:.2f} seconds) ---")
    if best_params:
        print(f"Best Parameters: {best_params}")
        print(f"Best Final Cumulative Return: {best_return:.4f}")
    else:
        print("No suitable parameters found or simulation failed for all valid combinations.")

    return best_params, best_return


# --- Main execution block for standalone testing ---
if __name__ == "__main__":
    print("Running Grid Search Optimizer as standalone script...")

    # --- Configuration (Match your main script's settings) ---
    DATA_DIR = 'data'
    DATA_FILENAME = 'Download Data - INDEX_US_S&P US_SPX.csv'
    DATA_FILE = os.path.join(DATA_DIR, DATA_FILENAME)

    # --- Load and Prepare Data (Same as in your main script) ---
    print(f"Loading data from {DATA_FILE}...")
    try:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file not found at {DATA_FILE}")
        raw_df = pd.read_csv(DATA_FILE)
        df_processed = preprocess_data(raw_df.copy())
        # Critical: Ensure calculate_features is suitable for all param ranges
        # It might drop more NaNs for larger windows needed in grid search
        # Consider calculating features *inside* the loop if needed, but less efficient.
        df_features = calculate_features(df_processed.copy())

        if df_features.empty:
            raise ValueError("DataFrame is empty after preprocessing and feature calculation.")

        price_returns = df_features['Price'].pct_change()
        price_returns.replace([np.inf, -np.inf], 0, inplace=True)
        price_returns = price_returns.fillna(0)

        sim_length = len(df_features) - 1
        if sim_length <= 0:
           raise ValueError(f"Effective simulation length ({sim_length}) is not positive after feature calculation. Check data/params.")
        print(f"Data prepared. Effective simulation horizon: {sim_length} steps.")


    except Exception as e:
        print(f"Error during data preparation: {e}")
        traceback.print_exc()
        exit()


    # --- Define Parameter Grids ---
    sma_param_grid = {
        'short_window': [10, 15, 20, 30, 50],
        'long_window': [30, 40, 50, 60, 100, 150, 200]
    }
    rsi_param_grid = {
        'rsi_window': [7, 9, 14, 21, 28],
        'low': [0.1,10,20, 30, 40, 50],
        'high': [60, 70, 80, 90, 100, 110]
    }
    bollinger_param_grid = {
        'window': [10, 15, 20, 30, 50],
        'num_std': [1.5, 1.8, 2.0, 2.1, 2.2, 2.5, 3.0]
    }

    # --- Perform Grid Search for each strategy ---
    print("\n=== Performing Grid Search for Base Strategies ===")

    best_sma_params, best_sma_return = perform_grid_search(
        strategy_class=SMACrossoverStrategy,
        param_grid=sma_param_grid,
        df_features=df_features.copy(),
        price_returns=price_returns.copy()
    )

    best_rsi_params, best_rsi_return = perform_grid_search(
        strategy_class=RSIStrategy,
        param_grid=rsi_param_grid,
        df_features=df_features.copy(),
        price_returns=price_returns.copy()
    )

    best_bollinger_params, best_bollinger_return = perform_grid_search(
        strategy_class=BollingerStrategy,
        param_grid=bollinger_param_grid,
        df_features=df_features.copy(),
        price_returns=price_returns.copy()
    )

    # --- Summarize Results ---
    print("\n=== Grid Search Summary ===")
    print(f"SMA Crossover: Best Params={best_sma_params}, Best Return={best_sma_return:.4f}")
    print(f"RSI:           Best Params={best_rsi_params}, Best Return={best_rsi_return:.4f}")
    print(f"Bollinger:     Best Params={best_bollinger_params}, Best Return={best_bollinger_return:.4f}")

    print("\nStandalone grid search finished.")