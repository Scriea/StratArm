import pandas as pd
import numpy as np # Import numpy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def preprocess_data(df):
    """Cleans and preprocesses raw financial data."""
    # Ensure 'Date' column exists and convert to datetime
    if 'Date' not in df.columns:
        raise ValueError("DataFrame must contain 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'])

    # Clean numerical columns (handle potential errors during conversion)
    for col in ['Price', 'Open', 'High', 'Low']:
        if col in df.columns:
            try:
                # Remove commas, convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
            except Exception as e:
                print(f"Warning: Could not process column '{col}'. Error: {e}")
                # Decide how to handle: drop column, fill NaN later, or raise error
                df[col] = np.nan # Set to NaN if conversion fails
        else:
             print(f"Warning: Expected column '{col}' not found.")

    # Clean Volume column if it exists
    if 'Vol.' in df.columns:
        # Function to convert volume strings (K, M, B, T) to numbers
        def convert_vol(x):
            if pd.isna(x): return np.nan
            x = str(x).strip().upper()
            if not x or x == '-': return np.nan # Handle empty or placeholder strings

            num_part = x
            multiplier = 1

            if x.endswith('K'):
                multiplier = 1e3
                num_part = x[:-1]
            elif x.endswith('M'):
                multiplier = 1e6
                num_part = x[:-1]
            elif x.endswith('B'):
                multiplier = 1e9
                num_part = x[:-1]
            elif x.endswith('T'):
                multiplier = 1e12
                num_part = x[:-1]

            try:
                return float(num_part) * multiplier
            except ValueError:
                return np.nan # Return NaN if conversion fails

        df['Vol.'] = df['Vol.'].apply(convert_vol)
    else:
        print("Warning: Column 'Vol.' not found.")

    # Clean Change % column if it exists
    if 'Change %' in df.columns:
        try:
            df['Change %'] = df['Change %'].astype(str).str.replace('%', '', regex=False)
            df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce') / 100.0
        except Exception as e:
            print(f"Warning: Could not process column 'Change %'. Error: {e}")
            df['Change %'] = np.nan
    else:
        print("Warning: Column 'Change %' not found.")

    # Sort by date (essential for time series analysis)
    df = df.sort_values('Date').reset_index(drop=True)

    # Optional: Handle NaNs introduced during cleaning (e.g., fill or drop)
    # df.fillna(method='ffill', inplace=True) # Example: Forward fill

    return df

def calculate_features(df):
    """Calculates technical indicators required by strategies."""
    df_feat = df.copy() # Work on a copy to avoid modifying original df

    if 'Price' not in df_feat.columns:
        raise ValueError("DataFrame must contain 'Price' column to calculate features.")

    # --- Simple Moving Averages ---
    df_feat['SMA_20'] = df_feat['Price'].rolling(20).mean()
    df_feat['SMA_50'] = df_feat['Price'].rolling(50).mean()

    # --- RSI (Relative Strength Index) ---
    delta = df_feat['Price'].diff()
    gain = delta.where(delta > 0, 0.0).fillna(0) # Handle potential NaNs from diff()
    loss = -delta.where(delta < 0, 0.0).fillna(0) # Handle potential NaNs from diff()

    # Use Exponential Moving Average (EMA) for RSI calculation (common practice)
    # Or use Simple Moving Average (SMA) as originally coded
    window_rsi = 14
    avg_gain = gain.rolling(window=window_rsi, min_periods=window_rsi).mean() # SMA
    avg_loss = loss.rolling(window=window_rsi, min_periods=window_rsi).mean() # SMA
    # Or use EMA:
    # avg_gain = gain.ewm(com=window_rsi - 1, adjust=False).mean()
    # avg_loss = loss.ewm(com=window_rsi - 1, adjust=False).mean()


    rs = avg_gain / avg_loss
    # Handle division by zero if avg_loss is 0
    df_feat['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df_feat['RSI'].replace([np.inf, -np.inf], 100, inplace=True) # If avg_loss is 0, RSI is 100
    df_feat['RSI'].fillna(50, inplace=True) # Fill initial NaNs with neutral 50


    # --- Bollinger Bands ---
    window_bb = 20
    df_feat['BB_MA20'] = df_feat['Price'].rolling(window=window_bb).mean()
    df_feat['BB_STD20'] = df_feat['Price'].rolling(window=window_bb).std()
    df_feat['BB_UPPER'] = df_feat['BB_MA20'] + 2 * df_feat['BB_STD20']
    df_feat['BB_LOWER'] = df_feat['BB_MA20'] - 2 * df_feat['BB_STD20']

    # --- Drop rows with NaNs introduced by rolling calculations ---
    # This ensures strategies only operate on full data
    initial_len = len(df_feat)
    df_feat = df_feat.dropna()
    print(f"Dropped {initial_len - len(df_feat)} rows with NaNs after feature calculation.")

    return df_feat



def plot_data(df):
    # --- Data Cleaning and Type Conversion ---
    # Ensure your DataFrame is named 'data' before running this section

    data = df.copy() # Work on a copy to avoid modifying original df
    # Optional: Print initial data types and head for debugging

    try:
        # 1. Convert 'Date' column to datetime objects
        #    (Adjust format if necessary, e.g., add dayfirst=True or format='%d-%b-%Y')
        if 'Date' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                try:
                    data['Date'] = pd.to_datetime(data['Date'])
                except Exception as e:
                    print(f"Error converting 'Date' column: {e}. Please check format.")
                    # Consider adding specific format hints if auto-parsing fails
                    # e.g., data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
                    raise # Stop if date conversion fails critically
        else:
            print("Error: 'Date' column not found.")
            # Add logic here if date is in the index already, or raise error


        # 2. Convert numeric columns ('Price', 'Open', 'High', 'Low')
        #    Remove commas first, then convert to numeric (float)
        numeric_cols = ['Price', 'Open', 'High', 'Low']
        for col in numeric_cols:
            if col in data.columns:
                # Only clean/convert if it's currently an object type
                if data[col].dtype == 'object':
                    data[col] = data[col].str.replace(',', '', regex=False)
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                # If it's already numeric but maybe int, ensure it's float
                elif pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].astype(float)


        # 3. Convert 'Vol.' (Volume) column
        #    Handle suffixes like 'K' (thousands) and 'M' (millions)
        if 'Vol.' in data.columns:
            # Only process if it's object type
            if data['Vol.'].dtype == 'object':
                def convert_volume(vol_str):
                    if pd.isna(vol_str): return np.nan
                    vol_str = str(vol_str).strip()
                    if vol_str == '-' or vol_str == '':
                        return np.nan
                    # Remove trailing '.0' if it exists before checking K/M
                    if vol_str.endswith('.0'):
                        vol_str = vol_str[:-2]
                    if 'M' in vol_str.upper():
                        return pd.to_numeric(vol_str.upper().replace('M', ''), errors='coerce') * 1_000_000
                    elif 'K' in vol_str.upper():
                        return pd.to_numeric(vol_str.upper().replace('K', ''), errors='coerce') * 1_000
                    else:
                        return pd.to_numeric(vol_str, errors='coerce')
                data['Vol.'] = data['Vol.'].apply(convert_volume)
            elif pd.api.types.is_numeric_dtype(data['Vol.']):
                data['Vol.'] = data['Vol.'].astype(float) # Ensure float if already numeric


        # 4. Convert 'Change %' column
        #    Remove '%' sign, then convert to numeric (float)
        if 'Change %' in data.columns:
            # Only process if it's object type
            if data['Change %'].dtype == 'object':
                data['Change %'] = data['Change %'].str.replace('%', '', regex=False)
                data['Change %'] = pd.to_numeric(data['Change %'], errors='coerce')
            elif pd.api.types.is_numeric_dtype(data['Change %']):
                data['Change %'] = data['Change %'].astype(float) # Ensure float


        # --- Optional: Post-Conversion Check ---
        print("Data types after conversion:")
        print(data.info())
        print("\nNaN values per column after conversion:")
        print(data.isnull().sum())


        # --- Prepare for Plotting ---

        # 5. Set 'Date' column as the index (if it's not already)
        if 'Date' in data.columns:
            data.set_index('Date', inplace=True)
        elif not isinstance(data.index, pd.DatetimeIndex):
            print("Warning: 'Date' column not found and index is not DatetimeIndex.")
            # Attempt to convert index if possible, otherwise plotting might fail
            try:
                data.index = pd.to_datetime(data.index)
                print("Converted index to DatetimeIndex.")
            except Exception as e:
                print(f"Could not convert index to DatetimeIndex: {e}")


        # 6. Sort the data by date index (important for time series plotting)
        if isinstance(data.index, pd.DatetimeIndex):
            data.sort_index(inplace=True)
        else:
            print("Warning: Index is not DatetimeIndex, cannot sort by date.")


        # --- Plotting ---

        # 7. Choose the column to plot (e.g., 'Price')
        column_to_plot = 'Price' # <-- You can change this to 'Open', 'High', 'Low', etc.

        if column_to_plot not in data.columns:
            print(f"\nError: Column '{column_to_plot}' not found in the DataFrame.")
        elif not isinstance(data.index, pd.DatetimeIndex):
            print(f"\nError: Cannot plot time series because the index is not a DatetimeIndex.")
        elif data[column_to_plot].isnull().all():
            print(f"\nError: Column '{column_to_plot}' contains only NaN values.")
        else:
            print(f"\nPlotting '{column_to_plot}' vs Date...")
            plt.figure(figsize=(14, 7)) # Adjust figure size as needed
            plt.plot(data.index, data[column_to_plot], label=column_to_plot)

            # --- Customize Plot ---
            plt.title(f'BSE Sensex Time Series - {column_to_plot}')
            plt.xlabel('Date')
            plt.ylabel(column_to_plot)
            plt.legend()
            plt.grid(True)

            # Improve date formatting on x-axis
            # Adjust locator interval based on your data's time span
            # e.g., MonthLocator(interval=6) for 6 months, YearLocator() for years
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3)) # Adjust interval as needed
            plt.gcf().autofmt_xdate() # Auto-rotate date labels

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"An error occurred during the process: {e}")
        # Consider printing data types or head again for debugging
        # print(data.info())
        # print(data.head())
    return data # Return the cleaned DataFrame for further use