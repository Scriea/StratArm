import pandas as pd 

def preprocess_data(df):
    # Convert columns to appropriate types
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean numerical columns
    for col in ['Price', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float) 
        
    # Clean Volume column
    def convert_vol(x):
        if 'K' in x:
            return float(x.replace('K', '')) * 1e3
        elif 'M' in x:
            return float(x.replace('M', '')) * 1e6
        elif 'B' in x:
            return float(x.replace('B', '')) * 1e9
        elif 'T' in x:  
            return float(x.replace('T', '')) * 1e12
        return float(x)
    
    df['Vol.'] = df['Vol.'].apply(convert_vol)
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def calculate_features(df):
    # Simple Moving Averages
    df['SMA_20'] = df['Price'].rolling(20).mean()
    df['SMA_50'] = df['Price'].rolling(50).mean()
    
    # RSI
    delta = df['Price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_MA20'] = df['Price'].rolling(20).mean()
    df['BB_STD20'] = df['Price'].rolling(20).std()
    df['BB_UPPER'] = df['BB_MA20'] + 2*df['BB_STD20']
    df['BB_LOWER'] = df['BB_MA20'] - 2*df['BB_STD20']
    
    return df.dropna()