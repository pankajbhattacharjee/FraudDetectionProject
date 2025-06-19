import pandas as pd

# 📦 Function to load the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# 🛠️ Function to preprocess the dataset
def preprocess_data(df):
    # Convert TX_DATETIME to datetime object
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

    # Extract transaction hour and day of month
    df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
    df['TX_DAY'] = df['TX_DATETIME'].dt.day

    # Create high amount flag
    df['HIGH_AMOUNT'] = df['TX_AMOUNT'].apply(lambda x: 1 if x > 220 else 0)

    return df
