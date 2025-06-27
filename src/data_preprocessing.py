import pandas as pd


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
  
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])

   
    df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
    df['TX_DAY'] = df['TX_DATETIME'].dt.day

   
    df['HIGH_AMOUNT'] = df['TX_AMOUNT'].apply(lambda x: 1 if x > 220 else 0)

    return df
