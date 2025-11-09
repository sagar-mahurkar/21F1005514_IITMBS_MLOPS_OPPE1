import pandas as pd
import glob
import os

def process_folder(folder_path):
    """
    Processes all CSV files in a folder:
    - Converts timestamps to datetime
    - Sorts by time and fills missing values
    - Computes rolling averages and sums (10-minute windows)
    - Generates a binary target: 1 if next 5-min close > current close, else 0
    - Combines all processed files into a single DataFrame
    - Returns the combined DataFrame
    """
    
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
    
    processed_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)

        # --- Extract stock name from filename ---
        stock_name = os.path.basename(file).replace("__EQ__NSE__NSE__MINUTE.csv", "")
        df['stock_name'] = stock_name

        # --- Convert timestamp and sort ---
        if 'timestamp' not in df.columns:
            raise KeyError(f"'timestamp' column missing in {file}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)

        # --- Fill missing values ---
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # --- Feature 1: 10-min rolling average of close price ---
        df['rolling_avg_10'] = df['close'].rolling(window='10min', min_periods=1).mean()

        # --- Feature 2: Total volume traded over last 10 minutes ---
        df['volume_sum_10'] = df['volume'].rolling(window='10min', min_periods=1).sum()

        # --- Feature 3: Binary target (future price movement in 5 minutes) ---
        df['close_5min_future'] = df['close'].shift(-5)  # future close price after 5 rows
        df['target'] = (df['close_5min_future'] > df['close']).astype(int)

        # --- Drop incomplete rows ---
        df.dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'], inplace=True)

        # --- Drop unnecessary intermediate column ---
        df.drop(columns=['close_5min_future'], inplace=True, errors='ignore')

        processed_dfs.append(df)

        print(f"✅ Processed file: {os.path.basename(file)} | Rows: {len(df)}")

    # --- Combine all processed data ---
    combined_df = pd.concat(processed_dfs, ignore_index=False)
    combined_df.reset_index(inplace=True)  # keep timestamp as column

    print(f"\n✅ All files processed from: {folder_path}")
    print(f"📊 Combined data shape: {combined_df.shape}")
    print(f"🕒 Time range: {combined_df['timestamp'].min()} → {combined_df['timestamp'].max()}")

    return combined_df


# --- Example usage ---
if __name__ == "__main__":
    folder = "StockAnalyticaData/v0"
    combined_df = process_folder(folder)

    # --- Export combined data to current directory ---
    output_path = "data.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n💾 Combined data exported to: {os.path.abspath(output_path)}")
