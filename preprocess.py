import pandas as pd
import glob
import os

def process_folder(folder_path):
    """
    Processes all CSV files in a folder:
    - Converts timestamps to datetime
    - Sorts by time and fills missing values
    - Computes rolling averages and sums (10-minute windows)
    - Generates binary target: 1 if next 5-min close > current close
    - Returns combined DataFrame
    """
    
    csv_files = glob.glob(f"{folder_path}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")
    
    processed_dfs = []

    for file in csv_files:
        df = pd.read_csv(file)

        # Extract stock name from filename
        stock_name = os.path.basename(file).replace("__EQ__NSE__NSE__MINUTE.csv", "")
        df['stock_name'] = stock_name

        # Convert timestamp
        if 'timestamp' not in df.columns:
            raise KeyError(f"'timestamp' column missing in {file}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)

        # Fill missing values
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Rolling features (10-minute window)
        df['rolling_avg_10'] = df['close'].rolling(window='10min', min_periods=1).mean()
        df['volume_sum_10'] = df['volume'].rolling(window='10min', min_periods=1).sum()

        # Future price and target
        df['close_5min_future'] = df['close'].shift(-5)
        df['target'] = (df['close_5min_future'] > df['close']).astype(int)

        # Clean up
        df.dropna(subset=['rolling_avg_10', 'volume_sum_10', 'target'], inplace=True)
        df.drop(columns=['close_5min_future'], inplace=True, errors='ignore')

        processed_dfs.append(df)
        print(f"✅ Processed: {os.path.basename(file)} | Rows: {len(df)}")

    # Combine all processed data
    combined_df = pd.concat(processed_dfs, ignore_index=False)
    combined_df.reset_index(inplace=True)  # keep timestamp as a column

    print(f"\n✅ Folder processed: {folder_path}")
    print(f"📊 Combined shape: {combined_df.shape}")
    return combined_df


# --- Process and combine both folders ---
if __name__ == "__main__":
    folder_v0 = "StockAnalyticaData/v0"
    folder_v1 = "StockAnalyticaData/v1"

    print("🚀 Processing v0 folder...")
    df_v0 = process_folder(folder_v0)

    print("\n🚀 Processing v1 folder...")
    df_v1 = process_folder(folder_v1)

    # Combine v0 and v1
    combined_df = pd.concat([df_v0, df_v1], ignore_index=True)
    combined_df.sort_values(by=['timestamp', 'stock_name'], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Export final combined data
    output_path = "data.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n💾 Final combined data exported to: {os.path.abspath(output_path)}")
    print(f"✅ Total rows: {len(combined_df)} | Columns: {list(combined_df.columns)}")
