import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

LATENCY_LOG_FILE = "latency_log.csv"  

def plot_latencies(df):
    """Plots various latency distributions and time series."""
    if df.empty:
        print("Latency data frame is empty. No plots will be generated.")
        return

    df['app_timestamp_ms'] = pd.to_datetime(df['app_timestamp_ms'], unit='ms', errors='coerce')
    numeric_cols = ['data_processing_latency_ms', 'ui_update_latency_ms', 'end_to_end_latency_ms']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['app_timestamp_ms'] + numeric_cols, inplace=True) # Drop rows where conversion failed

    if df.empty:
        print("No valid numeric latency data found after cleaning. No plots will be generated.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Histograms and Box Plots 
    fig1, axes1 = plt.subplots(len(numeric_cols), 2, figsize=(15, 5 * len(numeric_cols)))
    fig1.suptitle('Latency Distributions', fontsize=16)

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes1[i, 0])
        axes1[i, 0].set_title(f'Distribution of {col}')
        axes1[i, 0].set_xlabel('Latency (ms)')
        axes1[i, 0].set_ylabel('Frequency')
        sns.boxplot(x=df[col], ax=axes1[i, 1])
        axes1[i, 1].set_title(f'Box Plot of {col}')
        axes1[i, 1].set_xlabel('Latency (ms)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()

    # 2. Time Series Plots
    fig2, axes2 = plt.subplots(len(numeric_cols), 1, figsize=(15, 4 * len(numeric_cols)), sharex=True)
    fig2.suptitle('Latencies Over Time', fontsize=16)

    for i, col in enumerate(numeric_cols):
        axes2[i].plot(df['app_timestamp_ms'], df[col], label=col, marker='.', linestyle='-', markersize=2)
        axes2[i].set_ylabel('Latency (ms)')
        axes2[i].legend()
        axes2[i].grid(True)
    
    axes2[-1].set_xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # 3. Summary Statistics
    print("\n--- Latency Summary Statistics (ms) ---")
    summary = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max', 
                                    lambda x: x.quantile(0.90), 
                                    lambda x: x.quantile(0.95),
                                    lambda x: x.quantile(0.99)]).rename(
                                        index={'<lambda_0>': '90th Pctl', 
                                               '<lambda_1>': '95th Pctl',
                                               '<lambda_2>': '99th Pctl'}
                                    )
    print(summary)


if __name__ == "__main__":
    try:
        latency_df = pd.read_csv(LATENCY_LOG_FILE)
        if latency_df.empty:
            print(f"Log file '{LATENCY_LOG_FILE}' is empty.")
        else:
            print(f"Loaded {len(latency_df)} records from '{LATENCY_LOG_FILE}'.")
            plot_latencies(latency_df)
    except FileNotFoundError:
        print(f"Error: Latency log file '{LATENCY_LOG_FILE}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: Latency log file '{LATENCY_LOG_FILE}' is empty or not a valid CSV.")
    except Exception as e:
        print(f"An error occurred: {e}")