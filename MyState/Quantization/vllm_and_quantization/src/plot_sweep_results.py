import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_prepare_data(awq_path, origin_path):
    """Loads and combines the two CSV files into a single DataFrame."""
    try:
        df_awq = pd.read_csv(awq_path)
        df_origin = pd.read_csv(origin_path)
        
        df_awq['model'] = 'Qwen3-0.6B-awq-sym'
        df_origin['model'] = 'Qwen3-0.6B'
        
        # Concatenate the two dataframes
        df = pd.concat([df_awq, df_origin], ignore_index=True)
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the summary CSV files exist.")
        return pd.DataFrame()

def plot_performance_graphs(df, output_dir):
    """Generates and saves performance plots."""
    if df.empty:
        print("DataFrame is empty. No plots will be generated.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Aggregate data: group by request_rate and model, then calculate the mean for key metrics
    # This is important because there are multiple runs for each request rate.
    agg_df = df.groupby(['request_rate', 'model']).agg({
        'output_throughput': 'mean',
        'request_throughput': 'mean',
        'mean_ttft_ms': 'mean',
        'mean_e2el_ms': 'mean'
    }).reset_index()

    # Plot 1: Output Throughput vs. Request Rate
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=agg_df, x='request_rate', y='output_throughput', hue='model', marker='o')
    plt.title('Output Throughput vs. Request Rate')
    plt.xlabel('Request Rate (requests/s)')
    plt.ylabel('Output Throughput (tokens/s)')
    plt.grid(True)
    plt.legend(title='Model')
    output_path = os.path.join(output_dir, 'throughput_vs_request_rate.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

    # Plot 2: Request Goodput vs. Request Rate
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=agg_df, x='request_rate', y='request_throughput', hue='model', marker='o')
    plt.title('Request Goodput vs. Request Rate')
    plt.xlabel('Offered Request Rate (requests/s)')
    plt.ylabel('Goodput (requests/s)')
    plt.grid(True)
    plt.legend(title='Model')
    output_path = os.path.join(output_dir, 'goodput_vs_request_rate.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

    # Plot 3: Latency (TTFT and E2E) vs. Request Rate
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=agg_df, x='request_rate', y='mean_ttft_ms', hue='model', marker='o', linestyle='--')
    sns.lineplot(data=agg_df, x='request_rate', y='mean_e2el_ms', hue='model', marker='o')
    plt.title('Average Latency vs. Request Rate (Solid=E2E, Dashed=TTFT)')
    plt.xlabel('Request Rate (requests/s)')
    plt.ylabel('Latency (ms)')
    plt.grid(True)
    plt.legend(title='Model')
    output_path = os.path.join(output_dir, 'latency_vs_request_rate.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")

if __name__ == '__main__':
    awq_summary_path = '../sweep_res/awq_summary.csv'
    origin_summary_path = '../sweep_res/origin_summary.csv'
    plots_output_dir = '../sweep_res/plots'

    combined_df = load_and_prepare_data(awq_summary_path, origin_summary_path)
    plot_performance_graphs(combined_df, plots_output_dir)
