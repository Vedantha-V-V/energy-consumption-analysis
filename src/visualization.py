# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from pyspark.sql.functions import avg, count
from pathlib import Path

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")

def plot_actual_vs_predicted(predictions_df, save_path):
    """Plot actual vs predicted values"""
    logger.info("Creating actual vs predicted plot...")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions_df['Global_active_power'], 
                predictions_df['prediction'], 
                alpha=0.5)
    
    # Add diagonal line
    min_val = min(predictions_df['Global_active_power'].min(), 
                  predictions_df['prediction'].min())
    max_val = max(predictions_df['Global_active_power'].max(), 
                  predictions_df['prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Actual Power Consumption (kW)')
    plt.ylabel('Predicted Power Consumption (kW)')
    plt.title('Actual vs Predicted Energy Consumption')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Plot saved to {save_path}")

def plot_hourly_consumption(df, save_path):
    """Plot consumption by hour"""
    logger.info("Creating hourly consumption plot...")
    
    hourly_data = df.groupBy('Hour').agg(
        avg('Global_active_power').alias('Avg_Power')
    ).toPandas().sort_values('Hour')
    
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_data['Hour'], hourly_data['Avg_Power'], 
             marker='o', linewidth=2, markersize=8)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Power Consumption (kW)')
    plt.title('Average Energy Consumption by Hour of Day')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Plot saved to {save_path}")

def plot_peak_vs_offpeak(df, save_path):
    """Plot peak vs off-peak analysis"""
    logger.info("Creating peak vs off-peak plot...")
    
    peak_data = df.groupBy('IsPeakHour').agg(
        avg('Global_active_power').alias('Avg_Power'),
        count('*').alias('Count')
    ).toPandas()
    
    peak_data['Period'] = peak_data['IsPeakHour'].map({0: 'Off-Peak', 1: 'Peak'})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Consumption comparison
    ax1.bar(peak_data['Period'], peak_data['Avg_Power'], color=['skyblue', 'coral'])
    ax1.set_ylabel('Average Power Consumption (kW)')
    ax1.set_title('Energy Consumption: Peak vs Off-Peak Hours')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Count comparison
    ax2.bar(peak_data['Period'], peak_data['Count'], color=['skyblue', 'coral'])
    ax2.set_ylabel('Number of Records')
    ax2.set_title('Data Distribution: Peak vs Off-Peak Hours')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Plot saved to {save_path}")

def generate_all_plots(df, predictions, results_dir):
    """Generate all visualizations"""
    logger.info("Generating all visualizations...")
    
    plots_dir = Path(results_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Convert predictions to pandas
    pred_df = predictions.select('Global_active_power', 'prediction').toPandas()
    
    # Generate plots
    plot_actual_vs_predicted(pred_df, plots_dir / "actual_vs_predicted.png")
    plot_hourly_consumption(df, plots_dir / "hourly_consumption.png")
    plot_peak_vs_offpeak(df, plots_dir / "peak_vs_offpeak.png")
    
    logger.info("All visualizations generated successfully")