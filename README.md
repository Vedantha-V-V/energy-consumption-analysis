# Energy Consumption Analysis

Big Data Analytics pipeline for predicting household energy consumption using Apache Spark and PySpark MLlib.

## Project Overview

This project implements a complete end-to-end pipeline for analyzing and predicting energy consumption patterns:

- **Data Ingestion**: Load and validate raw household power consumption data
- **Data Processing**: Clean, transform, and prepare data for analysis
- **Feature Engineering**: Create time-based and domain-specific features
- **Machine Learning**: Train and evaluate Linear Regression model
- **Predictions**: Generate consumption forecasts
- **Visualization**: Create insightful charts and dashboards

## Architecture
```
Raw Data → Ingestion → Processing → Feature Engineering → 
ML Training → Predictions → Visualizations
```

## Quick Start

### Prerequisites
- Python 3.7+
- Apache Spark 3.0+
- 4GB RAM minimum

### Installation
```bash
# Clone repository
git clone https://github.com/Vedantha-V-V/energy-consumption-analysis.git
cd energy-consumption-analysis

# Install dependencies
pip install -r requirements.txt

# Configure Java PATH
export PATH=$JAVA_HOME/bin:$PATH

# Download dataset
# Place household_power_consumption.txt in data/raw/
```

### Run Pipeline
```bash
python main.py
```

## Results

The pipeline achieves:
- **RMSE**: ~0.45 kW
- **R²**: ~0.92
- **MAE**: ~0.32 kW

## Project Structure

See structure above in implementation.

## Key Features

- Modular, reusable code
- Comprehensive logging
- Automated model evaluation
- Production-ready architecture
- Clear visualizations

## Author

Vedantha V V