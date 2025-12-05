from flask import Flask, render_template, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pandas as pd
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Paths - using the existing structure
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.parquet"
MODEL_PATH = DATA_DIR / "models" / "energy_model"
METRICS_PATH = BASE_DIR / "results" / "metrics.json"

# Global variables for cached data
spark = None
df_processed = None
model = None
metrics = None

def initialize_spark():
    """Initialize Spark session"""
    global spark
    if spark is None:
        logger.info("Initializing Spark session...")
        spark = SparkSession.builder \
            .appName("EnergyVisualizationApp") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "10") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
    return spark

def load_data():
    """Load processed data and model"""
    global df_processed, model, metrics
    
    if df_processed is None:
        logger.info("Loading processed data...")
        spark_session = initialize_spark()
        df_processed = spark_session.read.parquet(str(PROCESSED_DATA_PATH))
        logger.info(f"Loaded {df_processed.count():,} rows")
    
    if model is None and MODEL_PATH.exists():
        logger.info("Loading trained model...")
        model = PipelineModel.load(str(MODEL_PATH))
    
    if metrics is None and METRICS_PATH.exists():
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
    
    return df_processed, model, metrics

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    """Get model performance metrics"""
    _, _, metrics_data = load_data()
    return jsonify(metrics_data)

@app.route('/api/hourly_consumption')
def hourly_consumption():
    """Get hourly consumption data"""
    df, _, _ = load_data()
    
    hourly_data = df.groupBy('Hour').agg(
        {'Global_active_power': 'avg'}
    ).withColumnRenamed('avg(Global_active_power)', 'avg_power') \
     .orderBy('Hour') \
     .toPandas()
    
    return jsonify({
        'hours': hourly_data['Hour'].tolist(),
        'avg_power': hourly_data['avg_power'].tolist()
    })

@app.route('/api/daily_consumption')
def daily_consumption():
    """Get consumption by day of week"""
    df, _, _ = load_data()
    
    daily_data = df.groupBy('DayOfWeek').agg(
        {'Global_active_power': 'avg'}
    ).withColumnRenamed('avg(Global_active_power)', 'avg_power') \
     .orderBy('DayOfWeek') \
     .toPandas()
    
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    daily_data['DayName'] = daily_data['DayOfWeek'].apply(lambda x: days[x-1])
    
    return jsonify({
        'days': daily_data['DayName'].tolist(),
        'avg_power': daily_data['avg_power'].tolist()
    })

@app.route('/api/monthly_consumption')
def monthly_consumption():
    """Get consumption by month"""
    df, _, _ = load_data()
    
    monthly_data = df.groupBy('Month').agg(
        {'Global_active_power': 'avg'}
    ).withColumnRenamed('avg(Global_active_power)', 'avg_power') \
     .orderBy('Month') \
     .toPandas()
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_data['MonthName'] = monthly_data['Month'].apply(lambda x: months[x-1])
    
    return jsonify({
        'months': monthly_data['MonthName'].tolist(),
        'avg_power': monthly_data['avg_power'].tolist()
    })

@app.route('/api/peak_analysis')
def peak_analysis():
    """Get peak vs off-peak analysis"""
    df, _, _ = load_data()
    
    from pyspark.sql.functions import avg, count
    
    peak_data = df.groupBy('IsPeakHour').agg(
        avg('Global_active_power').alias('avg_power'),
        count('Global_active_power').alias('count')
    ).orderBy('IsPeakHour').toPandas()
    
    return jsonify({
        'labels': ['Off-Peak', 'Peak'],
        'avg_power': peak_data['avg_power'].tolist(),
        'counts': peak_data['count'].tolist()
    })

@app.route('/api/seasonal_analysis')
def seasonal_analysis():
    """Get seasonal consumption patterns"""
    df, _, _ = load_data()
    
    seasonal_data = df.groupBy('Season').agg(
        {'Global_active_power': 'avg'}
    ).withColumnRenamed('avg(Global_active_power)', 'avg_power') \
     .toPandas()
    
    return jsonify({
        'seasons': seasonal_data['Season'].tolist(),
        'avg_power': seasonal_data['avg_power'].tolist()
    })

@app.route('/api/predictions_sample')
def predictions_sample():
    """Get sample predictions for scatter plot"""
    df, model_obj, _ = load_data()
    
    # Take a sample for visualization
    sample_df = df.limit(1000)
    
    if model_obj:
        predictions = model_obj.transform(sample_df)
        pred_data = predictions.select('Global_active_power', 'prediction').toPandas()
    else:
        pred_data = sample_df.select('Global_active_power').toPandas()
        pred_data['prediction'] = pred_data['Global_active_power']
    
    return jsonify({
        'actual': pred_data['Global_active_power'].tolist(),
        'predicted': pred_data['prediction'].tolist()
    })

@app.route('/api/distribution')
def distribution():
    """Get power consumption distribution"""
    df, _, _ = load_data()
    
    # Get distribution data
    dist_data = df.select('Global_active_power').toPandas()
    
    return jsonify({
        'values': dist_data['Global_active_power'].tolist()[:5000]  # Limit for performance
    })

@app.route('/api/stats')
def get_stats():
    """Get summary statistics"""
    df, _, _ = load_data()
    
    stats = df.select('Global_active_power').summary().toPandas()
    
    total_records = df.count()
    
    from pyspark.sql.functions import sum as spark_sum
    total_consumption = df.agg(spark_sum('Global_active_power')).collect()[0][0]
    
    return jsonify({
        'total_records': int(total_records),
        'total_consumption': float(total_consumption) if total_consumption else 0,
        'mean': float(stats[stats['summary'] == 'mean']['Global_active_power'].values[0]),
        'min': float(stats[stats['summary'] == 'min']['Global_active_power'].values[0]),
        'max': float(stats[stats['summary'] == 'max']['Global_active_power'].values[0]),
        'stddev': float(stats[stats['summary'] == 'stddev']['Global_active_power'].values[0])
    })

if __name__ == '__main__':
    # Load data on startup
    logger.info("Starting Flask application...")
    load_data()
    logger.info("Data loaded successfully!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)