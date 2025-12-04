# src/data_ingestion.py
from pyspark.sql import SparkSession
import logging

logger = logging.getLogger(__name__)

def create_spark_session(app_name="EnergyAnalysis", driver_memory="4g"):
    """Create and configure Spark session"""
    logger.info("Creating Spark session...")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created successfully")
    return spark

def load_data(spark, file_path, sep=';', header=True):
    """Load CSV data with validation"""
    logger.info(f"Loading data from {file_path}...")
    
    # Read with inferSchema=False to keep everything as strings initially
    df = spark.read.csv(
        file_path,
        sep=sep,
        header=header,
        inferSchema=False  # Changed to False to prevent auto-casting
    )
    
    row_count = df.count()
    col_count = len(df.columns)
    
    logger.info(f"Data loaded: {row_count:,} rows, {col_count} columns")
    
    # Show sample
    logger.info("Sample data:")
    df.show(5, truncate=False)
    
    return df

def validate_data(df):
    """Validate data quality"""
    logger.info("Validating data quality...")
    
    # Check for empty dataframe
    if df.count() == 0:
        raise ValueError("Dataset is empty!")
    
    # Check required columns
    required_cols = ['Date', 'Time', 'Global_active_power']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Data validation passed")
    return True