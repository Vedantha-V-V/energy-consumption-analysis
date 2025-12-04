# src/data_processing.py
from pyspark.sql.functions import col, concat, lit, to_timestamp
import logging

logger = logging.getLogger(__name__)

def clean_data(df):
    """Clean and prepare data"""
    logger.info("Starting data cleaning...")
    
    initial_count = df.count()
    
    # 1. Filter out rows with '?' (missing values)
    df_clean = df.filter(col('Global_active_power') != '?')
    
    filtered_count = df_clean.count()
    logger.info(f"Removed {initial_count - filtered_count:,} rows with missing values")
    
    # 2. Cast ONLY numeric columns to proper types (NOT Date and Time!)
    df_clean = df_clean.withColumn(
        'Global_active_power', 
        col('Global_active_power').cast('double')
    ).withColumn(
        'Global_reactive_power', 
        col('Global_reactive_power').cast('double')
    ).withColumn(
        'Voltage', 
        col('Voltage').cast('double')
    ).withColumn(
        'Global_intensity',
        col('Global_intensity').cast('double')
    )
    
    # Keep Date and Time as strings - don't cast them!
    
    logger.info("Converted numeric columns to proper types")
    
    # 3. Remove duplicates
    pre_dedup_count = df_clean.count()
    df_clean = df_clean.dropDuplicates(['Date', 'Time'])
    post_dedup_count = df_clean.count()
    
    logger.info(f"Removed {pre_dedup_count - post_dedup_count:,} duplicate rows")
    
    # 4. Remove nulls from target column
    df_clean = df_clean.na.drop(subset=['Global_active_power'])
    
    final_count = df_clean.count()
    logger.info(f"Data cleaning complete: {final_count:,} rows remaining")
    
    return df_clean

def create_datetime_column(df):
    """Create unified datetime column"""
    logger.info("Creating datetime column...")
    
    # Cast to string first
    df = df.withColumn('Date', col('Date').cast('string')) \
           .withColumn('Time', col('Time').cast('string'))
    
    # Use try_to_timestamp which returns NULL on parse failure instead of throwing error
    from pyspark.sql.functions import expr
    
    df = df.withColumn(
        'DateTime',
        expr("try_to_timestamp(concat(Date, ' ', Time), 'd/M/yyyy HH:mm:ss')")
    )
    
    # Count and filter out NULL datetimes
    total_before = df.count()
    df = df.filter(col('DateTime').isNotNull())
    total_after = df.count()
    
    removed = total_before - total_after
    if removed > 0:
        logger.info(f"Removed {removed:,} rows with invalid datetime")
    
    logger.info(f"DateTime column created: {total_after:,} valid rows")
    return df