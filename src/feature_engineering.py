# src/feature_engineering.py
from pyspark.sql.functions import *
import logging

logger = logging.getLogger(__name__)

def create_time_features(df):
    """Create time-based features"""
    logger.info("Creating time-based features...")
    
    df = df.withColumn('Hour', hour(col('DateTime'))) \
           .withColumn('DayOfWeek', dayofweek(col('DateTime'))) \
           .withColumn('Month', month(col('DateTime'))) \
           .withColumn('Year', year(col('DateTime'))) \
           .withColumn('DayOfMonth', dayofmonth(col('DateTime')))
    
    return df

def create_peak_hour_feature(df):
    """Create peak/off-peak indicator"""
    logger.info("Creating peak hour indicator...")
    
    df = df.withColumn(
        'IsPeakHour',
        when((col('Hour') >= 7) & (col('Hour') <= 10), 1)
        .when((col('Hour') >= 18) & (col('Hour') <= 22), 1)
        .otherwise(0)
    )
    
    return df

def create_seasonal_features(df):
    """Create seasonal indicators"""
    logger.info("Creating seasonal features...")
    
    df = df.withColumn(
        'Season',
        when(col('Month').isin([12, 1, 2]), 'Winter')
        .when(col('Month').isin([3, 4, 5]), 'Spring')
        .when(col('Month').isin([6, 7, 8]), 'Summer')
        .otherwise('Fall')
    )
    
    return df

def engineer_all_features(df):
    """Apply all feature engineering steps"""
    logger.info("Starting feature engineering pipeline...")
    
    df = create_time_features(df)
    df = create_peak_hour_feature(df)
    df = create_seasonal_features(df)
    
    logger.info(f"Feature engineering complete. Total columns: {len(df.columns)}")
    return df