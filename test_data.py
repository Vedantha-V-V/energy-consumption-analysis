from pyspark.sql import SparkSession
from src.data_ingestion import create_spark_session, load_data
from src.data_processing import clean_data, create_datetime_column
from src.feature_engineering import engineer_all_features
import config

# Create Spark session
spark = create_spark_session(**config.SPARK_CONFIG)

# Load and process data
data_path = config.RAW_DATA_DIR / config.DATA_FILE
df = load_data(spark, str(data_path), sep=config.DATA_SEPARATOR)
df = clean_data(df)
df = create_datetime_column(df)
df = engineer_all_features(df)

# Check what we have
print("\n=== DATAFRAME INFO ===")
print(f"Total rows: {df.count():,}")
print(f"\nColumns: {df.columns}")
print("\n=== SAMPLE DATA ===")
df.select('Hour', 'DayOfWeek', 'Month', 'Year', 'IsPeakHour', 'Global_active_power').show(10)

# Check for nulls in these specific columns
print("\n=== NULL CHECK ===")
from pyspark.sql.functions import col, sum as spark_sum
null_counts = df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c) 
    for c in ['Hour', 'DayOfWeek', 'Month', 'Year', 'IsPeakHour', 'Global_active_power']
])
null_counts.show()

spark.stop()