# config.py
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Spark Configuration
SPARK_CONFIG = {
    "app_name": "EnergyConsumptionAnalysis",
    "driver_memory": "4g"
}

# Data Configuration
DATA_FILE = "household_power_consumption.txt"
DATA_SEPARATOR = ";"

# Feature Configuration
FEATURE_COLS = [
    'Hour', 
    'DayOfWeek', 
    'Month',
    'Year',
    'IsPeakHour'
]

TARGET_COL = 'Global_active_power'

# Model Configuration
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42
MAX_ITER = 100
REG_PARAM = 0.1
ELASTIC_NET_PARAM = 0.5

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'pipeline.log'