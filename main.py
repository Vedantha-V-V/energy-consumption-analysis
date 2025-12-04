# main.py
import logging
import json
from datetime import datetime
from pathlib import Path

# Import our modules
from src.data_ingestion import create_spark_session, load_data, validate_data
from src.data_processing import clean_data, create_datetime_column
from src.feature_engineering import engineer_all_features
from src.model_training import train_model, evaluate_model
from src.visualization import generate_all_plots
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner(text, char="="):
    """Print formatted banner"""
    width = 60
    print("\n" + char * width)
    print(f" {text:^{width-2}} ")
    print(char * width + "\n")

def run_pipeline():
    """Run the complete end-to-end pipeline"""
    start_time = datetime.now()
    
    print_banner("ENERGY CONSUMPTION ANALYSIS PIPELINE", "=")
    logger.info(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # ============================================
        # STAGE 1: Data Ingestion
        # ============================================
        print_banner("STAGE 1/6: DATA INGESTION", "-")
        logger.info(f"config.SPARK_CONFIG")
        spark = create_spark_session(**config.SPARK_CONFIG)
        data_path = config.RAW_DATA_DIR / config.DATA_FILE
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {data_path}\n"
                f"Please download 'household_power_consumption.txt' and place it in {config.RAW_DATA_DIR}"
            )
        
        df = load_data(spark, str(data_path), sep=config.DATA_SEPARATOR)
        validate_data(df)
        
        # ============================================
        # STAGE 2: Data Processing
        # ============================================
        print_banner("STAGE 2/6: DATA PROCESSING", "-")
        
        df = clean_data(df)
        df = create_datetime_column(df)
        
        # ============================================
        # STAGE 3: Feature Engineering
        # ============================================
        print_banner("STAGE 3/6: FEATURE ENGINEERING", "-")
        
        df = engineer_all_features(df)
        
        # Save processed data
        processed_path = config.PROCESSED_DATA_DIR / "processed_data.parquet"
        logger.info(f"Saving processed data to {processed_path}...")
        df.write.mode('overwrite').parquet(str(processed_path))
        logger.info("Processed data saved")
        
        # ============================================
        # STAGE 4: Model Training
        # ============================================
        print_banner("STAGE 4/6: MODEL TRAINING", "-")
        
        model, train_data, test_data = train_model(
            df, 
            config.FEATURE_COLS, 
            config.TARGET_COL,
            test_size=1-config.TRAIN_TEST_SPLIT,
            seed=config.RANDOM_SEED,
            max_iter=config.MAX_ITER,
            reg_param=config.REG_PARAM,
            elastic_net=config.ELASTIC_NET_PARAM
        )
        
        # Save model
        model_path = config.MODEL_DIR / "energy_model"
        logger.info(f"Saving model to {model_path}...")
        model.write().overwrite().save(str(model_path))
        logger.info("Model saved")
        
        # ============================================
        # STAGE 5: Model Evaluation
        # ============================================
        print_banner("STAGE 5/6: MODEL EVALUATION", "-")
        
        metrics, predictions = evaluate_model(model, test_data)
        
        # Save metrics
        metrics_path = config.RESULTS_DIR / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save predictions
        predictions_path = config.RESULTS_DIR / "predictions.csv"
        logger.info(f"Saving predictions to {predictions_path}...")
        predictions.select('Global_active_power', 'prediction') \
                   .toPandas() \
                   .to_csv(predictions_path, index=False)
        logger.info("Predictions saved")
        
        # ============================================
        # STAGE 6: Visualization
        # ============================================
        print_banner("STAGE 6/6: VISUALIZATION", "-")
        
        plots_dir = generate_all_plots(df, predictions, config.RESULTS_DIR)
        
        # ============================================
        # Pipeline Summary
        # ============================================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_banner("PIPELINE EXECUTION SUMMARY", "=")
        
        summary = f"""
Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)

MODEL PERFORMANCE:
  → RMSE: {metrics['rmse']:.4f} kW
  → R²: {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%)
  → MAE: {metrics['mae']:.4f} kW

OUTPUT FILES:
  → Processed Data: {processed_path}
  → Trained Model: {model_path}
  → Metrics: {metrics_path}
  → Predictions: {predictions_path}
  → Visualizations: {plots_dir}
        """
        
        print(summary)
        logger.info(summary)
        
        print_banner("PIPELINE COMPLETED SUCCESSFULLY!", "=")
        
        spark.stop()
        return metrics
        
    except Exception as e:
        print_banner("PIPELINE FAILED", "=")
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        metrics = run_pipeline()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        logger.error("Pipeline execution failed")
        exit(1)