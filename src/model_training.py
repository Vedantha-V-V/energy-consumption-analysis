# src/model_training.py
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import logging

logger = logging.getLogger(__name__)

def prepare_features(df, feature_cols, target_col):
    """Prepare features for modeling"""
    logger.info(f"Preparing features: {feature_cols}")
    
    # Select relevant columns and drop nulls in one step
    df_model = df.select(feature_cols + [target_col]).na.drop()
    
    # Cache the dataframe for faster subsequent operations
    df_model = df_model.cache()
    
    # Get count
    final_count = df_model.count()
    
    logger.info(f"Features prepared: {final_count:,} rows ready for training")
    
    if final_count == 0:
        # If no data, show what columns we have
        logger.error(f"Available columns in dataframe: {df.columns}")
        raise ValueError(
            "No data remaining after feature preparation! "
            f"Tried to select: {feature_cols + [target_col]}"
        )
    
    return df_model

def create_ml_pipeline(feature_cols, max_iter=100, reg_param=0.1, elastic_net=0.5):
    """Create ML pipeline with preprocessing and model"""
    logger.info("Building ML pipeline...")
    
    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol='features'
    )
    
    # Scale features
    scaler = StandardScaler(
        inputCol='features',
        outputCol='scaled_features',
        withMean=True,
        withStd=True
    )
    
    # Linear Regression model
    lr = LinearRegression(
        featuresCol='scaled_features',
        labelCol='Global_active_power',
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_net
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    
    logger.info("Pipeline stages: VectorAssembler → StandardScaler → LinearRegression")
    logger.info(f"Model params: maxIter={max_iter}, regParam={reg_param}, elasticNet={elastic_net}")
    
    return pipeline

def train_model(df, feature_cols, target_col, test_size=0.2, seed=42, **model_params):
    """Train the model"""
    logger.info("="*50)
    logger.info("Starting Model Training")
    logger.info("="*50)
    
    # Prepare data
    df_model = prepare_features(df, feature_cols, target_col)
    
    # Split data
    train_data, test_data = df_model.randomSplit([1-test_size, test_size], seed=seed)
    
    train_count = train_data.count()
    test_count = test_data.count()
    
    logger.info(f"Data split (seed={seed}):")
    logger.info(f"Training set: {train_count:,} rows ({(1-test_size)*100:.0f}%)")
    logger.info(f"Test set: {test_count:,} rows ({test_size*100:.0f}%)")
    
    # Create and train pipeline (without seed parameter)
    pipeline = create_ml_pipeline(feature_cols, **model_params)
    
    logger.info("Training model... (this may take a few minutes)")
    model = pipeline.fit(train_data)
    
    logger.info("✓ Model training complete!")
    logger.info("="*50)
    
    return model, train_data, test_data

def evaluate_model(model, test_data):
    """Evaluate model performance"""
    logger.info("="*50)
    logger.info("Evaluating Model Performance")
    logger.info("="*50)
    
    # Make predictions
    logger.info("Making predictions on test set...")
    predictions = model.transform(test_data)
    
    # Calculate metrics
    evaluator = RegressionEvaluator(
        labelCol='Global_active_power',
        predictionCol='prediction'
    )
    
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    
    metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae)
    }
    
    logger.info("Model Performance Metrics:")
    logger.info(f"RMSE (Root Mean Squared Error): {rmse:.4f} kW")
    logger.info(f"R² (R-Squared): {r2:.4f} ({r2*100:.1f}%)")
    logger.info(f"MAE (Mean Absolute Error): {mae:.4f} kW")
    logger.info("="*50)
    
    # Show sample predictions
    logger.info("Sample Predictions:")
    predictions.select('Global_active_power', 'prediction').show(10)
    
    return metrics, predictions