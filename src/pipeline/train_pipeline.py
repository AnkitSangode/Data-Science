# src/pipeline/training_pipeline.py

from src.components.data_injestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging  # ✅ Add this
import sys

def run_training_pipeline():
    try:
        logging.info("🚀 Starting training pipeline")

        # 1. Data Ingestion
        print("\n🚀 Starting data ingestion...")
        logging.info("Starting data ingestion")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initate_data_ingestion()
        logging.info(f"Data ingestion complete. Train path: {train_path}, Test path: {test_path}")
        print("✅ Data ingestion complete.")

        # 2. Data Transformation
        print("\n🔧 Starting data transformation...")
        logging.info("Starting data transformation")
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)
        logging.info("Data transformation complete")
        print("✅ Data transformation complete.")

        # 3. Model Training
        print("\n🤖 Starting model training...")
        logging.info("Starting model training")
        trainer = ModelTrainer()
        r2_score = trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training complete. R² score: {r2_score:.4f}")
        print(f"✅ Model training complete with R² score: {r2_score:.4f}")

    except Exception as e:
        logging.error("Exception occurred in training pipeline", exc_info=True)
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()
