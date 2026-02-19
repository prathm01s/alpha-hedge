import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import gc
import joblib
from pricing_engines import BlackScholesEngine, HestonMCEngine, MertonJumpMCEngine, DeepSurrogateEngine
from generate_dataset import generate_data
from train_surrogate import train_model

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
N_TRAIN_SAMPLES = 500_000
N_TEST_SAMPLES = 10_000
EPOCHS = 20
BATCH_SIZE = 2048

# World Definitions for Pipeline
WORLDS = {
    'BS': {'engine': BlackScholesEngine(), 'train_file': 'data_train_BS.parquet', 'test_file': 'data_test_BS.parquet', 'model_path': 'models/model_BS.pth', 'scaler_path': 'models/scaler_BS.joblib'},
    'Heston': {'engine': HestonMCEngine(n_paths=5000, n_steps=30), 'train_file': 'data_train_Heston.parquet', 'test_file': 'data_test_Heston.parquet', 'model_path': 'models/model_Heston.pth', 'scaler_path': 'models/scaler_Heston.joblib'},
    'Merton': {'engine': MertonJumpMCEngine(n_paths=5000, n_steps=30), 'train_file': 'data_train_Merton.parquet', 'test_file': 'data_test_Merton.parquet', 'model_path': 'models/model_Merton.pth', 'scaler_path': 'models/scaler_Merton.joblib'}
}

def clean_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

import argparse

def run_pipeline(output_dir):
    # Ensure output dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Sequential Generation & Training
    for world_name, config in WORLDS.items():
        # ... (Same as before) ...
        # Skips handled by generate_dataset.py now
        logger.info(f"=== Processing World: {world_name} ===")
        
        # A. Generate Data
        logger.info(f"Generating Training Data for {world_name}...")
        generate_data(config['engine'], N_TRAIN_SAMPLES, config['train_file'])
        
        logger.info(f"Generating Test Data for {world_name}...")
        generate_data(config['engine'], N_TEST_SAMPLES, config['test_file'])
        
        clean_memory()
        
        # B. Train Model
        logger.info(f"Training Model for {world_name}...")
        model, scaler = train_model(
            data_path=config['train_file'],
            model_path=config['model_path'],
            scaler_path=config['scaler_path'],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        # C. Cleanup
        del model
        del scaler
        clean_memory()
        logger.info(f"Finished {world_name}. Memory Cleaned.")

    # 2. Evaluation Matrix
    mse_matrix = np.zeros((3, 3))
    world_names_list = list(WORLDS.keys())
    
    logger.info("=== Starting Generalization Evaluation ===")
    
    for i, train_world in enumerate(world_names_list):
        train_config = WORLDS[train_world]
        logger.info(f"Loading Model: {train_world} from {train_config['model_path']}")
        
        surrogate = DeepSurrogateEngine(model_path=train_config['model_path'], scaler_path=train_config['scaler_path'])
        
        for j, test_world in enumerate(world_names_list):
            test_config = WORLDS[test_world]
            logger.info(f"Evaluating Model({train_world}) vs World({test_world})...")
            
            try:
                df_test = pd.read_parquet(test_config['test_file'])
                X_test = df_test.drop(columns=['price']).values
                y_true = df_test['price'].values
                
                scaler = surrogate.scaler
                X_scaled = scaler.transform(X_test)
                
                X_t = torch.tensor(X_scaled, dtype=torch.float32).to(surrogate.device)
                
                surrogate.model.eval()
                with torch.no_grad():
                    y_pred = []
                    for k in range(0, len(X_t), BATCH_SIZE):
                        batch_X = X_t[k:k+BATCH_SIZE]
                        batch_pred = surrogate.model(batch_X)
                        y_pred.append(batch_pred.cpu().numpy().flatten())
                    y_pred = np.concatenate(y_pred)
                    
                mse = np.mean((y_true - y_pred)**2)
                mse_matrix[i, j] = mse
                logger.info(f"MSE({train_world} -> {test_world}) = {mse:.6f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {train_world} on {test_world}: {e}")
                mse_matrix[i, j] = np.nan
            
            clean_memory()
            
        del surrogate
        clean_memory()

    # 3. Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(mse_matrix, annot=True, fmt=".4f", xticklabels=world_names_list, yticklabels=world_names_list, cmap="Reds")
    plt.title("Generalization Error Matrix (MSE)\nRows: Trained Model | Cols: Test World")
    plt.xlabel("Test World (Data Source)")
    plt.ylabel("Trained Model (Architecture)")
    
    save_path = os.path.join(output_dir, "generalization_matrix.png")
    plt.savefig(save_path)
    logger.info(f"Saved {save_path}")
    
    # Print
    df_res = pd.DataFrame(mse_matrix, index=world_names_list, columns=world_names_list)
    print("\nGeneralization Matrix (MSE):")
    print(df_res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save outputs")
    parser.add_argument("--test", action="store_true", help="Run in fast test mode")
    args = parser.parse_args()
    
    if args.test:
        logger.info("TEST MODE DETECTED: Reducing sample sizes and epochs.")
        N_TRAIN_SAMPLES = 100
        N_TEST_SAMPLES = 50
        EPOCHS = 1
        BATCH_SIZE = 32
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    run_pipeline(args.output_dir)
