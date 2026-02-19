"""
Script to train the Deep Neural Network Surrogate for Option Pricing.
Uses PyTorch for model training and sklearn for data preprocessing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import matplotlib.pyplot as plt
from pricing_engines import DeepSurrogateModel

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptionPricingDataset(Dataset):
    """Custom PyTorch Dataset for Option Data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(data_path: str, model_path: str, scaler_path: str, epochs: int = 50, batch_size: int = 64):
    """
    Trains the Deep Surrogate Model using data from data_path and saves artifacts.
    Returns: model, scaler
    """
    # 1. Load Data
    logger.info(f"Loading dataset from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {data_path}!")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None, None
    
    logger.info(f"Dataset shape: {df.shape}")
    
    # Features and Target
    X = df.drop(columns=['price']).values
    y = df['price'].values

    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Ensure directory exists for scaler
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Datasets & Loaders
    train_dataset = OptionPricingDataset(X_train_scaled, y_train)
    val_dataset = OptionPricingDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = DeepSurrogateModel(input_dim=X_train_scaled.shape[1]).to(device) # Assuming DeepSurrogateModel still needs input_dim
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # Using original LR and WD
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    logger.info("Starting training loop...")
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * X_batch.size(0)
            
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                outputs = model(X_val_batch)
                loss = criterion(outputs, y_val_batch)
                epoch_val_loss += loss.item() * X_val_batch.size(0)
                
        epoch_val_loss /= len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved to {model_path} with Val Loss: {epoch_val_loss:.6f}")

    logger.info("Training complete.")
    
    # Save Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.yscale('log')
    plt.title('Deep Surrogate Training: Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Huber) - Log Scale')
    plt.legend()
    plt.grid(True, alpha=0.5)
    lc_path = os.path.join(os.path.dirname(model_path), 'learning_curve.png')
    plt.savefig(lc_path)
    logger.info(f"Saved {lc_path}")
    
    # Load Best Model for return (since model in memory is last epoch)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, scaler

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Deep Surrogate Model.")
    parser.add_argument("--data_path", type=str, default='heston_training_data.parquet', help="Path to training data parquet")
    parser.add_argument("--model_path", type=str, default='models/best_surrogate.pth', help="Path to save trained model")
    parser.add_argument("--scaler_path", type=str, default='models/scaler.joblib', help="Path to save scaler")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Training batch size")
    
    args = parser.parse_args()

    trained_model, trained_scaler = train_model(
        data_path=args.data_path,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    if trained_model and trained_scaler:
        logger.info("Model and scaler successfully trained and returned.")
    else:
        logger.error("Model training failed.")
