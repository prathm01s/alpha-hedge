"""
Project Alpha-Hedge: Autonomous Benchmark Pipeline

Generates N datasets, trains a surrogate per dataset, runs M hedging
simulations per model, generates all visualisation plots, and finishes
with the cross-world generalization matrix.

Directory layout:
    benchmark_suite/
    ├── dataset_01/
    │   ├── train_data.parquet
    │   ├── model.pth
    │   ├── scaler.joblib
    │   ├── learning_curve.png
    │   ├── run_01/           ← hedging + viz outputs
    │   │   ├── hedging_results.csv
    │   │   └── ...
    │   └── run_05/
    ├── dataset_02/ ...
    └── generalization_matrix.png
"""

import os
import gc
import sys
import time
import logging
import subprocess

import torch

# Project modules
import generate_dataset
import train_surrogate
from pricing_engines import HestonMCEngine

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Pipeline")

# ─── Configuration ──────────────────────────────────────────────────────────
TEST_MODE = False  # Full benchmark mode

N_DATASETS = 15          # 5 initial + 10 extended
N_RUNS_PER_DATASET = 5
GEN_N_SAMPLES = 100_000
GEN_BATCH_SIZE = 200
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 2048

# Override for quick smoke-tests
if TEST_MODE:
    N_DATASETS = 1
    N_RUNS_PER_DATASET = 1
    GEN_N_SAMPLES = 10_000
    TRAIN_EPOCHS = 2

BASE_DIR = "benchmark_suite"


# ─── Helpers ────────────────────────────────────────────────────────────────
def clean_memory():
    """Force garbage collection and flush GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_subprocess(cmd, description=""):
    """Run a subprocess with error handling."""
    logger.info(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"{description} complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed (exit {e.returncode})")
    except Exception as e:
        logger.error(f"{description} failed: {e}")


# ─── Main Pipeline ──────────────────────────────────────────────────────────
def run_pipeline():
    logger.info("Starting Autonomous Alpha-Hedge Pipeline (Extended Run)...")
    logger.info(f"TEST_MODE={TEST_MODE}  |  N_DATASETS={N_DATASETS}  |  "
                f"N_RUNS={N_RUNS_PER_DATASET}  |  SAMPLES={GEN_N_SAMPLES}  |  "
                f"EPOCHS={TRAIN_EPOCHS}")

    os.makedirs(BASE_DIR, exist_ok=True)
    total_start = time.time()

    for i in range(1, N_DATASETS + 1):
        ds_dir = os.path.join(BASE_DIR, f"dataset_{i:02d}")
        os.makedirs(ds_dir, exist_ok=True)

        dataset_path       = os.path.join(ds_dir, "train_data.parquet")
        current_model_path  = os.path.join(ds_dir, "model.pth")
        current_scaler_path = os.path.join(ds_dir, "scaler.joblib")

        # ── Phase 1: Dataset Generation ─────────────────────────────────
        logger.info(f"\n{'='*80}\nPhase 1: Dataset Generation "
                    f"({i}/{N_DATASETS})\n{'='*80}")

        if os.path.exists(dataset_path):
            logger.info(f"Dataset {dataset_path} already exists. Skipping Generation.")
        else:
            engine = HestonMCEngine(n_paths=20_000, n_steps=50)
            logger.info(f"Generating {dataset_path} ({GEN_N_SAMPLES} samples)...")
            generate_dataset.generate_data(engine, GEN_N_SAMPLES, dataset_path)
            del engine
            clean_memory()

        # ── Phase 2: Model Training ─────────────────────────────────────
        logger.info(f"\n{'='*80}\nPhase 2: Model Training "
                    f"({i}/{N_DATASETS})\n{'='*80}")

        if os.path.exists(current_model_path) and os.path.exists(current_scaler_path):
             logger.info(f"Model {current_model_path} exists. Skipping Training.")
        else:
            logger.info(f"Training model on {dataset_path}...")
            trained_model, trained_scaler = train_surrogate.train_model(
                data_path=dataset_path,
                model_path=current_model_path,
                scaler_path=current_scaler_path,
                epochs=TRAIN_EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
            )

            if trained_model is None:
                logger.error(f"Training failed for dataset {i}. Skipping.")
                clean_memory()
                continue
            
            # Aggressive cleanup after training
            del trained_model, trained_scaler
            clean_memory()

        logger.info(f"Model  -> {current_model_path}")
        logger.info(f"Scaler -> {current_scaler_path}")


        # ── Phase 3: Hedging Simulations ────────────────────────────────
        logger.info(f"\n{'='*80}\nPhase 3: Hedging Simulations "
                    f"({i}/{N_DATASETS})\n{'='*80}")

        for j in range(1, N_RUNS_PER_DATASET + 1):
            run_output_dir = os.path.join(ds_dir, f"run_{j:02d}")
            results_csv = os.path.join(run_output_dir, "hedging_results.csv")
            
            if os.path.exists(results_csv):
                 logger.info(f"Run {j}/{N_RUNS_PER_DATASET} exists ({results_csv}). Skipping.")
                 continue

            os.makedirs(run_output_dir, exist_ok=True)
            logger.info(f"Run {j}/{N_RUNS_PER_DATASET} -> {run_output_dir}")

            # 3a. Hedging Simulation
            cmd_hedge = [
                sys.executable, os.path.join(os.path.dirname(__file__), "hedge_simulator.py"),
                "--output_dir",  run_output_dir,
                "--model_path",  current_model_path,
                "--scaler_path", current_scaler_path,
            ]
            run_subprocess(cmd_hedge, f"Hedge Sim {j}")

            # 3b. Visualization Suite (vol_smile, greeks, error_heatmap, scatter)
            cmd_viz = [
                sys.executable, os.path.join(os.path.dirname(__file__), "visualize_alpha_hedge.py"),
                "--output_dir",  run_output_dir,
                "--model_path",  current_model_path,
                "--scaler_path", current_scaler_path,
            ]
            run_subprocess(cmd_viz, f"Visualize {j}")
            
            # Cleanup per run
            clean_memory()

        # Aggressive cleanup after simulation loop
        clean_memory()

    # ── Phase 4: Generalization Matrix (cross-world stress test) ────────
    logger.info(f"\n{'='*80}\nPhase 4: Generalization Matrix\n{'='*80}")

    cmd_gen = [
        sys.executable, os.path.join(os.path.dirname(__file__), "run_generalization_matrix.py"),
        "--output_dir", BASE_DIR,
    ]
    run_subprocess(cmd_gen, "Generalization Matrix")

    clean_memory()

    # ── Done ────────────────────────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info(f"\n{'='*80}\nPipeline Complete!\n"
                f"Total Time: {total_time:.2f}s\n{'='*80}")


if __name__ == "__main__":
    run_pipeline()
