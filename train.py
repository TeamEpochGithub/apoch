"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import wandb
from epochalyst.logging.section_separator import print_section_separator
from epochalyst.pipeline.ensemble import EnsemblePipeline
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.config.train_config import TrainConfig
from src.logging_utils.logger import logger
from src.utils.script.lock import Lock
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import load_training_data, setup_config, setup_data, setup_pipeline, setup_wandb

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split. Entry point for Hydra which loads the config file."""
    # Run the train config with a dask client, and optionally a lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock():
        run_train_cfg(cfg)


def run_train_cfg(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split."""
    print_section_separator("Q3 Detect Harmful Brain Activity - Training")
    set_torch_seed()

    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.wandb.enabled:
        setup_wandb(cfg, "train", output_dir)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg)

    # Cache arguments for x_sys
    processed_data_path = Path(cfg.processed_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path}",
    }

    # Read the data if required and split it in X, y
    x_cache_exists = model_pipeline.get_x_cache_exists(cache_args)
    y_cache_exists = model_pipeline.get_y_cache_exists(cache_args)

    X, y = load_training_data(
        metadata_path=cfg.metadata_path,
        eeg_path=cfg.eeg_path,
        spectrogram_path=cfg.spectrogram_path,
        cache_path=cfg.cache_path,
        x_cache_exists=x_cache_exists,
        y_cache_exists=y_cache_exists,
    )

    if y is None:
        raise ValueError("No labels loaded to train with")

    # If cache exists, need to read the meta data for the splitter
    if X is not None:
        splitter_data = X.meta
    else:
        X, _ = setup_data(cfg.metadata_path, None, None)
        splitter_data = X.meta

    # Split indices into train and test
    indices = np.arange(len(y))
    if cfg.test_size == 0:
        train_indices, test_indices = list(indices), []  # type: ignore[var-annotated]
    elif cfg.splitter is not None:
        logger.info("Using stratified splitter to split data into train and test sets.")
        train_indices, test_indices = instantiate(cfg.splitter).split(splitter_data, y)[0]
    else:
        logger.info("Using train_test_split to split data into train and test sets.")
        train_indices, test_indices = train_test_split(indices, test_size=cfg.test_size, random_state=42)

    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    print_section_separator("Train model pipeline")
    train_args = {
        "x_sys": {
            "cache_args": cache_args,
        },
        "train_sys": {
            "MainTrainer": {
                "train_indices": train_indices,
                "test_indices": test_indices,
            },
            "cache_args": cache_args,
        },
    }
    if isinstance(model_pipeline, EnsemblePipeline):
        train_args = {
            "ModelPipeline": train_args,
        }
    predictions, _ = model_pipeline.train(X, y, **train_args)

    if len(test_indices) > 0:
        print_section_separator("Scoring")
        scorer = instantiate(cfg.scorer)
        score = scorer(y[test_indices], predictions, metadata=X.meta.iloc[test_indices, :])
        accuracy, f1 = scorer.visualize_preds(y[test_indices], predictions, output_folder=output_dir)
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1: {f1}")
        logger.info(f"Score: {score}")

        if wandb.run:
            wandb.log({"Accuracy": accuracy, "F1": f1, "Score": score})

    wandb.finish()


if __name__ == "__main__":
    run_train()
