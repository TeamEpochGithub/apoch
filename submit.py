"""Submit.py is the main script for running inference on the test set and creating a submission."""
import os
import warnings
from pathlib import Path

import hydra
import pandas as pd
from epochalyst.logging.section_separator import print_section_separator
from epochalyst.pipeline.ensemble import EnsemblePipeline
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from src.config.submit_config import SubmitConfig
from src.logging_utils.logger import logger
from src.utils.setup import setup_config, setup_data, setup_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_submit", node=SubmitConfig)


@hydra.main(version_base=None, config_path="conf", config_name="submit")
# TODO(Epoch): Use SubmitConfig instead of DictConfig
def run_submit(cfg: DictConfig) -> None:
    """Run the main script for submitting the predictions."""
    print_section_separator("Q3 Detect Harmful Brain Activity - Submit")

    # Set up logging
    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg)

    # Load the test data
    eeg_path = Path(cfg.eeg_path)
    spectrogram_path = Path(cfg.spectrogram_path)
    metadata_path = Path(cfg.metadata_path)
    X, _ = setup_data(metadata_path, eeg_path, spectrogram_path, use_test_data=True)

    # Predict on the test data
    logger.info("Making predictions...")

    pred_args = {
        "train_sys": {
            "MainTrainer": {
                "batch_size": 16,
                "model_folds": cfg.model_folds,
            },
        },
    }
    if isinstance(model_pipeline, EnsemblePipeline):
        pred_args = {
            "ModelPipeline": pred_args,
        }
    predictions = model_pipeline.predict(X, **pred_args)

    # Make submission
    if predictions is not None:
        # Create a dataframe from the predictions
        label_columns = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
        submission = pd.DataFrame(predictions, columns=label_columns)

        # Add the eeg_id to the submission
        submission["eeg_id"] = X.meta["eeg_id"]

        # Reorder the columns
        submission = submission[["eeg_id", *label_columns]]

        # Save the submission
        result_path = Path(cfg.result_path)
        os.makedirs(result_path, exist_ok=True)
        submission_path = result_path / "submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    else:
        raise ValueError("Predictions are None")


if __name__ == "__main__":
    run_submit()
