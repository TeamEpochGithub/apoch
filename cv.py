"""The main script for Cross Validation. Takes in the raw data, does CV and logs the results."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import numpy.typing as npt
import randomname
import wandb
from epochalyst.logging.section_separator import print_section_separator
from epochalyst.pipeline.ensemble import EnsemblePipeline
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config.cross_validation_config import CVConfig
from src.logging_utils.logger import logger
from src.scoring.kldiv import KLDiv
from src.typing.typing import XData
from src.utils.script.lock import Lock
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import load_training_data, setup_config, setup_data, setup_pipeline, setup_wandb

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use CVConfig instead of DictConfig
    """Do cv on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
    # Run the cv config with a dask client, and optionally a lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock():
        run_cv_cfg(cfg)


def run_cv_cfg(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q3 Detect Harmful Brain Activity - CV")
    X: XData | None
    import coloredlogs

    coloredlogs.install()

    # Set seed
    set_torch_seed()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Set up Weights & Biases group name
    wandb_group_name = randomname.get_name()
    if cfg.wandb.enabled:
        setup_wandb(cfg, "cv", output_dir, name=wandb_group_name, group=wandb_group_name)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg)

    processed_data_path = Path(cfg.processed_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path}",
    }

    # Read the data if required and split in X, y
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

    scorer = instantiate(cfg.scorer)

    scores: list[float] = []
    accuracies: list[float] = []
    f1s: list[float] = []

    for fold_no, (train_indices, test_indices) in enumerate(instantiate(cfg.splitter).split(splitter_data, y)):
        score, accuracy, f1 = run_fold(fold_no, X, y, train_indices, test_indices, cfg, scorer, output_dir, cache_args)
        scores.append(score)
        accuracies.append(accuracy)
        f1s.append(f1)
        if score > 0.85:
            break

    avg_score = np.average(np.array(scores))
    avg_accuracy = np.average(np.array(accuracies))
    avg_f1 = np.average(np.array(f1s))

    print_section_separator("CV - Results")
    logger.info(f"Average Accuracy: {avg_accuracy}")
    logger.info(f"Average F1: {avg_f1}")
    logger.info(f"Score: {avg_score}")
    wandb.log({"Score": avg_score, "Accuracy": avg_accuracy, "F1": avg_f1})
    logger.info("Finishing wandb run")
    wandb.finish()


def run_fold(
    fold_no: int,
    X: XData,
    y: npt.NDArray[np.float32],
    train_indices: np.ndarray[Any, Any],
    test_indices: np.ndarray[Any, Any],
    cfg: DictConfig,
    scorer: KLDiv,
    output_dir: Path,
    cache_args: dict[str, Any],
) -> tuple[float, float, float]:
    """Run a single fold of the cross validation.

    :param i: The fold number.
    :param X: The input data.
    :param y: The labels.
    :param train_indices: The indices of the training data.
    :param test_indices: The indices of the test data.
    :param cfg: The config file.
    :param scorer: The scorer to use.
    :param output_dir: The output directory for the prediction plots.
    :param processed_y: The processed labels.
    :return: The score of the fold.
    """
    # Print section separator
    print_section_separator(f"CV - Fold {fold_no}")
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    logger.info("Creating clean pipeline for this fold")
    model_pipeline = setup_pipeline(cfg)

    train_args = {
        "x_sys": {
            "cache_args": cache_args,
        },
        "train_sys": {
            "MainTrainer": {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "save_model": cfg.save_folds,
                "fold": fold_no,
            },
            # "cache_args": cache_args, # TODO(Jasper): Allow for caching after training in fold
        },
    }
    if isinstance(model_pipeline, EnsemblePipeline):
        train_args = {
            "ModelPipeline": train_args,
        }
    predictions, _ = model_pipeline.train(X, y, **train_args)

    if predictions is None or isinstance(predictions, XData):
        raise ValueError("Predictions are not in correct format to get a score")

    # Make sure the predictions is the same length as the test indices
    if len(predictions) != len(test_indices):
        raise ValueError("Predictions and test indices are not the same length")

    score = scorer(y[test_indices], predictions, metadata=X.meta.iloc[test_indices, :])

    # Add fold_no to fold path using os.path.join
    output_dir = output_dir / str(fold_no)
    accuracy, f1 = scorer.visualize_preds(y[test_indices], predictions, output_folder=output_dir)
    logger.info(f"Score, fold {fold_no}: {score}")
    logger.info(f"Accuracy, fold {fold_no}: {accuracy}")
    logger.info(f"F1, fold {fold_no}: {f1}")

    wandb.log({f"Score_{fold_no}": score, f"Accuracy_{fold_no}": accuracy, f"F1_{fold_no}": f1})
    return score, accuracy, f1


if __name__ == "__main__":
    run_cv()
