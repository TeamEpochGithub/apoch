from typing import Any
from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline


def setup_train_args(
    pipeline: ModelPipeline | EnsemblePipeline,
    cache_args: dict[str, Any],
    train_indices: list[int],
    test_indices: list[int],
    fold: int = -1,
    *,
    save_model: bool = False,
    save_model_preds: bool = False
) -> dict[str, Any]:

    # Initial
    train_args = {
        "x_sys": {
            "cache_args": cache_args,
        },
        "train_sys": {
            "MainTrainer": {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "save_model": save_model,
            },
            # "cache_args": cache_args, # TODO(Jasper): Allow for caching after training in fold
        },
        "pred_sys": {}
    }

    if fold > -1:
        train_args["train_sys"]["MainTrainer"]["fold"] = fold

    if save_model_preds:
        train_args["train_sys"]["cache_args"] = cache_args

    if isinstance(pipeline, EnsemblePipeline):
        train_args = {
            "ModelPipeline": train_args,
        }

    raise NotImplementedError("setup_train_args is competition specific")


def setup_pred_args(pipeline: ModelPipeline | EnsemblePipeline) -> dict[str, Any]:
    raise NotImplementedError("setup_pred_args is competition specific")

    pred_args = {
        "train_sys": {
            "MainTrainer": {
                # "batch_size": 16,
                # "model_folds": cfg.model_folds,
            },
        },
    }

    if isinstance(pipeline, EnsemblePipeline):
        pred_args = {
            "ModelPipeline": pred_args,
        }

    return pred_args
