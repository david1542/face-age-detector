import hydra
import logging
import torch
from typing import List

import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from torch.nn import parameter
from src.utils.logging import get_logger

# Instantiate a new multi-GPU friendly logger
log = get_logger()


def train(config: DictConfig):
    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    if config.get('debug'):
        log.setLevel(logging.DEBUG)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    # Init optimizer
    log.info(f"Instantiating optimizer <{config.optimizer._target_}>")
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
        config.optimizer, model.parameters())

    # Attach optimizer to model
    model.optimizer = optimizer

    # Init lightning callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer)

    # # Train the model
    # log.info("Starting training!")
    # trainer.fit(model=model, datamodule=datamodule)

    # # Get metric score for hyperparameter optimization
    # score = trainer.callback_metrics.get(config.get("optimized_metric"))
