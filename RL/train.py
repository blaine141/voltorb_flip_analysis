import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from vanilla_policy_gradient_model import VanillaPolicyGradient
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
import voltorb_flip



if __name__ == "__main__":
    model = VanillaPolicyGradient(
        env="VoltorbFlip-v0",
        lr=1e-3,
        batch_size=64,
    )

    deterministic_checkpoint_callback = ModelCheckpoint(
        monitor="deterministic_win_rate",
        filename="deterministic-{epoch:02d}-{deterministic_win_rate:.2f}",
        save_top_k=3,
        mode="max"
    )

    non_deterministic_checkpoint_callback = ModelCheckpoint(
        monitor="non_deterministic_win_rate",
        filename="non_deterministic-{epoch:02d}-{non_deterministic_win_rate:.2f}",
        save_top_k=3,
        mode="max"
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=500000,
        callbacks=[deterministic_checkpoint_callback, non_deterministic_checkpoint_callback],
        logger=TensorBoardLogger('logs/'),
    )

    trainer.fit(model)