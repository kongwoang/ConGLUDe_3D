"""Hydra entrypoint for retraining ConGLUDe.

Mirrors `eval.py`: instantiates the datamodule, model, callbacks, logger and
trainer from Hydra configs, then calls `trainer.fit`. The mixed SB/LB sampling
is fully handled by `conglude.datamodule.MixedDataset` when the train datasets
list contains both `SB_train` and `LB_train`.
"""

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=False))

    torch.set_float32_matmul_precision(cfg.precision)

    datamodule = instantiate(cfg.datamodule, _recursive_=True)
    model = instantiate(cfg.model)

    callbacks = []
    if cfg.get("callbacks") is not None:
        for _, cb_cfg in cfg.callbacks.items():
            if cb_cfg is None:
                continue
            callbacks.append(instantiate(cb_cfg))

    logger = None
    if cfg.get("logger") is not None:
        logger_partial = instantiate(cfg.logger)
        logger = logger_partial(name=cfg.get("run_name", None))

    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    ckpt_path = cfg.get("resume_from_checkpoint", None)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if isinstance(logger, WandbLogger):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train()
