from hydra.utils import instantiate
from omegaconf import DictConfig
import hydra
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch


@hydra.main(config_path="configs", config_name="eval", version_base="1.2")
def eval(cfg: DictConfig):

    torch.set_float32_matmul_precision(cfg.precision)

    datamodule = instantiate(cfg.datamodule, _recursive_=True)
    model = instantiate(cfg.model)

    logger = None
    trainer = instantiate(cfg.trainer, logger=None, callbacks=None)

    # Optional: load from a Lightning .ckpt produced by `train.py`.
    # The existing per-component .pth loading remains controlled by
    # cfg.checkpoint_name (handled inside ConGLUDeModel.__init__).
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None:
        # Use weights_only=False because the checkpoint contains optimizer +
        # LR-scheduler state that PyTorch 2.6's default safe-load rejects.
        # We only consume `state_dict`, so this is safe with a trusted ckpt.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")

    trainer.test(datamodule=datamodule, model=model)

    if isinstance(logger, WandbLogger):
        wandb.finish()


if __name__ == "__main__":
    eval()
