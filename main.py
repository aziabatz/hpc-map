from typing import List, Optional, Tuple
from lightning import Callback, LightningModule
from omegaconf import DictConfig
import torch
from rl4co.models import AttentionModelPolicy, AutoregressivePolicy

import wandb
import hydra

# Para ir guardando el mejor modelo hasta ahora
from lightning.pytorch.callbacks import ModelCheckpoint

# Para mostrar un log de la estructura de la red
from lightning.pytorch.callbacks import RichModelSummary

from embeddings.init import MappingInitEmbedding
from embeddings.context import MappingContextEmbedding
from embeddings.dynamic import MappingDynamicEmbedding
from environment.env import MappingEnv

from rl4co.envs import ATSPEnv, TSPEnv
from rl4co.models.zoo.ppo.model import PPOModel
from rl4co.models.zoo import AttentionModel
from rl4co.utils.trainer import RL4COTrainer
from rl4co.models.zoo.ppo import PPOPolicy
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding

from lightning.pytorch.loggers import WandbLogger
from utils.hydra import instantiate_callbacks, instantiate_loggers
from utils.platform import get_device, get_accelerator
from utils.logger import get_pylogger

MAX_PROCESS = 8
MAX_NODES = 4
EMBEDDING_SIZE = 128
BATCH_SIZE = 4


log = get_pylogger(__name__)


def run(cfg: DictConfig) -> Tuple[dict, dict]:

    device = "cpu"  # get_device()
    accelerator = "cpu"  # get_accelerator(device)
    print(f"Using platform {device} with accelerator {accelerator}")

    wandb.login(key="55f9a8ce70d0e929d10a9f52c2ff146e8dbd7911")

    log.info(f"Init env:  <{cfg.env._target_}>")

    env = hydra.utils.instantiate(cfg.env)

    n2v_init = MappingInitEmbedding(
        embedding_dim=EMBEDDING_SIZE, linear_bias=True, device=device
    )

    context = MappingContextEmbedding(
        step_context_dim=(MAX_PROCESS + MAX_NODES + EMBEDDING_SIZE),
        embedding_dim=EMBEDDING_SIZE,
    )

    # TODO Get policy from hydra

    policy = AutoregressivePolicy(
        env.name,
        init_embedding=n2v_init,
        context_embedding=context,
        dynamic_embedding=StaticEmbedding(),
    )

    log.info(f"Init model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, env, policy=policy)

    # model = AttentionModel(
    #     env,
    #     baseline="rollout",
    #     policy=policy,
    #     batch_size=BATCH_SIZE,
    #     val_batch_size=BATCH_SIZE,
    #     test_batch_size=BATCH_SIZE,
    #     train_data_size=10_000,
    #     val_data_size=1000,
    #     test_data_size=100,
    #     optimizer_kwargs={"lr": 1e-4},
    # )

    log.info("Init callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Init loggers...")
    logger: List[Callback] = instantiate_loggers(cfg.get("logger"))

    log.info("Init trainer...")
    trainer: RL4COTrainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if cfg.get("compile", False):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path"))

        train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs/", config_name="main")
def train(cfg: DictConfig) -> Optional[float]:
    metric_dict, object_dict = run(cfg)
    return None


if __name__ == "__main__":
    train()
