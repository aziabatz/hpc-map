from typing import List, Optional, Tuple
from lightning import Callback, LightningModule
from omegaconf import DictConfig
import torch
from rl4co.models import AttentionModelPolicy, AutoregressivePolicy
from rl4co.models import MDAMPolicy

import wandb
import hydra
import numpy as np

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
from rl4co.models.zoo.deepaco.policy import DeepACOPolicy
from rl4co.models.nn.env_embeddings.edge import ATSPEdgeEmbedding

from lightning.pytorch.loggers import WandbLogger
from utils.hydra import instantiate_callbacks, instantiate_loggers
from utils.platform import get_device, get_accelerator
from utils.logger import get_pylogger
from utils.plot import plot
from eval_model import eval_mapping

from rl4co.models.zoo.matnet import  MatNet
from rl4co.models.zoo.matnetcaps import MatNetCapsPolicy

MAX_PROCESS = 8
MAX_NODES = 4
EMBEDDING_SIZE = 128
BATCH_SIZE = 4


log = get_pylogger(__name__)


def run(cfg: DictConfig) -> Tuple[dict, dict]:

    device = get_device()
    accelerator = get_accelerator(device)
    print(f"Using platform {device} with accelerator {accelerator}")

    #cfg.model.train_data_size = cfg.model.batch_size * 100
    #cfg.model.val_data_size = cfg.model.val_batch_size * 100

    #wandb.login(key="55f9a8ce70d0e929d10a9f52c2ff146e8dbd7911")

    env: MappingEnv = hydra.utils.instantiate(cfg.env, device=device)
    
    # TODO Get policy from hydra


    policy = MatNetCapsPolicy(
        "atsp",
        #embedding_dim=EMBEDDING_SIZE,
        num_encoder_layers=20,
        #init_embedding_kwargs={"mode": "Random"}
    )

    log.info(f"Init model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, env, policy=policy,)

    ######################### GREEDY UNTRAINED MODEL ######################

    td_init0 = env.reset(batch_size=[4]).to(device)
    td_init = td_init0.clone()
    model = model.to(device)
    out = model(td_init, phase="test", decode_type="greedy", return_actions=True)
    untrained = out['actions'].cpu().detach()
    rew_untrained = out['reward'].cpu().detach()

    #######################################################################

    log.info("Init callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Init loggers...")
    logger: List[Callback] = instantiate_loggers(cfg.get("logger"))

    log.info("Init trainer...")
    trainer: RL4COTrainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        devices=1,
        accelerator=accelerator,
        #auto_lr_find=True,
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
    

    model = model.to(device)

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        last_path = trainer.checkpoint_callback.last_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        log.info(f"Best ckpt path: {ckpt_path}")
        log.info("Testing with best model...")
        trainer.test(model=model, ckpt_path=ckpt_path, verbose=True)
        log.info(f"Last ckpt path: {last_path}")
        log.info("Testing with last model...")
        trainer.test(model=model, ckpt_path=last_path, verbose=True)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    model = model.to(device)
    td_init = td_init0.clone() #env.reset(batch_size=[])
    out = model(td_init, phase="test", decode_type="greedy", return_actions=True)
    trained = out['actions'].cpu().detach()
    rew_trained = out['reward'].cpu().detach()
    #plot(td_init, env, trained, rew_trained, untrained, rew_untrained)

    optimals = np.load(env.test_file)['optimals']
    cost_matrix = np.load(env.test_file)['cost_matrix']

    optimals = torch.from_numpy(optimals)
    cost_matrix = torch.from_numpy(cost_matrix)

    log.info(f"Evaluation with best model in {ckpt_path}")
    eval_mapping(model, ckpt_path, env, optimals, cost_matrix)

    log.info(f"Evaluation with last model in {last_path}")
    eval_mapping(model, last_path, env, optimals, cost_matrix)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs/", config_name="main")
def train(cfg: DictConfig) -> Optional[float]:
    metric_dict, object_dict = run(cfg)
    return None


if __name__ == "__main__":
    train()
