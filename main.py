import torch
from rl4co.models import AttentionModelPolicy, AutoregressivePolicy

import wandb

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

from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":

    # wandb.login(key="55f9a8ce70d0e929d10a9f52c2ff146e8dbd7911")

    # Split Delivery Vehicle Routing Problem
    env = MappingEnv(num_procs=8)
    
    n2v_init = MappingInitEmbedding(embedding_dim=128, linear_bias=True)

    context = MappingContextEmbedding(step_context_dim=136, embedding_dim=128)
    dynamic = MappingDynamicEmbedding(embedding_dim=128, num_nodes=2)

    policy = AutoregressivePolicy(
        env.name,
        init_embedding=n2v_init,
        context_embedding=context,
        dynamic_embedding=dynamic,
    )

    model = AttentionModel(
        env,
        policy=policy,
        train_data_size=1000,
        val_data_size=100,
        optimizer_kwargs={"lr": 1e-4},
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    td_init = env.reset(batch_size=[2]).to(device)
    model = model.to(device)
    out = model(
        td_init.clone(), phase="test", decode_type="greedy", return_actions=True
    )

    print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
    for td, actions in zip(td_init, out["actions"].cpu()):
        env.render(td, actions)

    # Training

    checkpoint = ModelCheckpoint(
        dirpath="./checkpoints_atsp",
        filename="./epoch_{epoch:03d}",
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )

    summary = RichModelSummary(max_depth=3)

    callbacks = [checkpoint, summary]
    logger = WandbLogger(
        project="rl4co", name="atsp"
    )  # TensorBoardLogger('tb_logs', name='atsp')

    trainer = RL4COTrainer(
        max_epochs=2, accelerator="auto", callbacks=callbacks, devices=1, logger=None
    )

    trainer.fit(model)
