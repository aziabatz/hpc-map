import rl4co
import torch
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.zoo.matnet import MatNetPolicy, MatNet

from environment.env import MappingEnv


def eval_mapping(model: MatNet, ckpt: str, env:MappingEnv):

    model = MatNet.load_from_checkpoint(ckpt)
    
    test_dataset = env.dataset(phase="test")
    batch = test_dataset.data_len
    dataloader = model._dataloader(test_dataset, batch)

    init_states = next(iter(dataloader))
    td_init_generalization = env.reset(init_states)

    out = model(td_init_generalization.clone(), phase="test", decode_type="greedy", return_actions=True)

    print(f"Comm. cost: {[f'{-r.item():.2f}' for r in out['reward']]}")
    for td, actions in zip(td_init_generalization, out['actions'].cpu()):
        env.render(td, actions)

    pass