import rl4co
import torch
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.zoo.matnet import MatNetPolicy, MatNet

from environment.env import MappingEnv


def eval_mapping(model: MatNet, ckpt: str, env:MappingEnv, optimals: torch.Tensor, cost_matrix: torch.Tensor):

    model = MatNet.load_from_checkpoint(ckpt)
    device='cuda'
    
    test_dataset = env.dataset(phase="test")
    batch = test_dataset.data_len
    dataloader = model._dataloader(test_dataset, batch)

    init_states = next(iter(dataloader))
    td_init_generalization = env.reset(init_states).to(device)

    out = model(td_init_generalization.clone(), phase="test", decode_type="greedy", return_actions=True)
    num_machines = env.num_machines

    rewards = out['reward'].cpu().detach()

    print(f"Comm. cost: {[f'{-r.item():.2f}' for r in rewards]}")
    for td, actions in zip(td_init_generalization, out['actions'].cpu()):
        env.render(td, actions)
        
    optimals_rewards = env.reward_from_placement(cost_matrix, optimals)
    print("Model rewards: ", rewards, "Optimal rewards: ", optimals_rewards)
    print("Model mapping against optimal mapping (improvement 1-scale):", (optimals_rewards/rewards))
