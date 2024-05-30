import rl4co
import torch
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.zoo.matnet import MatNetPolicy, MatNet
from rl4co.utils.pylogger import get_pylogger
from environment.env import MappingEnv

log = get_pylogger(__name__)



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

    actions = out['actions'].cpu().detach()
    rewards = out['reward'].cpu().detach()

    optimal_placements = optimals
    model_placements = env.actions2placement(actions, num_machines)
    optimal_rewards = env.reward_from_placement(cost_matrix, optimal_placements)
    model_rewards = rewards

    for model_mapping, optimal_mapping, model_reward, optimal_reward in zip(model_placements, optimal_placements, model_rewards, optimal_rewards):
        score = optimal_reward/model_reward
        log.info(f"===================\nModel Mapping: {model_mapping} Optimal Mapping: {optimal_mapping}\nModel Reward: {model_reward} Optimal Reward: {optimal_reward}\nScore (optimal/model ratio): {score}\n===================")
    

    # print(f"Comm. cost: {[f'{-r.item():.2f}' for r in rewards]}")
    # for td, actions, optimal in zip(td_init_generalization, out['actions'].cpu(), optimals):
    #     placement = env.actions2placement(actions, num_machines)


