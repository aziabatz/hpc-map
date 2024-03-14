import warnings
import os, sys

import matplotlib.pyplot as plt
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.env import MappingEnv

from rl4co.models.nn.utils import random_policy, rollout

plt.switch_backend("Agg")
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")


@pytest.mark.parametrize("env_class", [MappingEnv])
def test_name(env_class):
    env = env_class()
    assert env.name == "mpimap", "Environment name mismatch"


@pytest.mark.parametrize("env_class", [MappingEnv])
def test_init(env_class):
    env = env_class(
        num_procs=4, num_machines=2, min_cost=0, max_cost=234, max_machine_capacity=80
    )

    assert env.num_procs == 4
    assert env.num_machines == 2
    assert env.min_cost == 0
    assert env.max_cost == 234
    assert env.max_machine_capacity == 80
    assert env.node_capacities is None
    assert env.current_placement is None


@pytest.mark.parametrize(
    "num_procs,num_machines,max_machine_capacity, bs",
    [(4, 2, 8, 7), (10, 5, 5, 6), (1, 1, 1, 1), (4, 6, 2, 3)],
)
def test_reset(num_procs, num_machines, max_machine_capacity, bs):
    env = MappingEnv(
        num_procs=num_procs,
        num_machines=num_machines,
        max_machine_capacity=max_machine_capacity,
    )
    td = env.reset(batch_size=[bs])

    expected_keys = {
        "cost_matrix",
        "first_node",
        "current_node",
        "current_machine",
        "i",
        "action_mask",
        "node_capacities",
        "current_placement",
    }

    assert expected_keys.issubset(set(td.keys())), "TensorDict from reset is incomplete"

    ####### SHAPES #######

    assert td["cost_matrix"].shape == (
        bs,
        env.num_procs,
        env.num_procs,
    ), f"Shape of 'cost_matrix' should be {(bs, env.num_procs, env.num_procs)}"
    assert td["first_node"].shape == (bs,), "Shape of 'first_node' should be (bs, 1,)"
    assert td["current_node"].shape == (
        bs,
    ), "Shape of 'current_node' should be (bs, 1,)"
    assert td["current_machine"].shape == (
        bs,
    ), "Shape of 'current_machine' should be (bs, 1,)"
    assert td["i"].shape == (bs, 1), "Shape of 'i' should be (bs, 1,)"
    assert td["action_mask"].shape == (
        bs,
        env.num_procs,
    ), f"Shape of 'action_mask' should be {(bs, env.num_procs,)}"
    assert td["node_capacities"].shape == (
        bs,
        env.num_machines,
    ), f"Shape of 'node_capacities' should be {(bs, env.num_machines,)}"
    assert td["current_placement"].shape == (
        bs,
        env.num_procs,
    ), f"Shape of 'current_placement' should be {(bs, env.num_procs,)}"

    ######################

    ###### TYPES #########
    assert td["first_node"].dtype == torch.int64, "Incorrect data type for first_node"

    assert (
        td["current_node"].dtype == torch.int64
    ), "Incorrect data type for current_node"

    assert (
        td["current_machine"].dtype == torch.int32
    ), "Incorrect data type for current_machine"

    assert td["i"].dtype == torch.int64, "Incorrect data type for i"

    assert td["action_mask"].dtype == torch.bool, "Incorrect data type for action_mask"

    assert (
        td["node_capacities"].shape[-1] == env.num_machines
    ), "Incorrect shape for node_capacities"

    assert (
        td["current_placement"].shape[-1] == env.num_procs
    ), "Incorrect shape for current_placement"

    ######################

    ######### DEVICE ######
    devices = {tensor.device for tensor in td.values()}
    expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert len(devices) == 1, "Not all tensors are on the same device"
    assert all(device == expected_device for device in devices)
    ######################

    #### VALUES ##########

    assert (
        td["cost_matrix"].ge(env.min_cost).all()
        and td["cost_matrix"].le(env.max_cost).all()
    ), "Values in 'cost_matrix' are outside the costs range"
    assert (td["first_node"] == 0).all(), "'first_node' contains invalid values"
    assert (td["current_node"] == 0).all(), "'current_node' contains invalid values"
    assert (
        td["current_node"] == td["first_node"]
    ).all(), "'current_node' should be first_node"
    assert (td["i"] == 0).all(), "'i' should be initialized to 0"

    assert td["action_mask"].all(), "'action_mask' should initially allow all actions"
    assert (td["node_capacities"] >= 0).all() and (
        td["node_capacities"] <= env.max_machine_capacity
    ).all(), "'node_capacities' contains values outside the valid range"
    assert (
        td["current_placement"] == -1
    ).all(), "'current_placement' should initially indicate no process has been placed"


@pytest.mark.parametrize(
    "num_procs,num_machines,max_machine_capacity, bs",
    [(4, 2, 8, 7), (10, 5, 5, 6), (9, 2, 3, 1), (4, 6, 2, 3)],
)
def test_step(num_procs, num_machines, max_machine_capacity, bs):
    env = MappingEnv(
        num_procs=num_procs,
        num_machines=num_machines,
        max_machine_capacity=max_machine_capacity,
    )

    td_init = env.reset(batch_size=[bs])
    td = td_init.clone()

    random_policy(td)
    env.step(td)

    assert not torch.equal(
        td["current_node"], td_init["current_node"]
    ), "The environment's state should change after a step"
