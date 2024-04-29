from typing import Optional

import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.pylogger import get_pylogger
from rl4co.data.transforms import min_max_normalize

log = get_pylogger(__name__)


class MappingEnv(RL4COEnvBase):

    name = "mpimap"

    def __init__(
        self,
        num_procs: int = 32,
        min_cost: float = 0,
        max_cost: float = 1024,
        num_machines: int = 4,
        max_machine_capacity=8,
        tmat_class: bool = True,
        td_params: TensorDict = None,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_procs = num_procs
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.tmat_class = tmat_class
        self.max_machine_capacity = max_machine_capacity
        self.num_machines = num_machines

        self.node_capacities = None
        self.current_placement = None
        self.sparsity = 0.5
        self.normalize = normalize

        self.current_machine = None

        self._make_spec(td_params)

        if self.device != "cuda" or self.device != "mps":
            log.warn(f"Not using GPU for environment. Using {self.device} instead")

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize distance matrix
        cost_matrix = td["cost_matrix"] if td is not None else None

        if batch_size is None:
            batch_size = (
                # We take the first dim size from cost_matrix
                self.batch_size
                if cost_matrix is None
                else cost_matrix.shape[:-2]
            )
        device = cost_matrix.device if cost_matrix is not None else self.device
        self.to(device)

        generated_data = self.generate_data(batch_size=batch_size).to(device)

        if cost_matrix is None:
            cost_matrix = generated_data["cost_matrix"]

        cost_matrix.to(self.device)
        # Other variables
        current_node = torch.zeros(*batch_size, dtype=torch.int64, device=device).to(
            self.device
        )
        available = torch.ones(size=(*batch_size, self.num_procs), dtype=torch.bool).to(
            self.device
        )
        # current_machine = torch.zeros(*batch_size, dtype=torch.int32, device=device).to(
        #     self.device
        # )

        # The timestep
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device).to(
            self.device
        )

        # Generate node capacities tensor
        self.node_capacities = node_capacities = torch.full(
            size=(*batch_size, self.num_machines), fill_value=self.max_machine_capacity
        ).to(self.device)

        # Generate current placement tensor
        self.current_placement = current_placement = torch.full(
            size=(*batch_size, self.num_procs), fill_value=-1
        ).to(self.device)

        return TensorDict(
            {
                "cost_matrix": cost_matrix,
                "first_node": current_node,
                "current_node": current_node,
                #"current_machine": current_machine,
                "i": i,
                "action_mask": available,
                "node_capacities": node_capacities,
                "current_placement": current_placement,
            },
            batch_size=batch_size,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = action = td["action"]
        first_node = current_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]

        # Obtain where can we place this process
        machine_index = self.proc_to_machine(td).long().to(self.device)
        batch_index = torch.arange(machine_index.size(0)).long().to(self.device)

        self.node_capacities = td["node_capacities"]
        self.current_placement = td["current_placement"]

        # print(
        #     batch_index.shape,
        #     action.shape,
        #     machine_index.shape,
        #     self.current_placement.shape,
        # )
        # Decrement machine capacity
        self.node_capacities[batch_index, machine_index] -= 1
        
        # Assign process to machine
        self.current_placement[batch_index, action] = machine_index

        available = self.get_action_mask(td).to(self.device)
        done = self.get_done_state(self.node_capacities, self.current_placement).to(
            self.device
        )

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done).to(self.device)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
                "node_capacities": self.node_capacities,
                "current_placement": self.current_placement,
            },
        )

        return td

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        action: torch.Tensor = td["action"]
        available = td["action_mask"].scatter(
            -1, action.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )
        return available

    def get_done_state(
        self,
        node_capacities: torch.Tensor,
        current_placement: torch.Tensor,
    ) -> torch.Tensor:
        all_processes_placed = (current_placement != -1).all(dim=1)
        all_nodes_full = (node_capacities == 0).all(dim=1)

        done = torch.logical_or(all_processes_placed, all_nodes_full)

        return done

    def _make_spec(self, td_params: TensorDict = None):
        self.observation_spec = CompositeSpec(
            cost_matrix=BoundedTensorSpec(
                minimum=self.min_cost,
                maximum=self.max_cost,
                shape=(self.num_procs, self.num_procs),
                dtype=torch.float32,
            ),
            first_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.num_procs),
                dtype=torch.bool,
            ),
            node_capacities=BoundedTensorSpec(
                minimum=0,
                maximum=self.max_machine_capacity,
                shape=(self.num_machines),
                dtype=torch.int32,
            ),
            current_placement=BoundedTensorSpec(
                low=-1,
                high=self.num_machines,
                shape=(self.num_procs),
                dtype=torch.int32,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_procs,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def proc_to_machine(self, td: torch.Tensor) -> torch.Tensor:
        # These are the beauties of midnight programming
        capacities = td["node_capacities"]
        # print(capacities)
        # Adapt tensor considering the capacities tensor
        # steps = td["i"]
        # print(steps)

        #cumulative_caps = torch.cumsum(capacities, dim=1)

        # # Check for the first machine that can acommodate a process
        # # cumsum > steps if steps == 0, cumsum >= steps if steps > 0
        # placements = cumulative_caps >= steps
        # print(placements)
        placements_mask = capacities >= 1  # torch.logical_and(placements, capacities)
        # print(placements_mask)

        # Retrieve the first True regarding the previous condition
        machine_index = placements_mask.long().argmax(dim=1)
        # print(machine_index)

        return machine_index

    def get_reward(self, td, actions) -> TensorDict:

        cost_matrix = td["cost_matrix"].clone()
        current_placement = td["current_placement"].clone()

        reward = self.reward_from_placement(cost_matrix, current_placement)

        return reward.type(torch.float)
    
    @staticmethod
    def reward_from_placement(cost_matrix: torch.Tensor, current_placement:torch.Tensor, ):
        
        same_machine_mask = current_placement.unsqueeze(2) == current_placement.unsqueeze(1)
        masked_costs = torch.where(same_machine_mask,
            torch.zeros_like(cost_matrix),
            cost_matrix)

        
        batch_size = cost_matrix.size(0)

        worst = torch.sum(cost_matrix, dim=(1,2), dtype=torch.float).view(batch_size)

        #prevent inf's
        epsilon = 1e-9
        masked_costs_sum = torch.sum(masked_costs, dim=(1,2), dtype=torch.float)
        masked_costs_sum+=epsilon
        worst+=epsilon
        reward = -masked_costs_sum/worst


        return reward.type(torch.float)

    @staticmethod
    def actions2placement(actions: torch.Tensor, num_machines: int, ):
        #batched_actions_tensor = torch.tensor(actions) if not isinstance(actions, torch.Tensor) else actions
    
        batch_size, num_processes = actions.shape
        procs_per_machine = (num_processes + num_machines - 1) // num_machines
        placement = torch.zeros_like(actions, dtype=torch.int)
        
        for batch_idx in range(batch_size):
            actions_batch = actions[batch_idx]

            for i in range(num_processes):
                # Index of process (actions order)
                process_index = actions_batch.tolist().index(i)
                # Get machine from action index
                machine_id = process_index // procs_per_machine
                placement[batch_idx, i] = machine_id

        return placement

    def generate_data(self, batch_size, sparsity: float = None) -> TensorDict:
        
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        dms: torch.Tensor = torch.randint(
            low=0,
            high=self.max_cost,
            size=(*batch_size, self.num_procs, self.num_procs),
            generator=self.rng,
        )

        if sparsity is None:
            sparsity = torch.rand(batch_size[0], 1 ,1)

        # Processes do not communicate with themselves
        dms[..., torch.arange(self.num_procs), torch.arange(self.num_procs)] = 0
        # We create a mask to apply the sparsity factor (we will have sparsity% zeroed items)
        mask = (
            torch.rand(
                (*batch_size, self.num_procs, self.num_procs), generator=self.rng
            )
            < sparsity
        )
        # apply sparsity mask
        dms.masked_fill_(mask, 0)

        if self.normalize:
            dms = min_max_normalize(dms)

        return TensorDict(
            {
                "cost_matrix": dms,
            },
            batch_size=batch_size,
        )
    
    def get_num_starts(self, td:TensorDict):
        return td.batch_size[0]

    @staticmethod
    def render(td, actions=None, ax=None):

        td = td.detach().cpu()
        if actions is None:
            actions = td.get("action", None)

        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0]

        num_machines = td['node_capacities'].size(-1)
        num_machines  = actions.size(-1)//num_machines

        src_nodes = actions
        tgt_nodes = torch.roll(actions, 1, dims=0)

        log.info(td['cost_matrix'].cpu())

        G = nx.DiGraph(td["cost_matrix"].numpy())
        

        edge_labels = nx.get_edge_attributes(G, 'weight')
        for k, v in edge_labels.items():
                edge_labels[k] = float(str(f"{v:.2f}"))
        negated_edge_labels = {k: -v for k, v in edge_labels.items()}
        inverse_edge_labels = {k: 1/v if v != 0 else 0 for k, v in edge_labels.items()}
        inverse_edge_labels = [(u, v, w) for (u, v), w in inverse_edge_labels.items()]
        #G.add_weighted_edges_from(ebunch_to_add=negated_edge_labels, str="negated")
        G.add_weighted_edges_from(ebunch_to_add=inverse_edge_labels, str="inverse")

        pos = nx.spring_layout(G, weight="inverse")

        nodes_array = src_nodes.numpy()
        colors = plt.cm.tab10(np.linspace(0, 1, len(nodes_array)//num_machines))
        node_color_dict = {}
        node_color_dict = {}
        for i in range(0, len(nodes_array), num_machines):
            color = colors[i // num_machines]
            for j in range(num_machines):
                if i + j < len(nodes_array):
                    node_color_dict[nodes_array[i + j]] = color
            

        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_color=[node_color_dict[node] for node in G.nodes()],
            node_size=200,
            edge_color="black",
        )

        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, alpha=1)
