from typing import Optional

import torch

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

log = get_pylogger(__name__)


class MappingEnv(RL4COEnvBase):

    name = "mpimap"

    # TODO Add node_capacities, total_placed, current_placement

    def __init__(
        self,
        num_procs: int = 32,
        min_cost: float = 0,
        max_cost: float = 1024,
        num_machines: int = 4,
        max_machine_capacity=8,
        tmat_class: bool = True,
        td_params: TensorDict = None,
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

        self.current_machine = None

        self._make_spec(td_params)

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

        # Other variables
        current_node = torch.zeros(*batch_size, dtype=torch.int64, device=device)
        available = torch.ones(size=(*batch_size, self.num_procs), dtype=torch.bool)
        current_machine = torch.zeros(*batch_size, dtype=torch.int32, device=device)

        # The timestep
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Generate node capacities tensor
        self.node_capacities = node_capacities = torch.randint(
            low=0, high=self.max_machine_capacity, size=(*batch_size, self.num_machines)
        )

        # Generate current placement tensor
        self.current_placement = current_placement = torch.full(
            size=(*batch_size, self.num_procs), fill_value=-1
        )

        return TensorDict(
            {
                "cost_matrix": cost_matrix,
                "first_node": current_node,
                "current_node": current_node,
                "current_machine": current_machine,
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
        machine_index = self.proc_to_machine(td).long()
        batch_index = torch.arange(machine_index.size(0)).long()
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

        available = self.get_action_mask(td)
        done = self.get_done_state(self.node_capacities, self.current_placement)

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

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

        cumulative_caps = torch.cumsum(capacities, dim=1)

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

        node_diff_mask = current_placement.unsqueeze(2) != current_placement.unsqueeze(
            1
        )
        cost_matrix[node_diff_mask] = 0

        reward = -cost_matrix  # -total_communication_cost

        return reward.float()

    def generate_data(self, batch_size) -> TensorDict:
        # Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
        # We satifsy the triangle inequality (TMAT class) in a batch
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        dms = (
            torch.randint(
                low=0,
                high=self.max_cost,
                size=(*batch_size, self.num_procs, self.num_procs),
                generator=self.rng,
            )
            # * (self.max_cost - self.min_cost)
            # + self.min_cost
        )
        # TODO Add a new parameter sparsity?
        dms[..., torch.arange(self.num_procs), torch.arange(self.num_procs)] = 0
        log.debug("Using TMAT class (triangle inequality): {}".format(self.tmat_class))
        if self.tmat_class:
            while True:
                old_dms = dms.clone()
                dms, _ = (
                    dms[..., :, None, :] + dms[..., None, :, :].transpose(-2, -1)
                ).min(dim=-1)
                if (dms == old_dms).all():
                    break

        return TensorDict(
            {
                "cost_matrix": dms,
            },
            batch_size=batch_size,
        )

    @staticmethod
    def render(td, actions=None, ax=None):
        try:
            import networkx as nx
        except ImportError:
            log.warn(
                "Networkx is not installed. Please install it with `pip install networkx`"
            )
            return

        td = td.detach().cpu()
        if actions is None:
            actions = td.get("action", None)

        # if batch_size greater than 0 , we need to select the first batch element
        if td.batch_size != torch.Size([]):
            td = td[0]
            actions = actions[0]

        src_nodes = actions
        tgt_nodes = torch.roll(actions, 1, dims=0)

        # Plot with networkx
        G = nx.DiGraph(td["cost_matrix"].numpy())
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="skyblue",
            node_size=800,
            edge_color="white",
        )

        # draw edges src_nodes -> tgt_nodes
        edgelist = [
            (src_nodes[i].item(), tgt_nodes[i].item()) for i in range(len(src_nodes))
        ]
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist, width=2, alpha=1, edge_color="black"
        )
