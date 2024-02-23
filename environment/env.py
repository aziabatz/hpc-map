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
        max_machine_capacity = 8,
        tmat_class: bool = True,
        td_params: TensorDict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_procs = num_procs
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.tmat_class = tmat_class
        self.max_machine_capacity= max_machine_capacity
        self.num_machines = num_machines
        
        self.node_capacities = None
        self.current_placement = None
        
        self._make_spec(td_params)
        
    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize distance matrix
        cost_matrix = (
            td["cost_matrix"] if td is not None else None
        )
        
        if batch_size is None:
            batch_size = (
                # We take the first dim size from cost_matrix
                self.batch_size if cost_matrix is None else cost_matrix.shape[:-2]
            )
        device = cost_matrix.device if cost_matrix is not None else self.device
        self.to(device)

        generated_data = self.generate_data(batch_size=batch_size).to(device)

        if cost_matrix is None:
            cost_matrix = generated_data["cost_matrix"]

        # Other variables
        current_node = torch.zeros(batch_size, dtype=torch.int64, device=device)
        available = torch.ones(size=(*batch_size, self.num_machines))
        
        # The timestep
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        # Generate node capacities tensor
        self.node_capacities = node_capacities = torch.randint(low=0, high=self.max_machine_capacity, size=(*batch_size, self.num_machines))

        # Generate current placement tensor
        self.current_placement = current_placement = torch.full(size=(*batch_size, self.num_procs), fill_value=-1)

        return TensorDict(
            {
                "cost_matrix": cost_matrix,
                "first_node": current_node,
                "current_node": current_node,
                "i": i,
                "action_mask": available,

                "node_capacities": node_capacities,
                "current_placement": current_placement,
            },
            batch_size=batch_size,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]
        first_node = current_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]


        available = self.get_action_mask(td)
        # available = td["action_mask"].scatter(
        #     -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        # )
        

        # We are done there are no unvisited locations
        done = self.get_done_state(self.node_capacities, td['i']+1)

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
        left_capacities = td['node_capacities']
        available_nodes = left_capacities > 0
        return available_nodes

    def get_done_state(self, capacities: torch.Tensor, i: int) -> torch.Tensor:
        all_processes_placed = i == self.num_procs
        all_nodes_full = torch.count_nonzero(capacities, -1) == self.num_machines

        done = all_processes_placed or all_nodes_full

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
                dtype=torch.int32
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

    def get_reward(self, td, actions) -> TensorDict:
        distance_matrix = td["cost_matrix"]
        
        #Check we only visit every node once
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Get indexes of tour edges
        nodes_src = actions
        nodes_tgt = torch.roll(actions, 1, dims=1)
        batch_idx = torch.arange(
            distance_matrix.shape[0], device=distance_matrix.device
        ).unsqueeze(1)
        # return negative tour length
        return -distance_matrix[batch_idx, nodes_src, nodes_tgt].sum(-1)

    def generate_data(self, batch_size) -> TensorDict:
        # Generate distance matrices inspired by the reference MatNet (Kwon et al., 2021)
        # We satifsy the triangle inequality (TMAT class) in a batch
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        dms = (
            torch.randint(low=0, high=self.max_cost, size=(*batch_size, self.num_procs, self.num_procs), generator=self.rng)
            # * (self.max_cost - self.min_cost)
            # + self.min_cost
        )
        dms[..., torch.arange(self.num_procs), torch.arange(self.num_procs)] = 0
        log.info("Using TMAT class (triangle inequality): {}".format(self.tmat_class))
        if self.tmat_class:
            while True:
                old_dms = dms.clone()
                dms, _ = (
                    dms[..., :, None, :] + dms[..., None, :, :].transpose(-2, -1)
                ).min(dim=-1)
                if (dms == old_dms).all():
                    break

        

        return TensorDict({
            "cost_matrix": dms,
            }, batch_size=batch_size)

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
