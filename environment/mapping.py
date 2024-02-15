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
from rl4co.utils.ops import gather_by_index, get_tour_length

class MappingEnv(RL4COEnvBase):

    name = 'mapping'

    def __init__(self,
                 num_procs: int = 20,
                 min_proc: float = 0,
                 max_proc: float = 1,
                 td_params: TensorDict = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.num_procs = num_procs
        self.min_proc = min_proc
        self.max_proc = max_proc
        self._make_spec(td_params)

    def _reset(self,
               td: Optional[TensorDict] = None,
               batch_size = None) -> TensorDict:

        #init timesteps
        if td is not None:
            init_procs = td['procs']
        else:
            init_procs = None

        if batch_size is None:
            if init_procs is None:
                batch_size = self.batch_size
            else:
                batch_size = init_procs.shape[:-2]

        device = init_procs.device if init_procs is not None else self.device
        self.to(device)

        if init_procs is None:
            init_procs = self.generate_data(batch_size=batch_size).to(device)['procs']

        # Convert batch size to array
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        num_procs = init_procs.shape[-2]
        current_proc = torch.zeros((batch_size), dtype=torch.int64, device=device)
        # for masking
        available = torch.ones((*batch_size, num_procs), dtype=torch.bool, device=device)
        i_step = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        tensor_frame = TensorDict(
            {
                'procs': init_procs,
                'first_proc': current_proc,
                'current_proc': current_proc,
                'i': i_step,
                'action_mask': available,
                'reward': torch.zeros((*batch_size, 1), dtype=torch.float32)
            },
            batch_size=batch_size
        )

        return tensor_frame

    def _step(self, td: TensorDict) -> TensorDict:

        current_proc = td['action']
        if td['i'].all() == 0:
            first_proc = current_proc
        else:
            first_proc = td['first_proc']

        available = td['action_mask'].scatter(
            -1, current_proc.unsqueeze(-1).expand_as(td['action_mask']), 0
        )

        done = torch.sum(available, dim=-1) == 0
        # TODO compute reward elsewhere
        reward = torch.zeros_like(done)

        i_step = td['i'] + 1
        td.update(
            {
                'first_proc': first_proc,
                'current_proc': current_proc,
                'i': i_step,
                'action_mask': available,
                'reward': reward,
                'done': done
            },
        )

        return td

    def check_solution_validity(self, td, actions) -> TensorDict:
        # TODO Check if walk is valid here
        super().check_solution_validity(td, actions)

    def get_reward(self, td, actions) -> TensorDict:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # FIXME Compute communication cost between processes
        procs_order = gather_by_index(td['procs'], actions)
        return -get_tour_length(procs_order)

    def _make_spec(self, td_params: TensorDict = None):
        self.observation_spec = CompositeSpec(
            procs=BoundedTensorSpec(
                minimum=self.min_proc,
                maximum=self.max_proc,
                shape=(self.num_procs, 2),
                dtype=torch.float32,
            ),
            first_proc = UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            current_proc = UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            i = UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask = UnboundedDiscreteTensorSpec(
                shape=(self.num_procs),
                dtype=torch.bool
            ),
            shape=(),
        )

        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            minimum=0,
            maximum=self.num_procs
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1),)
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)

    def generate_data(self, batch_size) -> TensorDict:
        # TODO For tests
        pass

    def render(self, td, actions=None, ax=None):
        pass