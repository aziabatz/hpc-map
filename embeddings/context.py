import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext


class MappingContextEmbedding(EnvContext):

    def __init__(self, embedding_dim, num_procs, step_context_dim=None, linear_bias=False):
        super(MappingContextEmbedding, self).__init__(
            embedding_dim=embedding_dim,
            step_context_dim=step_context_dim,
            # step_context_dim=embedding_dim+1
        )

        self.placement_embedding = nn.Linear(num_procs, embedding_dim)

    """
        TODO: Here we can add from Node2Vec the embedded node vectors for the first and current node.
        TODO: If dynamic embedding doesnt help, add node capacities as context for the problem partial solution
        TODO: Also we can add the placement array as part of the context embedding
    """

    def _state_embedding(self, embeddings: torch.Tensor, td):
        # current_node = td["current_node"]  # batch
        # batch_size = current_node.size(0)

        # node = torch.gather(embeddings, 0, current_node)
        # current_node_embeddings = embeddings[torch.arange(batch_size), current_node]
        current_placement = td["current_placement"].float()
        #node_capacities = td["node_capacities"]

        #state = torch.cat((current_placement, node_capacities), dim=1)

        return self.placement_embedding(current_placement)
