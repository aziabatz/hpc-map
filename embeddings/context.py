import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext


class MappingContextEmbedding(EnvContext):

    def __init__(self, embedding_dim, step_context_dim=None, linear_bias=False):
        super(MappingContextEmbedding, self).__init__(
            embedding_dim=embedding_dim,
            step_context_dim=embedding_dim+1
        )
        
    """
        TODO: Here we can add from Node2Vec the embedded node vectors for the first and current node.
        TODO: If dynamic embedding doesnt help, add node capacities as context for the problem partial solution
        TODO: Also we can add the placement array as part of the context embedding
    """

    def _state_embedding(self, embeddings, td):
        total_placed = td['i']
        current_placement = td['current_placement']

        return torch.cat([total_placed, current_placement], -1)
