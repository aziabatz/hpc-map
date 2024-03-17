import torch.nn as nn

from rl4co.utils.pylogger import get_pylogger
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding


class MappingDynamicEmbedding(nn.Module):

    def __init__(self, num_nodes, num_procs, embedding_dim, linear_bias=False):
        super(MappingDynamicEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.num_procs = num_procs
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(
            num_nodes, 3 * (num_procs * embedding_dim), bias=linear_bias
        )  # nn.Linear(1, 3 * embedding_dim, bias=linear_bias)

    def forward(self, td):
        # Add singleton dim (at the end) to each node_capacities array (ie. 3,5 to 3,5,1)
        capacities = td["node_capacities"].clone().float()
        # capacities = td["node_capacities"].clone().float()
        batch_size = capacities.shape[0]
        glimpse = self.projection(capacities)
        glimpse = glimpse.view(-1, batch_size, self.num_procs, self.embedding_dim)

        glimpse_key, glimpse_value, logit_key = glimpse.chunk(3, dim=1)

        glimpse_key = glimpse_key.squeeze(1)
        glimpse_value = glimpse_value.squeeze(1)
        logit_key = logit_key.squeeze(1)

        # glimpse_key, glimpse_value, logit_key = glimpse.chunk(3, dim=-1)
        return glimpse_key, glimpse_value, logit_key
