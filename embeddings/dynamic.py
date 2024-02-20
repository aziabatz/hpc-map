import torch.nn as nn

from rl4co.utils.pylogger import get_pylogger
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding

class MappingDynamicEmbedding(nn.Module):
    
    def __init__(self, embedding_dim, num_nodes, linear_bias=False):
        super(MappingDynamicEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.projection = nn.Linear(num_nodes, 3 * embedding_dim, bias=linear_bias)
        
        
    def forward(self, td):
        # Add singleton dim (at the end) to each node_capacities array (ie. 3,5 to 3,5,1)
        capacities = td['node_capacities'][..., None].clone()
        glimpse = self.projection(capacities)
        
        glimpse_key, glimpse_value, logit_key = glimpse.chunk(3, dim=-1)
        return glimpse_key, glimpse_value, logit_key