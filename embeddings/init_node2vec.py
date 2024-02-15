
import torch.nn as nn
from rl4co.models.nn.env_embeddings.init import TSPInitEmbedding


class MappingInitEmbedding(nn.Module):

    def __init__(self,
                 num_procs: int,
                 embedding_dim: int,
                 linear_bias = True,

                 ):
        super(MappingInitEmbedding, self).__init__()
        node_dim = num_procs

        # FIXME Replace with n2v
        self.init_embed = nn.Linear(node_dim, embedding_dim, linear_bias)

    def forward(self, td):
        out = self.init_embed(td['cost_matrix'])
        return out

