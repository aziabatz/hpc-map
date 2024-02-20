import torch
import torch.nn as nn

from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext


class MappingContextEmbedding(EnvContext):

    def __init__(self, embedding_dim, step_context_dim=None, linear_bias=False):
        super(MappingContextEmbedding, self).__init__()

