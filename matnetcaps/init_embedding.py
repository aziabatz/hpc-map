import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict


# TODO MOVE TO rl4co.models.nn.env_embeddings

class MatNetCapsInitEmbedding(nn.Module):
    """
    Preparing the initial row and column embeddings for FFSP.

    Reference:
    https://github.com/yd-kwon/MatNet/blob/782698b60979effe2e7b61283cca155b7cdb727f/ATSP/ATSP_MatNet/ATSPModel.py#L51


    """

    def __init__(self, embedding_dim: int, mode: str = "RandomOneHot") -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        assert mode in {
            "RandomOneHot",
            "Random",
        }, "mode must be one of ['RandomOneHot', 'Random']"
        self.mode = mode

    def forward(self, td: TensorDict):
        dmat = td["cost_matrix"]

        # shape: (batch, caps)
        caps: torch.Tensor = td["node_capacities"]
        # machines
        m = caps.size(-1)

        # batches, rows, columns
        b, r, c = dmat.shape

        # zero vector (Embedding A)
        row_emb = torch.zeros(b, r, self.embedding_dim, device=dmat.device)

        if self.mode == "RandomOneHot":
            # MatNet uses one-hot encoding for column embeddings
            # https://github.com/yd-kwon/MatNet/blob/782698b60979effe2e7b61283cca155b7cdb727f/ATSP/ATSP_MatNet/ATSPModel.py#L60
            
            col_emb = torch.zeros(b, c, self.embedding_dim, device=dmat.device)
            rand = torch.rand(b, c)
            rand_idx = rand.argsort(dim=1)
            b_idx = torch.arange(b)[:, None].expand(b, c)
            n_idx = torch.arange(c)[None, :].expand(b, c)
            # One-hot vector (Embedding B)
            col_emb[b_idx, n_idx, rand_idx] = 1.0

        elif self.mode == "Random":
            col_emb = torch.rand(b, r, self.embedding_dim, device=dmat.device)
        else:
            raise NotImplementedError
        
        # FIXME Capacities is an extended embedding for now
        caps_embedding = nn.Linear(m, r, True)
        caps_float: torch.Tensor = caps.clone()
        caps_float = caps_float.float()
        caps = caps_embedding(caps_float)
        #caps = torch.rand(b, m, self.embedding_dim, device=dmat.device) #caps_embedding(caps)

        return row_emb, col_emb, dmat, caps

