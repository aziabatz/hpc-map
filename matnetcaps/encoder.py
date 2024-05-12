import math

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.ops import Normalization


class MatNetCrossMHACaps(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        bias: bool = False,
        mixer_hidden_dim: int = 16,
        mix1_init: float = (1 / 2) ** (1 / 2),
        mix2_init: float = (1 / 16) ** (1 / 2),
        mix3_init: float = (1 / 16) ** (1 / 2),

        dmat_score_factor: float = 0.5,
        caps_score_factor: float = 0.5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert (
            self.embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"
        self.head_dim = self.embedding_dim // num_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.Wkv = nn.Linear(embedding_dim, 2 * embedding_dim, bias=bias)

        # Score mixer
        # Taken from the official MatNet implementation
        # https://github.com/yd-kwon/MatNet/blob/main/ATSP/ATSP_MatNet/ATSPModel_LIB.py#L72
        mix_W1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, 2, mixer_hidden_dim)
        )
        mix_b1 = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample(
            (num_heads, mixer_hidden_dim)
        )
        self.mix_W1 = nn.Parameter(mix_W1)
        self.mix_b1 = nn.Parameter(mix_b1)

        mix_W2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, mixer_hidden_dim, 1)
        )
        mix_b2 = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample(
            (num_heads, 1)
        )
        self.mix_W2 = nn.Parameter(mix_W2)
        self.mix_b2 = nn.Parameter(mix_b2)

        mix1caps = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((num_heads, 2, mixer_hidden_dim))
        mix1bias_caps = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((num_heads, mixer_hidden_dim))
        mix2caps = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((num_heads, mixer_hidden_dim, 1))
        mix2bias_caps = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((num_heads, 1))

        mix3 = torch.torch.distributions.Uniform(low=-mix3_init, high=mix3_init).sample((2, 1))
        mix3bias = torch.torch.distributions.Uniform(low=-mix3_init, high=mix3_init).sample((num_heads,))

        self.mix1caps = nn.Parameter(mix1caps)
        self.mix1bias_caps = nn.Parameter(mix1bias_caps)
        self.mix2caps = nn.Parameter(mix2caps)
        self.mix2bias_caps = nn.Parameter(mix2bias_caps)
        self.mix3 = nn.Parameter(mix3)
        self.mix3bias = nn.Parameter(mix3bias)

        self.dmat_score_factor = dmat_score_factor
        self.caps_score_factor = caps_score_factor

        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)

    def mix_dmat(self, attn_scores, dmat, b, m, n):
        # stack dot product with dmat score
        mix_attn_scores = torch.stack(
            [attn_scores, dmat[:, None, :, :].expand(b, self.num_heads, m, n)], dim=-1
        )  # [b, h, m, n, 2]

        mix_attn_scores = (
            (
                torch.matmul(
                    F.relu(
                        torch.matmul(mix_attn_scores.transpose(1, 2), self.mix_W1)
                        + self.mix_b1[None, None, :, None, :]
                    ),
                    self.mix_W2,
                )
                + self.mix_b2[None, None, :, None, :]
            )
            .transpose(1, 2)
            .squeeze(-1)
        )  # [b, h, m, n]

        return mix_attn_scores

    def mix_caps(self, attn_scores, caps_emb, b, m, n):
        h = self.num_heads
        d = self.embedding_dim


        caps_score = caps_emb[:, None, None, :].expand(b, h, m, n)
        
        two_scores_caps = torch.stack((attn_scores, caps_score), dim=-1)
        two_scores_caps = two_scores_caps.transpose(1,2)
        # shape: (batch, caps, head_num, caps, 2)

        ms1caps = torch.matmul(two_scores_caps, self.mix1caps)
        ms1caps += self.mix1bias_caps[None, None, :, None, :]
        ms1caps = F.relu(ms1caps)
        
        ms2caps = torch.matmul(ms1caps, self.mix2caps)
        ms2caps += self.mix2bias_caps[None, None, :, None, :]

        mixed_scores_caps = ms2caps.transpose(1,2).squeeze(-1)
        #mixed_scores_caps = ms2caps.squeeze(-1)

        return mixed_scores_caps

    def forward(self, q_input, kv_input, dmat, caps_emb):
        """

        Args:
            q_input (Tensor): [b, m, d]
            kv_input (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]

        Returns:
            Tensor: [b, m, d]
        """

        b, m, n = dmat.shape

        q = rearrange(
            self.Wq(q_input), "b m (h d) -> b h m d", h=self.num_heads
        )  # [b, h, m, d]
        k, v = rearrange(
            self.Wkv(kv_input), "b n (two h d) -> two b h n d", two=2, h=self.num_heads
        ).unbind(
            dim=0
        )  # [b, h, n, d]

        scale = math.sqrt(q.size(-1))  # scale factor
        attn_scores = torch.matmul(q, k.transpose(2, 3)) / scale  # [b, h, m, n]
        
        dmat_score = self.mix_dmat(attn_scores, dmat, b, m, n)
        caps_score = self.mix_caps(attn_scores, caps_emb, b, m, n)

        #stacked_mix = dmat_score*self.dmat_score_factor + caps_score*self.caps_score_factor
        stacked_mix = torch.stack([dmat_score, caps_score], dim=-1)
        stacked_mix = F.relu(stacked_mix)

        stacked_mix = stacked_mix.transpose(1,-2)
        
        #ms3
        mixed = torch.matmul(stacked_mix, self.mix3)
        # FIXME funcionará bien como suma ponderada???
        mixed = mixed.squeeze(-1)
        mixed += self.mix3bias[None, None, None, :]

        mixed = mixed.transpose(1, -1)
        
        attn_probs = F.softmax(mixed, dim=-1)
        out = torch.matmul(attn_probs, v)
        #return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))

        #out = out.transpose(0,1)
        #out = out.transpose(2,3)
        #out_concat = out.reshape(b, self.embedding_dim, 2 * self.num_heads * self.head_dim)

        out_concat = self.out_proj(rearrange(out, "b h s d -> b s (h d)"))

        return out_concat

# ENCODERLAYER CLASS
class MatNetMHACaps(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.row_encoding_block = MatNetCrossMHACaps(embedding_dim, num_heads, bias)
        self.col_encoding_block = MatNetCrossMHACaps(embedding_dim, num_heads, bias)

    def forward(self, row_emb, col_emb, dmat, caps_emb):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]
            caps_emb (Tensor): [b, d]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
        """

        updated_row_emb = self.row_encoding_block(row_emb, col_emb, dmat, caps_emb)
        updated_col_emb = self.col_encoding_block(
            col_emb, row_emb, dmat.transpose(-2, -1), caps_emb
        )
        
        return updated_row_emb, updated_col_emb

class MatNetMHALayerCaps(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        bias: bool = False,
        feed_forward_hidden: int = 512,
        normalization: Optional[str] = "instance",
    ):
        super().__init__()
        self.MHA = MatNetMHACaps(embedding_dim, num_heads, bias)

        self.F_a = nn.ModuleDict(
            {
                "norm1": Normalization(embedding_dim, normalization),
                "ffn": nn.Sequential(
                    nn.Linear(embedding_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embedding_dim),
                ),
                "norm2": Normalization(embedding_dim, normalization),
            }
        )

        self.F_b = nn.ModuleDict(
            {
                "norm1": Normalization(embedding_dim, normalization),
                "ffn": nn.Sequential(
                    nn.Linear(embedding_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embedding_dim),
                ),
                "norm2": Normalization(embedding_dim, normalization),
            }
        )

        # self.F_c = nn.ModuleDict(
        #     {
        #         "norm1": Normalization(embedding_dim, normalization),
        #         "ffn": nn.Sequential(
        #             nn.Linear(embedding_dim, feed_forward_hidden),
        #             nn.ReLU(),
        #             nn.Linear(feed_forward_hidden, embedding_dim),
        #         ),
        #         "norm2": Normalization(embedding_dim, normalization)
        #     }
        # )

    def forward(self, row_emb, col_emb, dmat, caps_emb):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]
            caps_emb (Tensor): [b, d]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
            Updated col_emb (Tensor): [b, d]
        """

        row_emb_out, col_emb_out = self.MHA(row_emb, col_emb, dmat, caps_emb)

        row_emb_out = self.F_a["norm1"](row_emb + row_emb_out)
        row_emb_out = self.F_a["norm2"](row_emb_out + self.F_a["ffn"](row_emb_out))

        col_emb_out = self.F_b["norm1"](col_emb + col_emb_out)
        col_emb_out = self.F_b["norm2"](col_emb_out + self.F_b["ffn"](col_emb_out))

        # caps_emb_out = self.F_c["norm1"](caps_emb + caps_emb_out)
        # caps_emb_out = self.F_c["norm2"](caps_emb_out + self.F_c["ffn"](caps_emb_out))
        return row_emb_out, col_emb_out #, caps_emb_out

class MatNetMHANetworkCaps(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feed_forward_hidden: int = 512,
        bias: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MatNetMHALayerCaps(
                    num_heads=num_heads,
                    embedding_dim=embedding_dim,
                    feed_forward_hidden=feed_forward_hidden,
                    normalization=normalization,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, row_emb, col_emb, dmat, caps_emb):
        """
        Args:
            row_emb (Tensor): [b, m, d]
            col_emb (Tensor): [b, n, d]
            dmat (Tensor): [b, m, n]
            caps_emb (Tensor): [b, d]

        Returns:
            Updated row_emb (Tensor): [b, m, d]
            Updated col_emb (Tensor): [b, n, d]
            Updated caps_emb (Tensor): [b, d]
        """

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, dmat, caps_emb)
        return row_emb, col_emb

class MatNetCapsEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 5,
        normalization: str = "instance",
        feed_forward_hidden: int = 512,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = None,
        bias: bool = False,
    ):
        super().__init__()

        if init_embedding is None:
            init_embedding = env_init_embedding(
                "matnetcaps", {"embedding_dim": embedding_dim, **init_embedding_kwargs}
            )

        self.init_embedding = init_embedding
        self.net = MatNetMHANetworkCaps(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            normalization=normalization,
            feed_forward_hidden=feed_forward_hidden,
            bias=bias,
        )

    def forward(self, td):
        row_emb, col_emb, dmat, caps_emb = self.init_embedding(td)
        row_emb, col_emb = self.net(row_emb, col_emb, dmat, caps_emb)

        embedding = (row_emb, col_emb)
        init_embedding = None
        return embedding, init_embedding  # match output signature for the AR policy class
