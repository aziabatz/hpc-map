import networkx as nx
import numpy as np
import torch.nn as nn

from node2vec.node2vec import Node2Vec


class MappingInitEmbedding(nn.Module):

    def __init__(
        self,
        num_procs: int,
        embedding_dim: int,
        linear_bias=True,
        normalize=True,
        n2v_kwargs=None,
        w2v_kwargs=None,
    ):
        super(MappingInitEmbedding, self).__init__()

        if n2v_kwargs is None:
            n2v_kwargs = dict(dimensions=10, num_walks=200, workers=4)

        if w2v_kwargs is None:
            w2v_kwargs = dict(window=64, min_count=2, batch_words=4)

        self.n2v_kwargs = n2v_kwargs
        self.w2v_kwargs = w2v_kwargs

        self.embedded_matrix = None
        self.n2v_model = None
        self.n2v = None
        self.num_procs = num_procs
        self.normalize = normalize

    @property
    def embedded(self, matrix):
        if self.embedded_matrix is None:
            self.embedded_matrix = self.embed_matrix(matrix)
        return self.embedded_matrix

    @property
    def nx_graph(self, cost_matrix):

        if self.normalize:
            mat_min = self.cost_matrix.min()
            mat_max = self.cost_matrix.max()
            cost_matrix = (cost_matrix - mat_min) / (mat_max - mat_min)

        edges = list()
        weights = list()

        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix)):
                if cost_matrix[i, j] != 0:
                    edges.append((i + 1, j + 1))
                    weights.append(i + 1, j + 1, cost_matrix[i, j])

        G = nx.Graph()
        G.add_weighted_edges_from(weights)

        return G

    def embed_matrix(self, matrix):
        if self.n2v_model is None:
            G = self.nx_graph(matrix)
            self.n2v = Node2Vec(G, **self.n2v_kwargs)
            self.n2v_model = self.n2v.fit(**self.w2v_kwargs)

        return self.n2v_model.wv.vectors

    def forward(self, td):
        recompute = td["recompute"]
        matrix = td["cost_matrix"]

        if recompute is True:
            self.n2v = None
            self.n2v_model = (None,)
            self.embedded_matrix = None

        return self.embedded(matrix)
