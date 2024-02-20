import networkx as nx
import numpy as np
import torch.nn as nn

from node2vec.node2vec import Node2Vec


class MappingInitEmbedding(nn.Module):
    """
    A class for initializing and obtaining vector embeddings for cost matrices using Node2Vec.

    This class creates embeddings for every node of a graph generated from a cost (or adjacency) matrix.
    It is based on Node2Vec to generate those embeddings and represent
    the nodes in a continuous vector space.

    Attributes:
        num_procs (int): Number of processes for parallelizing graph creation and Node2Vec training.
        embedding_dim (int): Dimension of the vector space for the node embeddings.
        linear_bias (bool): Whether to force a linear bias. NOT USED, only to preserve RL4CO embedding classes declaration.
        normalize (bool): Whether to normalize the cost matrix before generating the graph.
        n2v_kwargs (dict): Additional arguments for configuring Node2Vec.
        w2v_kwargs (dict): Additional arguments for configuring Word2Vec during Node2Vec training.
    """

    def __init__(
        self,
        embedding_dim,
        linear_bias=True,  # Unused
        normalize=True,
        n2v_kwargs=None,
        w2v_kwargs=None,
    ):
        """
        Initializes the class with the given configuration.

        Parameters:
            embedding_dim (int): Dimension of the node embeddings.
            linear_bias (bool, optional): Force linear bias (unused). Defaults to True.
            normalize (bool, optional): Normalize the cost matrix before graph generation. Defaults to True.
            n2v_kwargs (dict, optional): Node2Vec configuration arguments. Defaults to a preset if None.
            w2v_kwargs (dict, optional): Word2Vec training configuration arguments for Node2Vec. Defaults to a preset if None.
        """
        super(MappingInitEmbedding, self).__init__()

        if n2v_kwargs is None:
            n2v_kwargs = dict(dimensions=embedding_dim, num_walks=200, workers=4)

        if w2v_kwargs is None:
            w2v_kwargs = dict(window=64, min_count=2, batch_words=4)

        self.n2v_kwargs = n2v_kwargs
        self.w2v_kwargs = w2v_kwargs

        self.embedded_matrix = None
        self.n2v_model = None
        self.n2v = None

        self.normalize = normalize

    @property
    def embedded(self, matrix):
        """
        Generates or retrieves already computed embeddings for a given matrix.

        Parameters:
            matrix (np.ndarray): Cost matrix from which to generate the graph and embeddings.

        Returns:
            np.ndarray: Generated or cached embeddings for each node in the graph.
        """
        if self.embedded_matrix is None:
            self.embedded_matrix = self.embed_matrix(matrix)
        return self.embedded_matrix

    @property
    def nx_graph(self, cost_matrix):
        """
        Generates a NetworkX graph from a given cost matrix, optionally normalizing the matrix.

        Parameters:
            cost_matrix (np.ndarray): The cost matrix to generate the graph from.

        Returns:
            nx.Graph: The generated graph with weighted edges based on the cost matrix.
        """
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
        """
        Generates embeddings for a matrix using Node2Vec, caching the model and embeddings for future use.

        Parameters:
            matrix (np.ndarray): The cost matrix to generate embeddings for.

        Returns:
            np.ndarray: The generated embeddings for the matrix.
        """
        if self.n2v_model is None:
            G = self.nx_graph(matrix)
            self.n2v = Node2Vec(G, **self.n2v_kwargs)
            self.n2v_model = self.n2v.fit(**self.w2v_kwargs)

        return self.n2v_model.wv.vectors

    def forward(self, td):
        """
        PyTorch forward method implementation. It takes a dictionary containing a cost matrix and a recompute flag,
        and returns the corresponding embeddings.

        Parameters:
            td (dict): Dictionary with keys "cost_matrix" for the cost matrix and "recompute" indicating whether
                    to recompute embeddings or use cached ones.

        Returns:
            np.ndarray: The embeddings for the given cost matrix.
        """
        recompute = td["recompute"]
        matrix = td["cost_matrix"]

        if recompute is True:
            self.n2v = None
            self.n2v_model = (None,)
            self.embedded_matrix = None

        return self.embedded(matrix)
