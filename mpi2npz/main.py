import numpy as np
import json
import glob
import os

def init_matrix(P, edges, volume, n_msgs, reward_type="volume", normalize=False):
    adj_matrix = np.zeros((P, P))
    if reward_type == "volume":
        rewards = volume
    elif reward_type == "n_msgs":
        rewards = n_msgs
    else:
        raise ValueError("Unsupported reward type")
    
    for edge, cost in zip(edges, rewards):
        node1, node2 = edge
        adj_matrix[node1][node2] = cost

    if normalize:
        max_cost = np.max(adj_matrix)
        if max_cost > 0:  # Avoid division by zero
            adj_matrix = adj_matrix / max_cost

    return adj_matrix

def process_json_files(folder_path, reward_type="volume", normalize=False):
    file_paths = glob.glob(os.path.join(folder_path, '*.json')) 
    matrices = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)

            P = data["Graph"]["P"]
            edges = data["Graph"]["comms"]["edges"]
            volume = data["Graph"]["comms"]["volume"]
            n_msgs = data["Graph"]["comms"]["n_msgs"]

            adj_matrix = init_matrix(P, edges, volume, n_msgs, reward_type, normalize)
            matrices.append(adj_matrix)

    stacked_matrices = np.stack(matrices, axis=0, dtype=np.float32)
    
    np.savez(f"{folder_path}/all.npz", cost_matrix=stacked_matrices)

process_json_files("./8", reward_type="volume", normalize=False)
