import os 
import torch
import networkx as nx
import numpy as np
import random

from typing import *
from itertools import product
from itertools import combinations
from torch_geometric.typing import Adj
from torch.nn.functional import softmax

Tensor = torch.tensor
Pair = Tuple[int, int]

def to_numpy(x: Tensor):
    return x.cpu().detach().numpy()

def create_networkx_graph(adj_list):
    end, origin = adj_list
    graph = nx.Graph()
    edges = []
    for (i, j) in zip(origin, end):
        if i > j:
            continue
        edge = (int(to_numpy(i)), int(to_numpy(j)))
        edges.append(edge)

    graph.add_edges_from(edges)

    return graph

def sp_gen(paths):
    for node_i, path_dict in paths:
        for node_j, path_ij in path_dict.items():
            yield (node_i, node_j), path_ij

def add_to_dict(dict, key, val):
    if key in dict:
        dict[key].append(val)
    else:
        dict[key] = [val] 

def alter_paths(sp, cutoff, method, num_nodes):      
    new_paths = {}  
    if method == 'random':
        new_paths = {k+1: np.empty((0, k+1)) for k in range(2, cutoff+1)}  # for self loops
    elif method in ['sim_walk', 'greedy']:
        new_paths = torch.inf * torch.ones((num_nodes, num_nodes))
        
    add_returns = {} 
    add_returns['k_hop_neighbours'] = {(k+1, node_i): [] for k, node_i in product(range(cutoff), range(num_nodes))}     # some nodes may not be connected

    for (node_i, node_j), path_ij in sp_gen(sp):
        k = len(path_ij)
        if method in ['sim_walk', 'greedy']:
            new_paths[node_i, node_j] = k - 1
        else:
            new_paths[(node_i, node_j)] = k - 1 

        k_hop_neighbours = add_returns['k_hop_neighbours']

        if k == 1:
            continue 

        k_hop_neighbours[(k-1, node_i)].append(node_j)

    return new_paths, add_returns


def random_walk(adj_list, k_hop_neighbours, num_samples, num_nodes, k_hop, device):
    total_visited = 0 
    restart = True 

    for _ in range(2*num_samples):
        if restart:
            node_i = np.random.randint(0, num_nodes)
        
        neighbour_set = k_hop_neighbours[(k_hop, node_i)]
        
        if len(neighbour_set) == 0:
            restart = True
            continue 
        
        idx = np.random.randint(0, len(neighbour_set))
        node_j = neighbour_set[idx]
        add_tensor(adj_list, node_i, node_j, total_visited, device)
        node_i = node_j 
        total_visited += 2 

        if total_visited == num_samples:
            break 

    return adj_list 


def random_select(adj_list, k_hop_neighbours, num_samples, num_nodes, k_hop, device):
    total = 0 

    while total < num_samples:
        node_i = np.random.randint(0, num_nodes)
        neighbours_i = k_hop_neighbours[(k_hop, node_i)]
        if len(neighbours_i) == 0:
            continue 

        idx = np.random.randint(0, len(neighbours_i))
        node_j = neighbours_i[idx]
        add_tensor(adj_list, node_i, node_j, total, device)
        total += 2
    
    return adj_list 


def add_tensor(set_of_tensors, node_i, node_j, index, device):
    forward = torch.tensor([node_i, node_j], device=device, dtype=torch.int64)
    backward = torch.tensor([node_j, node_i], device=device, dtype=torch.int64)
    set_of_tensors[:, index] = forward
    if index < set_of_tensors.shape[-1] - 1:
        set_of_tensors[:, index+1] = backward
    else:
        print('Only forward tensor assignment due to index out of bounds')


def cosine_sim(xj, xt):
    xj = xj.repeat(xt.shape[0], 1)
    norm = torch.norm(xj) * torch.norm(xt)  

    return torch.sum(xj*xt, dim=-1) / norm


def sim_walk(
                adj, 
                k_hop_neighbours, 
                walk_config, 
                shortest_path_lengths, 
                feature_set, 
                k, 
                num_samples,
                device,
                mving_avg=True
            ):
    """
        Populate the adjacency list from the shortest path by taking walks on the graph.
    """
    gamma = walk_config.gamma 
    jump_prob = walk_config.jump_prob
    use_cosine = walk_config.use_cosine 
    num_nodes = feature_set.shape[0]

    max_iters = 2 * num_samples
    total_visited = 0 
    last_visited = -1 

    walk = torch.empty((2, num_samples), device=device, dtype=torch.int64)
    weighted_sum = None
    restart = True

    for iter_idx in range(max_iters):
        if iter_idx % 1000 == 0:
            print(f'{iter_idx}/{max_iters}', walk.shape)
            
        if restart:
            i = torch.randint(0, num_nodes, (1,), device=device).item()
            neighbour_set = k_hop_neighbours[(k, i)]
            weighted_sum = gamma * feature_set[i]
            restart = False

        # Restart condition
        if len(neighbour_set) == 0 or torch.rand(1, device=device).item() <= jump_prob:
            restart = True
            continue

        neighbour_set_tensor = torch.tensor(neighbour_set, device=device, dtype=torch.long)

        # Extract shortest path lengths for neighbors
        neighbor_distances = shortest_path_lengths[neighbour_set_tensor, i] 

        # Create a boolean mask for valid neighbors
        mask = (neighbor_distances == k) & (neighbour_set_tensor != last_visited)

        # Apply the mask to filter valid neighbors
        valid_neighbors = neighbour_set_tensor[mask]

        if valid_neighbors.shape[0] == 0:
            restart = True
            continue

        # Compute similarity scores
        s = 1 - cosine_sim(feature_set[i].unsqueeze(0), feature_set[valid_neighbors])
        if use_cosine:
            s = torch.arccos(s)
        if mving_avg:
            m = 1 - cosine_sim(weighted_sum.unsqueeze(0), feature_set[valid_neighbors])
            s = gamma * s + (1 - gamma) * m

        # Compute probabilities and select node
        probabilities = torch.cumsum(softmax(s, dim=-1), dim=0)
        gt_msk = probabilities > torch.randn((1,), device=device)
        j_index = len(gt_msk) - max(torch.sum(gt_msk).item(), 1) - 1
        j = valid_neighbors[j_index]

        # Update walk tensor
        add_tensor(walk, i, j, total_visited, device)    
        total_visited += 2
        if total_visited >= num_samples:
            break

        # Update for the next iteration
        weighted_sum = gamma * (weighted_sum + feature_set[j])
        last_visited, i = i, j
        neighbour_set = k_hop_neighbours[(k, i.item())]

    return walk


def graph_search(
                    adj_list, 
                    k_hop_neighbours: Dict[int, List[int]], 
                    num_samples: int, 
                    num_nodes: int, 
                    k_hop: int, 
                    method: str, 
                    device
                ):

    assert method in ['bfs', 'dfs'], "Must be either depth first or breadth first search for this setting"

    visited = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)

    start_node = np.random.randint(0, num_nodes)
    queue = [start_node]
    visited[start_node] = True
    previous_node = None
    
    total_visited = 0 

    for i in range(2*num_samples):
        if method == 'bfs':
            node_i = queue.pop(0) 
        elif method == 'dfs':
            node_i = queue.pop()    

        if previous_node is not None:
            add_tensor(adj_list, previous_node, node_i, total_visited, device)
            total_visited += 2
        previous_node = node_i 
        
        neighbors = k_hop_neighbours[(k_hop, node_i)]
        for node_j in neighbors:
            if not visited[node_i, node_j]:
                visited[node_i, node_j] = visited[node_j, node_i] = True
                queue.append(node_j)

        if len(queue) <= 1:
            start_node = np.random.randint(0, num_nodes)
            queue = [start_node]
            visited[start_node] = True
        
        if total_visited >= num_samples - 1: 
            break 

    return adj_list 


def get_shortest_paths(graph, K):
    shortest_paths = nx.all_pairs_shortest_path(graph, K) 

    return shortest_paths

import pickle 

def get_K_adjs(adj_list, model_config, ds_config, feature_set=None, device=torch.device('cpu')) -> List[torch.tensor]:
    K = model_config.K_hops 
    self_conn = model_config.loops
    method = model_config.select_method

    num_samples, num_nodes = adj_list.shape[-1], ds_config.num_nodes

    print('createing networkx graph')

    graph = create_networkx_graph(adj_list)
    adj_lists = [adj_list.to(device) if k == 0 else torch.empty(size=(2, num_samples+num_samples%2), device=device, dtype=torch.int64) for k in range(K)] # could be made faster 

    print('finding shortest paths')

    shortest_paths = get_shortest_paths(graph, K)
    shortest_paths, add_returns = alter_paths(shortest_paths, K, method, num_nodes)

    print('shortest paths complete')
    
    for k in range(1, K):
        k_hop_neighbours = add_returns['k_hop_neighbours']
        hop_dist = k+1 
        
        if method in ['sim_walk', 'greedy']:    
            walk_config = model_config.walk
            shortest_paths = shortest_paths.to(device)
            adj_k = sim_walk(adj_lists[k], k_hop_neighbours, walk_config, shortest_paths, feature_set, hop_dist, \
                             num_samples, device, mving_avg=(method != 'greedy')) 
        elif method == 'random':
            adj_k = random_select(adj_lists[k], k_hop_neighbours, num_samples, num_nodes, hop_dist, device)
        elif method == 'random_walk':                                      
            adj_k = random_walk(adj_lists[k], k_hop_neighbours, num_samples, num_nodes, hop_dist, device)
        elif method in ['bfs', 'dfs']:
            adj_k = graph_search(adj_lists[k], k_hop_neighbours, num_samples, num_nodes, hop_dist, method, device) 
        else:
            raise ValueError('Invalid method specified for periphery graph selection')
        
        adj_lists[k] = adj_k

    return adj_lists


