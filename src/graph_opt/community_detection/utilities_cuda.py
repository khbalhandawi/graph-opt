

import math
import numba
import numpy as np

from numba import float32, float64
from ..utilities_cuda import replace_values, cuda, DeviceNDArray

@cuda.jit
def set_membership(
        start_idx_i:int,
        start_idx_j:int,
        end_idx_i:int,
        end_idx_j:int,
        n_nodes:int,
        node_weight,
        node_score,
        communities:DeviceNDArray,
        membership:DeviceNDArray,
        membership_pairs:DeviceNDArray,
        weights_pairs,
        score_pairs
    ):

    # Compute flattened index inside the array  
    x, y, z = cuda.grid(3)  

    # Compute the grid stride  
    dx, dy, dz = cuda.gridsize(3)  

    batch_size_i = end_idx_i - start_idx_i
    batch_size_j = end_idx_j - start_idx_j

    # Grid stride loop  
    for i in range(x, batch_size_i, dx):
        for j in range(y, batch_size_j, dy):
            for k in range(z, n_nodes, dz):
                # Adjust indices for the current batch
                i_global = i + start_idx_i
                j_global = j + start_idx_j

                community1 = communities[i_global]
                community2 = communities[j_global]
                if community1 == community2:
                    membership_pairs[i,j,k] = membership[k]
                    weights_pairs[i,j,k] = node_weight[k]
                    score_pairs[i,j,k] = node_score[k]
                else:
                    cur_member = membership[k] if membership[k] != community1 else community2
                    membership_pairs[i,j,k] = cur_member

@cuda.jit
def compute_market_share(
        start_idx_i:int,
        start_idx_j:int,
        end_idx_i:int,
        end_idx_j:int,
        m_edges,
        n_communities,
        communities,
        market_shares_sparse_nodes,
        market_shares_sparse_values,
        market_shares_communities,
        market_shares_values,
        membership_pairs,
    ):

    # Compute flattened index inside the array  
    x, y = cuda.grid(2)  

    # Compute the grid stride  
    dx, dy = cuda.gridsize(2)  

    batch_size_i = end_idx_i - start_idx_i
    batch_size_j = end_idx_j - start_idx_j

    # Grid stride loop  
    for i in range(x, batch_size_i, dx):
        for j in range(y, batch_size_j, dy):

            # Adjust indices for the current batch
            i_global = i + start_idx_i
            j_global = j + start_idx_j

            community1 = communities[i_global]
            community2 = communities[j_global]
            if community1 != community2:
                for k in range(n_communities):
                    for i_edge in range(m_edges):
                        if membership_pairs[i,j,market_shares_sparse_nodes[i_edge]] == k:
                            market_shares_communities[i,j,i_edge] = k
                            market_shares_values[i,j,i_edge] = market_shares_sparse_values[i_edge]

@cuda.jit
def calc_hhi(
        start_idx_i:int,
        start_idx_j:int,
        end_idx_i:int,
        end_idx_j:int,
        n_edges,
        m_edges,
        n_communities,
        communities,
        market_shares_sparse_nodes,
        market_shares_sparse_values,
        unique_edge,
        edge_HHI,
        membership_pairs,
        hhi_pairs
    ):

    # Compute flattened index inside the array  
    x, y = cuda.grid(2)  

    # Compute the grid stride  
    dx, dy = cuda.gridsize(2)  

    batch_size_i = end_idx_i - start_idx_i
    batch_size_j = end_idx_j - start_idx_j

    ms_k = cuda.local.array((n_edges,), dtype=float32)

    # Grid stride loop  
    for i in range(x, batch_size_i, dx):
        for j in range(y, batch_size_j, dy):

            # Adjust indices for the current batch
            i_global = i + start_idx_i
            j_global = j + start_idx_j

            community1 = communities[i_global]
            community2 = communities[j_global]
            if community1 == community2:
                for i_edge in range(n_edges):
                    hhi_pairs[i,j,i_edge] = edge_HHI[i_edge]
            else:
                for k in range(n_communities):
                    for i_m in range(n_edges):
                        ms_k[i_m] = 0.0

                    for i_edge in range(m_edges):
                        if membership_pairs[i,j,market_shares_sparse_nodes[i_edge]] == k:
                            ms_k[unique_edge[i_edge]] += market_shares_sparse_values[i_edge]

                    for i_m in range(n_edges):
                        hhi_pairs[i,j,i_m] += ms_k[i_m]**2

@cuda.jit
def calc_ami(
        start_idx_i:int,
        start_idx_j:int,
        end_idx_i:int,
        end_idx_j:int,
        n_nodes,
        m_edges_pairs,
        communities,
        membership_pairs,
        edge_nodes,
        edge_weights,
        edges_neighbour,
        weights_pairs,
        score_pairs
    ):
        
    # Compute flattened index inside the array  
    x, y, z = cuda.grid(3)  

    # Compute the grid stride  
    dx, dy, dz = cuda.gridsize(3)  

    batch_size_i = end_idx_i - start_idx_i
    batch_size_j = end_idx_j - start_idx_j

    # Grid stride loop  
    for i in range(x, end_idx_i - start_idx_i, dx):
        for j in range(y, end_idx_j - start_idx_j, dy):
            i_global = i + start_idx_i
            j_global = j + start_idx_j
            community1 = communities[i_global]
            community2 = communities[j_global]
            if community1 != community2:
                for k in range(z, n_nodes, dz):
                    member_k = membership_pairs[i, j, k]
                    for pi in range(m_edges_pairs):
                        k1 = edges_neighbour[pi,0]
                        k2 = edges_neighbour[pi,1]
                        if edge_nodes[k1] == k and member_k == membership_pairs[i, j, edge_nodes[k2]]:
                            edge_weight_k2 = edge_weights[k2]
                            weights_pairs[i, j, k] += edge_weight_k2
                            score_pairs[i, j, k] += edge_weight_k2