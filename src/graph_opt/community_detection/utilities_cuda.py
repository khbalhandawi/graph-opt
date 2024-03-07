

import math
import numba
import numpy as np
from pandas import isna

from ..utilities_cuda import replace_values, cuda, DeviceNDArray, float64, float32, \
    xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64, create_xoroshiro128p_states, \
    cumsum_d, searchsorted_d, cumsum_global_2d, searchsorted_global_2d,cumsum_global_3d, searchsorted_global_3d

from typing import Callable, Tuple

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
                elif community1 < community2:
                    cur_member = membership[k] if membership[k] != community1 else community2
                    membership_pairs[i,j,k] = cur_member

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
        hhi_pairs,
        local_array,
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
            if community1 == community2:
                for i_edge in range(n_edges):
                    hhi_pairs[i,j,i_edge] = edge_HHI[i_edge]
            elif community1 < community2: 
                for k in range(n_communities):
                    for i_m in range(n_edges):
                        local_array[i,j,i_m] = 0.0

                    for i_edge in range(m_edges):
                        if membership_pairs[i,j,market_shares_sparse_nodes[i_edge]] == k:
                            local_array[i,j,unique_edge[i_edge]] += market_shares_sparse_values[i_edge]

                    for i_m in range(n_edges):
                        hhi_pairs[i,j,i_m] += local_array[i,j,i_m]**2

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
    for i in range(x, batch_size_i, dx):
        for j in range(y, batch_size_j, dy):
            i_global = i + start_idx_i
            j_global = j + start_idx_j
            community1 = communities[i_global]
            community2 = communities[j_global]
            if community1 < community2:
                for k in range(z, n_nodes, dz):
                    member_k = membership_pairs[i, j, k]
                    for pi in range(m_edges_pairs):
                        k1 = edges_neighbour[pi,0]
                        k2 = edges_neighbour[pi,1]
                        if edge_nodes[k1] == k and member_k == membership_pairs[i, j, edge_nodes[k2]]:
                            edge_weight_k2 = edge_weights[k2]
                            weights_pairs[i, j, k] += edge_weight_k2
                            score_pairs[i, j, k] += edge_weight_k2

@cuda.jit
def spatial_random_walk(
        n_nodes,
        n_samples,
        walk_length,
        nodes:DeviceNDArray,
        adj:DeviceNDArray,
        adj_i:DeviceNDArray,
        walks:DeviceNDArray,
        walks_edges_d:DeviceNDArray,
        rng_states:DeviceNDArray,
        cumsum:DeviceNDArray,
        weights:DeviceNDArray
    ):
    # Compute flattened index inside the array  
    x, y = cuda.grid(2)  

    # Compute the grid stride 
    dx, dy = cuda.gridsize(2)  

    for i in range(x, n_nodes, dx):
        for j in range(y, n_samples, dy):

                idx = i * dy + j
                current_node = nodes[i]
                walks[i,j,0] = current_node

                for k in range(walk_length - 1):

                    # select spatial node
                    for n in range(n_nodes):
                        weights[i,j,n] = adj[current_node,n]

                    cumsum_global_2d(weights,cumsum,i,j)
                    threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                    if cumsum[i,j,0] == 0 and cumsum[i,j,-1] == 0:
                        chosen = current_node
                    else:
                        chosen = searchsorted_global_2d(cumsum,threshold,i,j)

                    walks[i,j,k+1] = chosen
                    chosen_edge = adj_i[current_node,chosen]
                    if math.isnan(chosen_edge):
                        walks_edges_d[i,j,k] = -1e6
                    else:
                        walks_edges_d[i,j,k] = chosen_edge
                    current_node = chosen

@cuda.jit
def attr_sampler(
        n_nodes,
        n_nodes_spatial,
        n_samples,
        walk_length,
        walks_edges:DeviceNDArray,
        adj3:DeviceNDArray,
        samples:DeviceNDArray,
        rng_states:DeviceNDArray,
        cumsum:DeviceNDArray,
        weights:DeviceNDArray
    ):
    # Compute flattened index inside the array  
    x, y, z = cuda.grid(3)  

    # Compute the grid stride 
    dx, dy, dz = cuda.gridsize(3)  

    for i in range(x, n_nodes_spatial, dx):
        for j in range(y, n_samples, dy):
            for k in range(z, walk_length-1, dz):

                idx = i * (dy * dz) + j * dz + k

                current_edge = walks_edges[i,j,k]
                if math.isnan(current_edge) or current_edge < 0:
                    samples[i,j,k] = -1e6
                else:
                    # select a node
                    for n in range(n_nodes):
                        weights[n] = adj3[current_edge, n]

                    cumsum_global_3d(weights,cumsum,i,j,k)
                    threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                    chosen = searchsorted_global_3d(cumsum,threshold,i,j,k)

                    samples[i,j,k] = chosen

def define_cuda_kernels(n_nodes:int,n_nodes_spatial:int) -> Tuple[Callable, Callable]:

    @cuda.jit
    def spatial_random_walk(
            n_samples,
            walk_length,
            nodes:DeviceNDArray,
            adj:DeviceNDArray,
            adj_i:DeviceNDArray,
            walks:DeviceNDArray,
            walks_edges_d:DeviceNDArray,
            rng_states:DeviceNDArray,
            cumsum:DeviceNDArray,
            weights:DeviceNDArray
        ):
        # Compute flattened index inside the array  
        x, y = cuda.grid(2)  

        # Compute the grid stride 
        dx, dy = cuda.gridsize(2)  

        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):

                    idx = i * dy + j
                    current_node = nodes[i]
                    walks[i,j,0] = current_node

                    for k in range(walk_length - 1):

                        # select spatial node
                        for n in range(n_nodes):
                            weights[i,j,n] = adj[current_node,n]

                        cumsum_global_2d(weights,cumsum,i,j)
                        threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                        if cumsum[i,j,0] == 0 and cumsum[i,j,-1] == 0:
                            chosen = current_node
                        else:
                            chosen = searchsorted_global_2d(cumsum,threshold,i,j)

                        walks[i,j,k+1] = chosen
                        chosen_edge = adj_i[current_node,chosen]
                        if math.isnan(chosen_edge):
                            walks_edges_d[i,j,k] = -1e6
                        else:
                            walks_edges_d[i,j,k] = chosen_edge
                        current_node = chosen

    @cuda.jit
    def attr_sampler(
            n_samples,
            walk_length,
            walks_edges:DeviceNDArray,
            adj:DeviceNDArray,
            adj3:DeviceNDArray,
            samples_weights:DeviceNDArray,
            samples:DeviceNDArray,
            rng_states:DeviceNDArray,
        ):
        # Compute flattened index inside the array  
        x, y, z = cuda.grid(3)  

        # Compute the grid stride 
        dx, dy, dz = cuda.gridsize(3)  

        cumsum = cuda.local.array((n_nodes,),dtype=float32)
        weights = cuda.local.array((n_nodes,),dtype=float32)

        for i in range(x, n_nodes_spatial, dx):
            for j in range(y, n_samples, dy):
                for k in range(z, walk_length-1, dz):

                    idx = i * (dy * dz) + j * dz + k

                    current_edge = walks_edges[i,j,k]
                    if math.isnan(current_edge) or current_edge < 0:
                        samples[i,j,k] = -1e6
                        samples_weights[i,j,k] = 0.0
                    else:
                        # select a node
                        for n in range(n_nodes):
                            weights[n] = adj3[current_edge, n]

                        cumsum_d(weights,cumsum)
                        threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                        chosen = searchsorted_d(cumsum,threshold)

                        samples[i,j,k] = chosen
                        samples_weights[i,j,k] = adj[current_edge] * (weights[chosen]/100.0)

    return spatial_random_walk, attr_sampler