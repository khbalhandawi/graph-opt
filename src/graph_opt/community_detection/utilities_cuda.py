

import math
import numba
import numpy as np
from pandas import isna

from ..utilities_cuda import cuda, DeviceNDArray, float64, float32, int32, int16, \
    xoroshiro128p_uniform_float32, cumsum_d, searchsorted_d, cumsum_global_2d, searchsorted_global_2d

from typing import Callable, Tuple

def define_cuda_kernels_deterministic() -> Tuple[Callable, Callable, Callable]:

    @cuda.jit
    def set_batch_membership(
            start_idx_i:int,
            start_idx_j:int,
            end_idx_i:int,
            end_idx_j:int,
            n_attr:int,
            node_weight,
            communities:DeviceNDArray,
            membership:DeviceNDArray,
            membership_pairs:DeviceNDArray,
            weights_pairs,
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
                for k in range(z, n_attr, dz):
                    # Adjust indices for the current batch
                    i_global = i + start_idx_i
                    j_global = j + start_idx_j

                    community1 = communities[i_global]
                    community2 = communities[j_global]
                    if community1 == community2:
                        membership_pairs[i,j,k] = membership[k]
                        weights_pairs[i,j,k] = node_weight[k]
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
            cp_sparse_attr,
            cp_sparse_values,
            unique_edge,
            edge_HHI,
            membership_pairs,
            cp_pairs,
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
                        cp_pairs[i,j,i_edge] = edge_HHI[i_edge]
                elif community1 < community2:
                    for k in range(n_communities):
                        for i_m in range(n_edges):
                            local_array[i,j,i_m] = 0.0

                        for i_edge in range(m_edges):
                            if membership_pairs[i,j,cp_sparse_attr[i_edge]] == k:
                                local_array[i,j,unique_edge[i_edge]] += cp_sparse_values[i_edge]

                        for i_m in range(n_edges):
                            cp_pairs[i,j,i_m] += local_array[i,j,i_m]**2

    @cuda.jit
    def calc_bfs(
            start_idx_i:int,
            start_idx_j:int,
            end_idx_i:int,
            end_idx_j:int,
            n_attr,
            m_edges_pairs,
            communities,
            membership_pairs,
            edge_attr,
            edge_weights,
            edges_neighbour,
            weights_pairs,
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
                    for k in range(z, n_attr, dz):
                        member_k = membership_pairs[i, j, k]
                        for pi in range(m_edges_pairs):
                            k1 = edges_neighbour[pi,0]
                            k2 = edges_neighbour[pi,1]
                            if edge_attr[k1] == k and member_k == membership_pairs[i, j, edge_attr[k2]]:
                                edge_weight_k2 = edge_weights[k2]
                                weights_pairs[i, j, k] += edge_weight_k2

    return set_batch_membership, calc_hhi, calc_bfs

def define_cuda_kernels_stochastic(
        n_attr:int,
        n_nodes:int,
        n_samples_edge:int,
        tile_size:int,
        max_size:int
    ) -> Tuple[Callable, Callable, Callable, Callable]:

    @cuda.jit
    def init_edge_counts(
            n_bins,
            n_unique_edges,
            counts:DeviceNDArray,
        ):
        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_bins, dy):
                counts[i,j] = 0

    @cuda.jit
    def init_attr(
            cum_weight:DeviceNDArray,
        ):
        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        for i in range(x, n_nodes, dx):
            for j in range(y, n_attr, dy):
                cum_weight[i,j] = 0        

    @cuda.jit
    def init_node_weights(node_weights:DeviceNDArray):
        # Compute flattened index inside the array
        x = cuda.grid(1)

        # Compute the grid stride
        dx = cuda.gridsize(1)

        for i in range(x, n_nodes, dx):
            node_weights[i] = 0

    @cuda.jit
    def init_community(
            n_communities:DeviceNDArray,
            community_weights:DeviceNDArray,
        ):
        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        for i in range(x, n_nodes, dx):
            for j in range(y, n_communities, dy):
                community_weights[i,j] = 0

    @cuda.jit
    def set_membership(
            communities:DeviceNDArray,
            membership:DeviceNDArray,
            membership_pairs:DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y, z = cuda.grid(3)

        # Compute the grid stride
        dx, dy, dz = cuda.gridsize(3)

        # Grid stride loop
        for i in range(x, communities.shape[0], dx):
            for j in range(y, communities.shape[0], dy):
                for k in range(z, n_attr, dz):
                    # Adjust indices for the current batch

                    community1 = communities[i]
                    community2 = communities[j]
                    if community1 == community2:
                        membership_pairs[i,j,k] = membership[k]
                    elif community1 != community2:
                        cur_member = membership[k] if membership[k] != community1 else community2
                        membership_pairs[i,j,k] = cur_member

    @cuda.jit
    def spatial_random_walk(
            n_samples,
            walk_length,
            nodes:DeviceNDArray,
            adj:DeviceNDArray,
            adj_i:DeviceNDArray,
            walks_nodes:DeviceNDArray,
            walks_edges:DeviceNDArray,
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
                walks_nodes[i,j,0] = current_node

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

                    walks_nodes[i,j,k+1] = chosen
                    chosen_edge = adj_i[current_node,chosen]
                    if math.isnan(chosen_edge):
                        walks_edges[i,j,k] = -1e6
                    else:
                        walks_edges[i,j,k] = chosen_edge
                    current_node = chosen

    @cuda.jit
    def attr_sampler(
            n_samples,
            walk_length,
            walks_edges:DeviceNDArray,
            adj:DeviceNDArray,
            adj3:DeviceNDArray,
            attr_cp:DeviceNDArray,
            attr_sample:DeviceNDArray,
            rng_states:DeviceNDArray,
        ):
        # Compute flattened index inside the array
        x, y, z = cuda.grid(3)

        # Compute the grid stride
        dx, dy, dz = cuda.gridsize(3)

        cumsum = cuda.local.array((n_attr,),dtype=float32)
        weights = cuda.local.array((n_attr,),dtype=float32)

        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):
                for k in range(z, walk_length-1, dz):

                    idx = i * (dy * dz) + j * dz + k

                    current_edge = walks_edges[i,j,k]
                    if math.isnan(current_edge) or current_edge < 0:
                        attr_sample[i,j,k] = -1e6
                        attr_cp[i,j,k] = 0.0
                    else:
                        # select a node
                        for n in range(n_attr):
                            weights[n] = adj3[current_edge, n]

                        cumsum_d(weights,cumsum)
                        threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                        chosen = searchsorted_d(cumsum,threshold)

                        attr_sample[i,j,k] = chosen
                        attr_cp[i,j,k] = adj[current_edge] * (weights[chosen]/100.0)

    @cuda.jit
    def attr_source_postprocess(
            n_samples,
            attr_sample:DeviceNDArray,
            source_attr_counts:DeviceNDArray
        ):
        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):

                source_attr = attr_sample[i,j,0]
                if not math.isnan(source_attr) and source_attr >= 0:
                    cuda.atomic.add(source_attr_counts[i], source_attr, 1)

    @cuda.jit
    def edge_sampler(
            n_samples,
            n_unique_edges,
            adj3:DeviceNDArray,
            samples:DeviceNDArray,
            rng_states:DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        cumsum = cuda.local.array((n_attr,),dtype=float32)
        weights = cuda.local.array((n_attr,),dtype=float32)

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_samples, dy):

                    idx = i * dy + j

                    # select a node
                    for n in range(n_attr):
                        weights[n] = adj3[i, n]

                    cumsum_d(weights,cumsum)
                    threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                    chosen = searchsorted_d(cumsum,threshold)

                    samples[i,j] = chosen

    @cuda.jit
    def edge_postprocess(
            ci,
            cj,
            n_samples,
            n_unique_edges,
            samples:DeviceNDArray,
            counts:DeviceNDArray,
            membership:DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_samples, dy):

                item = membership[ci,cj,samples[i, j]]
                cuda.atomic.add(counts[i], item, 1)  # Pass the array and index separately

    n_samples_edge+=1

    @cuda.jit
    def check_edge_distribution(counts_base, counts_new, result):
        x = cuda.grid(1)
        dx = cuda.gridsize(1)
        hist1 = cuda.local.array(n_samples_edge, dtype=np.int32)
        hist2 = cuda.local.array(n_samples_edge, dtype=np.int32)

        for i in range(x, counts_base.shape[0], dx):
            result[i] = 0
            # Initialize histograms
            for k in range(n_samples_edge):
                hist1[k] = 0
                hist2[k] = 0

        cuda.syncthreads()

        for i in range(x, counts_base.shape[0], dx):
            
            # Build histograms for the current row
            for j in range(counts_base.shape[1]):
                hist1[counts_base[i, j]] += 1
                hist2[counts_new[i, j]] += 1
            
            # Compare histograms
            for k in range(n_samples_edge):
                if hist1[k] != hist2[k]:
                    result[i] = 1
                    break

    @cuda.jit
    def load_counts(
            ci,
            cj,
            n_bins,
            n_samples,
            n_unique_edges,
            samples:DeviceNDArray,
            counts:DeviceNDArray,
            membership:DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_samples, dy):
                for k in range(n_bins):
                    counts[i][j][k] = 0

        cuda.syncthreads()

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_samples, dy):
                item = membership[ci, cj, samples[i, j]]
                counts[i][j][item] += 1

    @cuda.jit
    def edge_histogram(
            n_bins,
            n_samples,
            n_unique_edges,
            counts:DeviceNDArray,
        ):
        # Size of shared_counts is now determined by tile_size
        shared_counts = cuda.shared.array(shape=(max_size, max_size, tile_size), dtype=int16)

        x, y = cuda.grid(2)
        dx, dy = cuda.gridsize(2)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y

        for tile_start in range(0, n_bins, tile_size):  # loop over tiles
            # Initialize shared memory for this tile
            for k in range(tile_size):
                shared_counts[tx][ty][k] = 0

            cuda.syncthreads()

            for i in range(x, n_unique_edges, dx):
                for j in range(y, n_samples, dy):
                    for k in range(tile_size):
                        if tile_start + k < n_bins:  # check we're not out of bounds
                            shared_counts[tx][ty][k] = counts[i][j][tile_start + k]

                    cuda.syncthreads()

                    stride = 1
                    while stride < max_size:
                        if ty % (2*stride) == 0:
                            for k in range(tile_size):
                                shared_counts[tx][ty][k] += shared_counts[tx][ty+stride][k]
                        stride *= 2
                        cuda.syncthreads()

                    if cuda.threadIdx.y == 0:
                        for k in range(tile_size):
                            if tile_start + k < n_bins:  # check we're not out of bounds
                                counts[i][cuda.blockIdx.y][tile_start + k] = shared_counts[tx][0][k]

    @cuda.jit
    def copy_reduced_counts(
            n_communities,
            n_unique_edges,
            counts:DeviceNDArray,
            output:DeviceNDArray
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_communities, dy):
                output[i][j] = counts[i][0][j]

    @cuda.jit
    def node_weight(
            n_samples, 
            walk_length, 
            samples:DeviceNDArray, 
            samples_weights:DeviceNDArray, 
            node_weight:DeviceNDArray
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        # Iterate over each sample and walk step
        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):
                for l in range(walk_length - 1):  # Start from 1 since 0 is the source node
                    current_attr = samples[i, j, l]

                    # Check if the current node is valid
                    if current_attr < 0:
                        break  # Stop processing this walk if an invalid node is encountered

                    current_weight = samples_weights[i, j, l]
                    cuda.atomic.add(node_weight, i, current_weight)

    @cuda.jit
    def attr_weight(
            n_samples, 
            walk_length, 
            n_attr, 
            samples:DeviceNDArray, 
            samples_weights:DeviceNDArray, 
            attr_weight:DeviceNDArray
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        # Iterate over each sample and walk step
        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):
                for l in range(walk_length - 1):  # Start from 1 since 0 is the source node
                    for k in range(n_attr):
                        target_attr = k
                        current_attr = samples[i, j, l]

                        # Check if the current node is valid
                        if current_attr < 0:
                            break  # Stop processing this walk if an invalid node is encountered

                        current_weight = samples_weights[i, j, l]
                        # Check if current node matches the source node condition
                        if target_attr == current_attr:
                            cuda.atomic.add(attr_weight[i], k, current_weight)

    @cuda.jit
    def community_weight(
            ci, 
            cj, 
            n_samples, 
            walk_length, 
            n_communities,
            samples:DeviceNDArray, 
            samples_weights:DeviceNDArray, 
            membership:DeviceNDArray, 
            community_weight:DeviceNDArray
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        # Iterate over each sample and walk step
        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):
                for l in range(walk_length - 1):  # Start from 1 since 0 is the source node
                    for k in range(n_communities):
                        target_community = k
                        current_attr = samples[i, j, l]

                        # Check if the current node is valid
                        if current_attr < 0:
                            break  # Stop processing this walk if an invalid node is encountered
                        
                        current_community = membership[ci,cj,current_attr]
                        current_weight = samples_weights[i, j, l]
                        # Check if current node matches the source node condition
                        if target_community == current_community:
                            cuda.atomic.add(community_weight[i], k, current_weight)

    @cuda.jit
    def attr_postprocess(
            ci,
            cj,
            n_attr,
            attr_weight:DeviceNDArray,
            community_weight:DeviceNDArray,
            node_weight:DeviceNDArray,
            membership:DeviceNDArray,
            cum_weight:DeviceNDArray
        ):
        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        for i in range(x, n_nodes, dx):
            for j in range(y, n_attr, dy):

                attr_community = membership[ci,cj,j]
                if node_weight[i] > 0:
                    cum_weight[i,j] = (community_weight[i,attr_community] * attr_weight[i,j]) / (node_weight[i]*node_weight[i])
                else:
                    cum_weight[i,j] = 0.0

    return set_membership, spatial_random_walk, \
        attr_sampler, attr_source_postprocess, edge_sampler, \
        init_edge_counts, edge_postprocess, check_edge_distribution, \
        load_counts, edge_histogram, copy_reduced_counts, \
        init_attr, init_node_weights, init_community, node_weight, \
        attr_weight, community_weight, attr_postprocess