

import math
import numba
import numpy as np
from pandas import isna

from ..utilities_cuda import cuda, DeviceNDArray, float64, float32, int32, int16, \
    xoroshiro128p_uniform_float32, cumsum_d, searchsorted_d, cumsum_global_2d, searchsorted_global_2d

from typing import Callable, Tuple, List

def define_cuda_kernels_sampling(
        n_attr:int,
        n_nodes:int,
    ) -> Tuple[Callable, Callable, Callable]:

    @cuda.jit
    def edge_sampler(
            adj3:DeviceNDArray,
            N_uv:DeviceNDArray,
            rng_states:DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        cumsum = cuda.local.array((n_attr,),dtype=float32)
        weights = cuda.local.array((n_attr,),dtype=float32)

        n_unique_edges, n_samples = N_uv.shape

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_samples, dy):

                    idx = i * dy + j

                    # select a node
                    for n in range(n_attr):
                        weights[n] = adj3[i, n]

                    cumsum_d(weights,cumsum)
                    threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                    chosen = searchsorted_d(cumsum,threshold)

                    N_uv[i,j] = chosen

    @cuda.jit
    def spatial_random_walk(
            adj:DeviceNDArray,
            adj_i:DeviceNDArray,
            V:DeviceNDArray,
            walks_edges:DeviceNDArray,
            rng_states:DeviceNDArray,
            cumsum:DeviceNDArray,
            weights:DeviceNDArray
        ):
        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_nodes, n_samples, walk_length = walks_edges.shape

        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):

                idx = i * dy + j
                current_node = i
                V[i,j,0] = current_node

                for k in range(1,walk_length+1,1):

                    # select spatial node
                    for n in range(n_nodes):
                        weights[i,j,n] = adj[current_node,n]

                    cumsum_global_2d(weights,cumsum,i,j)
                    threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                    if cumsum[i,j,0] == 0 and cumsum[i,j,-1] == 0:
                        chosen = current_node
                    else:
                        chosen = searchsorted_global_2d(cumsum,threshold,i,j)

                    V[i,j,k] = chosen
                    chosen_edge = adj_i[current_node,chosen]
                    if math.isnan(chosen_edge):
                        walks_edges[i,j,k-1] = -1e6
                    else:
                        walks_edges[i,j,k-1] = chosen_edge
                    current_node = chosen

    @cuda.jit
    def attr_sampler(
            walks_edges:DeviceNDArray,
            adj:DeviceNDArray,
            adj3:DeviceNDArray,
            T:DeviceNDArray,
            W:DeviceNDArray,
            rng_states:DeviceNDArray,
        ):
        # Compute flattened index inside the array
        x, y, z = cuda.grid(3)

        # Compute the grid stride
        dx, dy, dz = cuda.gridsize(3)

        cumsum = cuda.local.array((n_attr,),dtype=float32)
        weights = cuda.local.array((n_attr,),dtype=float32)
        
        _, n_samples, walk_length_1 = T.shape

        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):
                for k in range(z, walk_length_1, dz):

                    idx = i * (dy * dz) + j * dz + k

                    current_edge = walks_edges[i,j,k]
                    if math.isnan(current_edge) or current_edge < 0:
                        T[i,j,k] = -1e6
                        W[i,j,k] = 0.0
                    else:
                        # select a node
                        for n in range(n_attr):
                            weights[n] = adj3[current_edge, n]

                        cumsum_d(weights,cumsum)
                        threshold = xoroshiro128p_uniform_float32(rng_states, idx)
                        chosen = searchsorted_d(cumsum,threshold)

                        T[i,j,k] = chosen
                        W[i,j,k] = adj[current_edge]

    return edge_sampler, spatial_random_walk, attr_sampler

def define_cuda_kernels_deterministic() -> Tuple[Callable, Callable, Callable]:

    @cuda.jit
    def edge_postprocess(
            edges_cp:DeviceNDArray,
            membership:DeviceNDArray,
            edges_communities:DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_unique_edges, n_communities = edges_communities.shape
    
        #initialize the array
        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_communities, dy):
                edges_communities[i,j] = 0

        cuda.syncthreads()

        n_unique_edges, n_attr = edges_cp.shape

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_attr, dy):

                cp = edges_cp[i, j]
                k = membership[j]
                cuda.atomic.add(edges_communities[i], k, cp)  # Pass the array and index separately

    @cuda.jit
    def calc_hhi(
            edges_communities: DeviceNDArray,
            edges_hhi: DeviceNDArray
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_unique_edges, n_communities = edges_communities.shape

        #initialize the array
        for i in range(x, n_unique_edges, dx):
            edges_hhi[i] = 0

        cuda.syncthreads()

        # Grid stride loop
        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_communities, dy):
                
                cp = edges_communities[i, j]
                cuda.atomic.add(edges_hhi, i, cp*cp)  # Pass the array and index separately

    @cuda.jit
    def calc_w_it(
            adj: DeviceNDArray,
            adj3: DeviceNDArray,
            w_it: DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_unique_edges, n_attr = adj3.shape
        n_nodes, n_attr = w_it.shape

        for i in range(x, n_nodes, dx):
            for j in range(y, n_attr, dy):
                w_it[i, j] = 0

        cuda.syncthreads()

        # Grid stride loop
        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_attr, dy):

                source = adj[i,0]
                cuda.atomic.add(w_it, (source, j), adj3[i, j])

    @cuda.jit
    def calc_w_i(
            w_it: DeviceNDArray,
            w_i: DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x = cuda.grid(1)

        # Compute the grid stride
        dx = cuda.gridsize(1)

        n_nodes, n_attr = w_it.shape

        for i in range(x, n_nodes, dx):
            w_i[i] = 0

        cuda.syncthreads()

        n_nodes, n_attr = w_it.shape

        # Grid stride loop
        for i in range(x, n_nodes, dx):
            for j in range(n_attr):
                cuda.atomic.add(w_i, i, w_it[i, j])

    @cuda.jit
    def calc_w_ik(
            membership: DeviceNDArray,
            w_it: DeviceNDArray,
            w_ik: DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_nodes, n_communities = w_ik.shape
        for i in range(x, n_nodes, dx):
            for j in range(y, n_communities, dy):
                w_ik[i, j] = 0

        cuda.syncthreads()

        n_nodes, n_attr = w_it.shape

        # Grid stride loop
        for i in range(x, n_nodes, dx):
            for j in range(y, n_attr, dy):
                k = membership[j]
                cuda.atomic.add(w_ik[i], k, w_it[i,j])

    @cuda.jit
    def calc_mpc(
            membership: DeviceNDArray,
            w_it: DeviceNDArray,
            w_ik: DeviceNDArray,
            w_i: DeviceNDArray,
            mpc: DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_nodes, n_attr = w_it.shape
        n_nodes, n_communities = w_ik.shape

        # Grid stride loop
        for t in range(x, n_attr, dx):
            for i in range(y, n_nodes, dy):
                if w_i[i] > 0:
                    k = membership[t]
                    cuda.atomic.add(mpc, t, w_it[i, t] * w_ik[i, k] / (w_i[i] ** 2))

    return edge_postprocess, calc_hhi, calc_w_it, calc_w_i, calc_w_ik, calc_mpc

def define_cuda_kernels_stochastic() -> Tuple[Callable, Callable, Callable, Callable]:

    @cuda.jit
    def edge_sample_postprocess(
            samples:DeviceNDArray,
            membership:DeviceNDArray,
            counts:DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_unique_edges, n_communities = counts.shape

        # initialize counts
        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_communities, dy):
                counts[i,j] = 0

        cuda.syncthreads()

        n_unique_edges, n_samples = samples.shape

        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_samples, dy):

                item = membership[samples[i, j]]
                cuda.atomic.add(counts[i], item, 1)  # Pass the array and index separately

    @cuda.jit
    def calc_sample_hhi(
            counts:DeviceNDArray,
            hhi:DeviceNDArray,
            n_samples:int
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_unique_edges, n_communities = counts.shape

        #initialize the array
        for i in range(x, n_unique_edges, dx):
            hhi[i] = 0

        cuda.syncthreads()

        # Grid stride loop
        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_communities, dy):
                
                cp = counts[i, j] / n_samples
                cuda.atomic.add(hhi, i, cp**2)

    @cuda.jit
    def init_sample_w_i(
            w_i:DeviceNDArray
        ):

        # Compute flattened index inside the array
        x = cuda.grid(1)

        # Compute the grid stride
        dx = cuda.gridsize(1)

        n_nodes = w_i.shape[0]

        for i in range(x, n_nodes, dx):
            w_i[i] = 0.0

    @cuda.jit
    def calc_sample_w_i(
            T_ijk:DeviceNDArray, 
            w_i:DeviceNDArray
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_nodes, n_samples, walk_length = T_ijk.shape
        
        # Iterate over each sample and walk step
        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):
                
                walk_weight = 0.0
                for l in range(0,walk_length,1):  # Start from 1 since 0 is the source node
                    current_attr = T_ijk[i, j, l]

                    # Check if the current node is valid
                    if current_attr < 0:
                        break  # Stop processing this walk if an invalid node is encountered
                    else:
                        walk_weight += 1.0

                cuda.atomic.add(w_i, i, walk_weight)

    @cuda.jit
    def calc_sample_w_it(
            T_ijk:DeviceNDArray, 
            w_it:DeviceNDArray
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_nodes, n_attr = w_it.shape

        for i in range(x, n_nodes, dx):
            for j in range(y, n_attr, dy):
                w_it[i, j] = 0.0

        n_nodes, n_samples, walk_length = T_ijk.shape

        # Iterate over each sample and walk step
        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):
                for l in range(walk_length):  # Start from 1 since 0 is the source node
                    for t in range(n_attr):
                        target_attr = t
                        current_attr = T_ijk[i, j, l]

                        # Check if the current node is valid
                        if current_attr < 0:
                            break  # Stop processing this walk if an invalid node is encountered

                        # Check if current node matches the source node condition
                        if target_attr == current_attr:
                            cuda.atomic.add(w_it[i], t, 1.0)

    @cuda.jit
    def calc_sample_w_ik(
            T_ijk:DeviceNDArray, 
            membership:DeviceNDArray, 
            w_ik:DeviceNDArray
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_nodes, n_communities = w_ik.shape

        for i in range(x, n_nodes, dx):
            for j in range(y, n_communities, dy):
                w_ik[i, j] = 0.0

        n_nodes, n_samples, walk_length = T_ijk.shape

        # Iterate over each sample and walk step
        for i in range(x, n_nodes, dx):
            for j in range(y, n_samples, dy):
                for l in range(walk_length):  # Start from 1 since 0 is the source node
                    for k in range(n_communities):
                        target_community = k
                        current_attr = T_ijk[i, j, l]

                        # Check if the current node is valid
                        if current_attr < 0:
                            break  # Stop processing this walk if an invalid node is encountered
                        
                        current_community = membership[current_attr]
                        # Check if current node matches the source node condition
                        if target_community == current_community:
                            cuda.atomic.add(w_ik[i], k, 1)

    @cuda.jit
    def calc_sample_mpc(
            w_it:DeviceNDArray,
            w_ik:DeviceNDArray,
            w_i:DeviceNDArray,
            membership:DeviceNDArray,
            mpc:DeviceNDArray
        ):
        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_nodes, n_attr = w_it.shape

        for i in range(x, n_nodes, dx):
            for j in range(y, n_attr, dy):

                attr_community = membership[j]
                if w_i[i] > 0:
                    mpc_ij = (w_ik[i,attr_community] * w_it[i,j]) / (w_i[i]*w_i[i])
                else:
                    mpc_ij = 0.0

                cuda.atomic.add(mpc, j, mpc_ij)

    return edge_sample_postprocess, calc_sample_hhi, init_sample_w_i, calc_sample_w_i, calc_sample_w_it, calc_sample_w_ik, calc_sample_mpc

def define_cuda_kernels_sampling_utils() -> Tuple[Callable, Callable]:
    
    @cuda.jit
    def calc_w_k_edges(
            edges_communities: DeviceNDArray,
            w_k: DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_unique_edges, n_communities = edges_communities.shape

        for j in range(y, n_communities, dy):
            w_k[j] = 0.0

        cuda.syncthreads()
        
        n_unique_edges, n_communities = edges_communities.shape
        
        # Iterate over each sample and walk step
        for i in range(x, n_unique_edges, dx):
            for j in range(y, n_communities, dy):
                cuda.atomic.add(w_k, j, edges_communities[i, j] / n_unique_edges)

    @cuda.jit
    def calc_w_k_nodes(
            w_ik: DeviceNDArray,
            w_k: DeviceNDArray,
        ):

        # Compute flattened index inside the array
        x, y = cuda.grid(2)

        # Compute the grid stride
        dx, dy = cuda.gridsize(2)

        n_nodes, n_communities = w_ik.shape
        
        # Iterate over each sample and walk step
        for i in range(x, n_nodes, dx):
            for j in range(y, n_communities, dy):
                cuda.atomic.add(w_k, j, w_ik[i, j] / n_nodes)

    return calc_w_k_edges, calc_w_k_nodes

def define_cuda_kernels_utils(n_samples_edge:int) -> Callable:

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

    return check_edge_distribution