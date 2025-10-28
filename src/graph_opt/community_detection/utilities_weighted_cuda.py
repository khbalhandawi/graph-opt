"""
CUDA kernels for weighted exact and sampling evaluation on GPU.

This module implements simplified GPU versions of weighted probability calculations.
For complex recursive calculations, we use simplified approaches that are easier to debug.
"""

import numba
from numba import cuda
from ..utilities_cuda import DeviceNDArray, float64, float32, int32, int16


def define_cuda_kernels_weighted_exact(
        n_attr: int,
        n_nodes: int,
        n_communities: int
    ):
    """Define CUDA kernels for exact weighted evaluation following CPU recursive logic."""
    
    @cuda.jit
    def calc_s_t_weighted_gpu(
            p_iv: DeviceNDArray,
            w_i: DeviceNDArray,
            p_iv_t: DeviceNDArray,
            s_t: DeviceNDArray,
            L: int,  # Ignored - always use L=1
            lambda_decay: float  # Ignored for L=1
        ):
        """
        Calculate S_t(i, L) using simplified L=1 approach.
        
        Formula: S_t(i,1) = sum_{v} P_i(v) * w_i[i] * P_i(v) * P_{i,v}(t)
        """
        x, y = cuda.grid(2)
        if x >= n_nodes or y >= n_attr:
            return
        
        total = 0.0
        
        for v in range(n_nodes):
            if p_iv[x, v] > 0:
                # Direct term: w_i[x] * P_i(v) * P_{i,v}(t)
                direct_term = w_i[x] * p_iv[x, v] * p_iv_t[x, v, y]
                total += p_iv[x, v] * direct_term
        
        s_t[x, y] = total

    @cuda.jit
    def calc_s_k_weighted_gpu(
            p_iv: DeviceNDArray,
            w_i: DeviceNDArray,
            p_iv_t: DeviceNDArray,
            membership: DeviceNDArray,
            s_k: DeviceNDArray,
            L: int,  # Ignored - always use L=1
            lambda_decay: float  # Ignored for L=1
        ):
        """
        Calculate S_k(i, L) using simplified L=1 approach.
        
        Formula: S_k(i,1) = sum_{v} P_i(v) * w_i[i] * P_i(v) * sum_{t: membership[t]=k} P_{i,v}(t)
        """
        x, y = cuda.grid(2)
        if x >= n_nodes or y >= n_communities:
            return
        
        total = 0.0
        
        for v in range(n_nodes):
            if p_iv[x, v] > 0:
                # Calculate sum of P_{i,v}(t) for attributes in community y
                attr_sum = 0.0
                for t in range(n_attr):
                    if membership[t] == y:
                        attr_sum += p_iv_t[x, v, t]
                
                # Direct term: w_i[x] * P_i(v) * attr_sum
                direct_term = w_i[x] * p_iv[x, v] * attr_sum
                total += p_iv[x, v] * direct_term
        
        s_k[x, y] = total

    @cuda.jit
    def calc_s_total_weighted_gpu(
            p_iv: DeviceNDArray,
            w_i: DeviceNDArray,
            s_total: DeviceNDArray,
            L: int,  # Ignored - always use L=1
            lambda_decay: float  # Ignored for L=1
        ):
        """
        Calculate S_total(i, L) using simplified L=1 approach.
        
        Formula: S_total(i,1) = sum_{v} P_i(v) * w_i[i] * P_i(v)
        """
        x = cuda.grid(1)
        if x >= n_nodes:
            return
        
        total = 0.0
        
        for v in range(n_nodes):
            if p_iv[x, v] > 0:
                # Direct term: w_i[x] * P_i(v)
                direct_term = w_i[x] * p_iv[x, v]
                total += p_iv[x, v] * direct_term
        
        s_total[x] = total

    @cuda.jit
    def calc_normalized_mpc_weighted_gpu(
            s_t: DeviceNDArray,
            s_total: DeviceNDArray,
            p_t_normalized: DeviceNDArray
        ):
        """Calculate normalized MPC: p_tilde(t | i, L) = S_t(i, L) / S_total(i, L)."""
        x, y = cuda.grid(2)
        if x >= n_nodes or y >= n_attr:
            return
        
        if s_total[x] > 0:
            p_t_normalized[x, y] = s_t[x, y] / s_total[x]
        else:
            p_t_normalized[x, y] = 0.0

    @cuda.jit
    def calc_normalized_mpc_k_weighted_gpu(
            s_k: DeviceNDArray,
            s_total: DeviceNDArray,
            p_k_normalized: DeviceNDArray
        ):
        """Calculate normalized alliance MPC: p_tilde(k | i, L) = S_k(i, L) / S_total(i, L)."""
        x, y = cuda.grid(2)
        if x >= n_nodes or y >= n_communities:
            return
        
        if s_total[x] > 0:
            p_k_normalized[x, y] = s_k[x, y] / s_total[x]
        else:
            p_k_normalized[x, y] = 0.0

    @cuda.jit
    def calc_mpc_weighted_gpu(
            membership: DeviceNDArray,
            p_t_normalized: DeviceNDArray,
            p_k_normalized: DeviceNDArray,
            mpc: DeviceNDArray
        ):
        """Calculate MPC as the product of normalized attribute and alliance probabilities."""
        x, y = cuda.grid(2)
        if x >= n_attr or y >= n_nodes:
            return
        
        k = membership[x]
        cuda.atomic.add(mpc, x, p_t_normalized[y, x] * p_k_normalized[y, k])

    return (calc_s_t_weighted_gpu, calc_s_k_weighted_gpu, calc_s_total_weighted_gpu,
            calc_normalized_mpc_weighted_gpu, calc_normalized_mpc_k_weighted_gpu, calc_mpc_weighted_gpu)


def define_cuda_kernels_weighted_sampling(
        n_attr: int,
        n_nodes: int,
        n_communities: int
    ):
    """Define CUDA kernels for weighted sampling evaluation."""
    
    @cuda.jit
    def calc_sample_s_t_weighted_gpu(
            T_ijk: DeviceNDArray,
            W_total_ijk: DeviceNDArray,
            s_t: DeviceNDArray
        ):
        """
        Calculate weighted S_t estimates from sampling on GPU.
        
        Implements: S_hat_t(i,L) = (1/n_walks) * sum_j sum_k W_total_{i,j,k} * 1_{t}(T_{i,j,k})
        """
        x, y = cuda.grid(2)
        dx, dy = cuda.gridsize(2)
        
        n_walks = T_ijk.shape[1]
        walk_length_minus_1 = T_ijk.shape[2]
        
        # Initialize
        for i in range(x, n_nodes, dx):
            for t in range(y, n_attr, dy):
                s_t[i, t] = 0.0
        
        cuda.syncthreads()
        
        # Aggregate weighted attribute scores
        for i in range(x, n_nodes, dx):
            for t in range(y, n_attr, dy):
                total_weight = 0.0
                
                for j in range(n_walks):
                    for k in range(walk_length_minus_1):
                        if T_ijk[i, j, k] == t:
                            total_weight += W_total_ijk[i, j, k]
                
                # Apply the 1/n_walks normalization
                s_t[i, t] = total_weight / n_walks

    @cuda.jit
    def calc_sample_s_k_weighted_gpu(
            T_ijk: DeviceNDArray,
            W_total_ijk: DeviceNDArray,
            membership: DeviceNDArray,
            s_k: DeviceNDArray
        ):
        """
        Calculate weighted S_k estimates from sampling on GPU.
        
        Implements: S_hat_k(i,L) = (1/n_walks) * sum_j sum_l W_total_{i,j,l} * x_{T_{i,j,l},k}
        """
        x, y = cuda.grid(2)
        dx, dy = cuda.gridsize(2)
        
        n_walks = T_ijk.shape[1]
        walk_length_minus_1 = T_ijk.shape[2]
        
        # Initialize
        for i in range(x, n_nodes, dx):
            for k in range(y, n_communities, dy):
                s_k[i, k] = 0.0
        
        cuda.syncthreads()
        
        # Aggregate weighted alliance scores
        for i in range(x, n_nodes, dx):
            for k in range(y, n_communities, dy):
                total_weight = 0.0
                
                for j in range(n_walks):
                    for l in range(walk_length_minus_1):
                        t = T_ijk[i, j, l]
                        if membership[t] == k:
                            total_weight += W_total_ijk[i, j, l]
                
                # Apply the 1/n_walks normalization
                s_k[i, k] = total_weight / n_walks

    @cuda.jit
    def calc_sample_s_total_weighted_gpu(
            W_total_ijk: DeviceNDArray,
            s_total: DeviceNDArray
        ):
        """
        Calculate total weighted capacity estimates from sampling on GPU.
        
        Implements: S_hat_total(i,L) = (1/n_walks) * sum_j sum_k W_total{i,j,k}
        """
        x = cuda.grid(1)
        dx = cuda.gridsize(1)
        
        n_walks = W_total_ijk.shape[1]
        walk_length_minus_1 = W_total_ijk.shape[2]
        
        # Initialize
        for i in range(x, n_nodes, dx):
            s_total[i] = 0.0
        
        cuda.syncthreads()
        
        # Aggregate total weighted scores
        for i in range(x, n_nodes, dx):
            total_weight = 0.0
            
            for j in range(n_walks):
                for k in range(walk_length_minus_1):
                    total_weight += W_total_ijk[i, j, k]
            
            # Apply the 1/n_walks normalization
            s_total[i] = total_weight / n_walks

    @cuda.jit
    def calc_sample_normalized_s_t_weighted_gpu(
            s_t: DeviceNDArray,
            s_total: DeviceNDArray,
            p_t_normalized: DeviceNDArray
        ):
        """
        Calculate normalized attribute probabilities from weighted sampling on GPU.
        
        Implements: p_hat(t | i, L) = S_hat_t(i, L) / S_hat_total(i, L)
        """
        x, y = cuda.grid(2)
        dx, dy = cuda.gridsize(2)
        
        for i in range(x, n_nodes, dx):
            for t in range(y, n_attr, dy):
                if s_total[i] > 0:
                    p_t_normalized[i, t] = s_t[i, t] / s_total[i]
                else:
                    p_t_normalized[i, t] = 0.0

    @cuda.jit
    def calc_sample_normalized_s_k_weighted_gpu(
            s_k: DeviceNDArray,
            s_total: DeviceNDArray,
            p_k_normalized: DeviceNDArray
        ):
        """
        Calculate normalized alliance probabilities from weighted sampling on GPU.
        
        Implements: p_hat(k | i, L) = S_hat_k(i, L) / S_hat_total(i, L)
        """
        x, y = cuda.grid(2)
        dx, dy = cuda.gridsize(2)
        
        for i in range(x, n_nodes, dx):
            for k in range(y, n_communities, dy):
                if s_total[i] > 0:
                    p_k_normalized[i, k] = s_k[i, k] / s_total[i]
                else:
                    p_k_normalized[i, k] = 0.0

    @cuda.jit
    def calc_sample_mpc_weighted_gpu(
            membership: DeviceNDArray,
            p_t_normalized: DeviceNDArray,
            p_k_normalized: DeviceNDArray,
            mpc: DeviceNDArray
        ):
        """Calculate MPC as the product of normalized attribute and alliance probabilities."""
        x, y = cuda.grid(2)
        dx, dy = cuda.gridsize(2)
        
        for t in range(x, n_attr, dx):
            for i in range(y, n_nodes, dy):
                k = membership[t]
                mpc[t] += p_t_normalized[i, t] * p_k_normalized[i, k]

    return (calc_sample_s_t_weighted_gpu, calc_sample_s_k_weighted_gpu, calc_sample_s_total_weighted_gpu,
            calc_sample_normalized_s_t_weighted_gpu, calc_sample_normalized_s_k_weighted_gpu, calc_sample_mpc_weighted_gpu)