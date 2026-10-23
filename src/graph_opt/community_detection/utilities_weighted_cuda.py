"""
CUDA kernels for weighted exact and sampling evaluation on GPU.

The exact kernels compute the depth-L recursion one layer per launch; the sampling kernels
aggregate the random-walk tensors produced by NeighbourSampler.
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
    
    # The three score kernels below compute one layer of the depth-L recursion of the CPU
    # implementation (utilities_weighted_cpu.py):
    #     S(i, d) = sum_v P_i(v) * [w_i * P_i(v) * X_{i,v} + lambda * S(v, d-1)],  S(i, 0) = 0
    # Each launch reads S(., d-1) from s_prev and writes S(., d) to s_out; the host launches
    # them L times, swapping the two buffers (see ExactWeightedEvaluator._recurse_weighted_gpu).

    @cuda.jit
    def calc_s_t_weighted_gpu(
            p_iv: DeviceNDArray,
            w_i: DeviceNDArray,
            p_iv_t: DeviceNDArray,
            s_prev: DeviceNDArray,
            s_out: DeviceNDArray,
            lambda_decay: float
        ):
        """One layer of S_t(i,d) = sum_v P_i(v) * [w_i * P_i(v) * P_{i,v}(t) + lambda * S_t(v,d-1)]."""
        x, y = cuda.grid(2)
        if x >= n_nodes or y >= n_attr:
            return

        total = 0.0
        for v in range(n_nodes):
            if p_iv[x, v] > 0:
                direct_term = w_i[x] * p_iv[x, v] * p_iv_t[x, v, y]
                total += p_iv[x, v] * (direct_term + lambda_decay * s_prev[v, y])
        s_out[x, y] = total

    @cuda.jit
    def calc_s_k_weighted_gpu(
            p_iv: DeviceNDArray,
            w_i: DeviceNDArray,
            p_iv_t: DeviceNDArray,
            membership: DeviceNDArray,
            s_prev: DeviceNDArray,
            s_out: DeviceNDArray,
            lambda_decay: float
        ):
        """One layer of S_k(i,d) = sum_v P_i(v) * [w_i * P_i(v) * sum_{t in k} P_{i,v}(t) + lambda * S_k(v,d-1)]."""
        x, y = cuda.grid(2)
        if x >= n_nodes or y >= n_communities:
            return

        total = 0.0
        for v in range(n_nodes):
            if p_iv[x, v] > 0:
                attr_sum = 0.0
                for t in range(n_attr):
                    if membership[t] == y:
                        attr_sum += p_iv_t[x, v, t]
                direct_term = w_i[x] * p_iv[x, v] * attr_sum
                total += p_iv[x, v] * (direct_term + lambda_decay * s_prev[v, y])
        s_out[x, y] = total

    @cuda.jit
    def calc_s_total_weighted_gpu(
            p_iv: DeviceNDArray,
            w_i: DeviceNDArray,
            s_prev: DeviceNDArray,
            s_out: DeviceNDArray,
            lambda_decay: float
        ):
        """One layer of S_total(i,d) = sum_v P_i(v) * [w_i * P_i(v) + lambda * S_total(v,d-1)]."""
        x = cuda.grid(1)
        if x >= n_nodes:
            return

        total = 0.0
        for v in range(n_nodes):
            if p_iv[x, v] > 0:
                direct_term = w_i[x] * p_iv[x, v]
                total += p_iv[x, v] * (direct_term + lambda_decay * s_prev[v])
        s_out[x] = total

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
            T_ijk,         # int32[:, :, :]
            W_total_ijk,   # float32[:, :, :]
            s_t            # float32[:, :]
        ):
        """
        GPU equivalent of calc_sample_s_t_weighted_cpu.
        Implements:
        S_hat_t(i,L) = (1/n_walks) * sum_j sum_k W_total_{i,j,k} * 1{ T_{i,j,k} == t }
        Early-terminates a walk when T_ijk[i,j,k] < 0 (invalid).
        """
        x, y = cuda.grid(2)
        dx, dy = cuda.gridsize(2)

        n_nodes     = T_ijk.shape[0]
        n_walks     = T_ijk.shape[1]
        walk_len_m1 = T_ijk.shape[2]
        n_attr      = s_t.shape[1]

        # grid-stride over (i, t) pairs; each thread writes s_t[i, t] exactly once
        for i in range(x, n_nodes, dx):
            for t in range(y, n_attr, dy):
                total = float32(0.0)

                for j in range(n_walks):
                    for k in range(walk_len_m1):
                        a = T_ijk[i, j, k]
                        if a < 0:
                            break  # match CPU: stop consuming the rest of this walk
                        if a == t:
                            total += W_total_ijk[i, j, k]

                s_t[i, t] = total / float32(n_walks)

    @cuda.jit
    def calc_sample_s_k_weighted_gpu(
            T_ijk,            # int32[:, :, :]
            W_total_ijk,      # float32[:, :, :]
            membership,       # int32[:]
            s_k               # float32[:, :]
        ):
        """
        GPU version equivalent to calc_sample_s_k_weighted_cpu.
        Implements:
        S_hat_k(i,L) = (1/n_walks) * sum_j sum_l W_total_{i,j,l} * 1{ membership[T_{i,j,l}] == k }
        Early-terminates a walk when T_ijk[i,j,l] < 0 (invalid).
        """
        x, y = cuda.grid(2)
        dx, dy = cuda.gridsize(2)

        n_nodes        = T_ijk.shape[0]
        n_walks        = T_ijk.shape[1]
        walk_len_m1    = T_ijk.shape[2]
        n_communities  = s_k.shape[1]

        # grid-stride over (i, k) pairs; each thread writes a unique s_k[i,k]
        for i in range(x, n_nodes, dx):
            for k in range(y, n_communities, dy):
                total_weight = float32(0.0)

                for j in range(n_walks):
                    # consume this walk until invalid attr
                    for l in range(walk_len_m1):
                        t = T_ijk[i, j, l]
                        if t < 0:
                            break  # stop this walk early, match CPU behavior
                        if membership[t] == k:
                            total_weight += W_total_ijk[i, j, l]

                # normalize by number of walks (float)
                s_k[i, k] = total_weight / float32(n_walks)

    @cuda.jit
    def calc_sample_s_total_weighted_gpu(W_total_ijk, s_total):
        """
        GPU equivalent of calc_sample_s_total_weighted_cpu.
        S_hat_total(i,L) = (1/n_walks) * sum_j sum_k W_total_{i,j,k}
        """
        x = cuda.grid(1)
        dx = cuda.gridsize(1)

        n_nodes  = W_total_ijk.shape[0]
        n_walks  = W_total_ijk.shape[1]
        wl_m1    = W_total_ijk.shape[2]

        # grid-stride over i; each thread writes s_total[i] once (no atomics needed)
        for i in range(x, n_nodes, dx):
            total = 0.0  # float64 if your arrays are float64; cast to float32 if they are float32
            for j in range(n_walks):
                for k in range(wl_m1):
                    total += W_total_ijk[i, j, k]

            s_total[i] = total / float32(n_walks)  # use float32 or float64 to match your dtype

    @cuda.jit
    def calc_sample_normalized_p_t_weighted_gpu(
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
    def calc_sample_normalized_p_k_weighted_gpu(
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
                mpc_it = p_t_normalized[i, t] * p_k_normalized[i, k]

                cuda.atomic.add(mpc, t, mpc_it)

    return (calc_sample_s_t_weighted_gpu, calc_sample_s_k_weighted_gpu, calc_sample_s_total_weighted_gpu,
            calc_sample_normalized_p_t_weighted_gpu, calc_sample_normalized_p_k_weighted_gpu, calc_sample_mpc_weighted_gpu)