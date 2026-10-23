"""
Utility functions for weighted exact evaluation on CPU.

This module implements the recursive weighted probability calculations
for computing S_t(i,L) and S_k(i,L) based on edge weights.
"""

import numpy as np
import numba


def calc_s_t_weighted_cpu(
        p_iv,
        w_i,
        p_iv_t,
        L,
        lambda_decay,
        s_t
    ):
    """
    Calculate S_t(i, L) recursively using the weighted formulation.
    
    S_t(i,L) = sum_{v in N_i} P_i(v) * [w_i * P_i(v) * P_{i,v}(t) + lambda * S_t(v, L-1)]
    
    Parameters
    ----------
    p_iv : np.ndarray
        Neighbor PMF of shape (n_nodes, n_nodes), computed as P_i(v) = w_{i,v} / w_i
    w_i : np.ndarray
        Total outgoing weights of shape (n_nodes,), equivalent to Z_i
    p_iv_t : np.ndarray
        Conditional attribute PMF of shape (n_nodes, n_nodes, n_attr)
    L : int
        Walk depth
    lambda_decay : float
        Decay factor for recursive term
    s_t : np.ndarray
        Output array of shape (n_nodes, n_attr) to store attribute scores
    """
    n_nodes, n_attr = s_t.shape
    
    # Recursive function to calculate S_t(i, depth)
    def calc_s_t_recursive(i, t, depth):
        if depth == 0:
            return 0.0
        
        sum_v = 0.0
        
        for v in range(n_nodes):
            if p_iv[i, v] == 0:
                continue
                
            # Direct term: w_i * P_i(v) * P_{i,v}(t)
            direct_term = w_i[i] * p_iv[i, v] * p_iv_t[i, v, t]
            
            # Recursive term: lambda * S_t(v, depth-1)
            recursive_term = lambda_decay * calc_s_t_recursive(v, t, depth - 1)
            
            sum_v += p_iv[i, v] * (direct_term + recursive_term)
        
        return sum_v
    
    # Calculate for each node and attribute
    for i in range(n_nodes):
        for t in range(n_attr):
            s_t[i, t] = calc_s_t_recursive(i, t, L)


def calc_s_k_weighted_cpu(
        p_iv,
        w_i,
        p_iv_t,
        L,
        lambda_decay,
        membership,
        s_k
    ):
    """
    Calculate S_k(i, L) recursively using the weighted formulation.
    
    S_k(i,L) = sum_{v in N_i} P_i(v) * [w_i * P_i(v) * sum_{t: x_t,k=1} P_{i,v}(t) + lambda * S_k(v, L-1)]
    
    Parameters
    ----------
    p_iv : np.ndarray
        Neighbor PMF of shape (n_nodes, n_nodes)
    w_i : np.ndarray
        Total outgoing weights of shape (n_nodes,)
    p_iv_t : np.ndarray
        Conditional attribute PMF of shape (n_nodes, n_nodes, n_attr)
    L : int
        Walk depth
    lambda_decay : float
        Decay factor for recursive term
    membership : np.ndarray
        Alliance membership array of shape (n_attr,) mapping attributes to alliances
    s_k : np.ndarray
        Output array of shape (n_nodes, n_communities) to store alliance scores
    """
    n_nodes, n_communities = s_k.shape
    n_attr = p_iv_t.shape[2]
    
    # Recursive function to calculate S_k(i, depth)
    def calc_s_k_recursive(i, k, depth):
        if depth == 0:
            return 0.0
        
        sum_v = 0.0
        
        for v in range(n_nodes):
            if p_iv[i, v] == 0:
                continue
            
            # Direct term: w_i * P_i(v) * sum_{t: membership[t]=k} P_{i,v}(t)
            direct_sum = 0.0
            for t in range(n_attr):
                if membership[t] == k:
                    direct_sum += p_iv_t[i, v, t]
            
            direct_term = w_i[i] * p_iv[i, v] * direct_sum
            
            # Recursive term: lambda * S_k(v, depth-1)
            recursive_term = lambda_decay * calc_s_k_recursive(v, k, depth - 1)
            
            sum_v += p_iv[i, v] * (direct_term + recursive_term)
        
        return sum_v
    
    # Calculate for each node and community
    for i in range(n_nodes):
        for k in range(n_communities):
            s_k[i, k] = calc_s_k_recursive(i, k, L)


def calc_s_total_weighted_cpu(
        p_iv,
        w_i,
        L,
        lambda_decay,
        s_total
    ):
    """
    Calculate S_total(i, L) recursively (attribute-agnostic capacity score).
    
    S_total(i,L) = sum_{v in N_i} P_i(v) * [w_i * P_i(v) + lambda * S_total(v, L-1)]
    
    Parameters
    ----------
    p_iv : np.ndarray
        Neighbor PMF of shape (n_nodes, n_nodes)
    w_i : np.ndarray
        Total outgoing weights of shape (n_nodes,)
    L : int
        Walk depth
    lambda_decay : float
        Decay factor for recursive term
    s_total : np.ndarray
        Output array of shape (n_nodes,) to store total capacity scores
    """
    n_nodes = s_total.shape[0]
    
    # Recursive function to calculate S_total(i, depth)
    def calc_s_total_recursive(i, depth):
        if depth == 0:
            return 0.0
        
        sum_v = 0.0
        
        for v in range(n_nodes):
            if p_iv[i, v] == 0:
                continue
            
            # Direct term: w_i * P_i(v)
            direct_term = w_i[i] * p_iv[i, v]
            
            # Recursive term: lambda * S_total(v, depth-1)
            recursive_term = lambda_decay * calc_s_total_recursive(v, depth - 1)
            
            sum_v += p_iv[i, v] * (direct_term + recursive_term)
        
        return sum_v
    
    # Calculate for each node
    for i in range(n_nodes):
        s_total[i] = calc_s_total_recursive(i, L)


@numba.njit
def calc_normalized_p_t_weighted_cpu(
        s_t,
        s_total,
        p_t_normalized
    ):
    """
    Calculate normalized MPC: p_tilde(t | i, L) = S_t(i, L) / S_total(i, L).
    
    Parameters
    ----------
    s_t : np.ndarray
        Attribute scores of shape (n_nodes, n_attr)
    s_total : np.ndarray
        Total capacity scores of shape (n_nodes,)
    p_t_normalized : np.ndarray
        Output normalized probabilities of shape (n_nodes, n_attr)
    """
    n_nodes, n_attr = s_t.shape
    
    for i in range(n_nodes):
        for t in range(n_attr):
            if s_total[i] > 0:
                p_t_normalized[i, t] = s_t[i, t] / s_total[i]
            else:
                p_t_normalized[i, t] = 0.0


@numba.njit
def calc_normalized_p_k_weighted_cpu(
        s_k,
        s_total,
        p_k_normalized
    ):
    """
    Calculate normalized alliance MPC: p_tilde(k | i, L) = S_k(i, L) / S_total(i, L).
    
    Parameters
    ----------
    s_k : np.ndarray
        Alliance scores of shape (n_nodes, n_communities)
    s_total : np.ndarray
        Total capacity scores of shape (n_nodes,)
    p_k_normalized : np.ndarray
        Output normalized probabilities of shape (n_nodes, n_communities)
    """
    n_nodes, n_communities = s_k.shape
    
    for i in range(n_nodes):
        for k in range(n_communities):
            if s_total[i] > 0:
                p_k_normalized[i, k] = s_k[i, k] / s_total[i]
            else:
                p_k_normalized[i, k] = 0.0


@numba.njit
def calc_mpc_weighted_cpu(
        membership,
        p_t_normalized,
        p_k_normalized,
        mpc
    ):
    """
    Calculate MPC as the product of normalized attribute and alliance probabilities.
    
    MPC[t] = sum_i p_tilde(t | i, L) * p_tilde(k | i, L) where k = membership[t]
    
    Parameters
    ----------
    membership : np.ndarray
        Alliance membership of shape (n_attr,)
    p_t_normalized : np.ndarray
        Normalized attribute probabilities of shape (n_nodes, n_attr)
    p_k_normalized : np.ndarray
        Normalized alliance probabilities of shape (n_nodes, n_communities)
    mpc : np.ndarray
        Output MPC array of shape (n_attr,)
    """
    n_nodes, n_attr = p_t_normalized.shape
    
    for t in range(n_attr):
        mpc[t] = 0.0
        k = membership[t]
        for i in range(n_nodes):
            mpc[t] += p_t_normalized[i, t] * p_k_normalized[i, k]


# Weighted sampling functions for SampleWeightedEvaluator
@numba.njit
def calc_sample_s_t_weighted_cpu(
        T_ijk,
        W_total_ijk,
        s_t
    ):
    """
    Calculate weighted S_t estimates from sampling.
    
    Implements: S_hat_t(i,L) = (1/n_walks) * sum_j sum_k W_total_{i,j,k} * 1_{t}(T_{i,j,k})
    
    Parameters
    ----------
    T_ijk : np.ndarray
        Attribute tensor of shape (n_nodes, n_walks, walk_length-1)
    W_total_ijk : np.ndarray  
        Total weight tensor of shape (n_nodes, n_walks, walk_length-1)
    s_t : np.ndarray
        Output weighted attribute scores of shape (n_nodes, n_attr)
    """
    n_nodes, n_walks, walk_length_minus_1 = T_ijk.shape
    n_nodes, n_attr = s_t.shape
    
    # Initialize
    for i in range(n_nodes):
        for t in range(n_attr):
            s_t[i, t] = 0.0
    
    # Aggregate weighted attribute scores
    for i in range(n_nodes):
        for j in range(n_walks):
            for k in range(walk_length_minus_1):
                current_attr = T_ijk[i, j, k]
                current_weight = W_total_ijk[i, j, k]
                
                # Check if the current attribute is valid
                if current_attr < 0:
                    break  # Stop processing this walk if invalid
                
                s_t[i, current_attr] += current_weight
    
    # Apply the 1/n_walks normalization
    for i in range(n_nodes):
        for t in range(n_attr):
            s_t[i, t] /= n_walks


@numba.njit
def calc_sample_s_k_weighted_cpu(
        T_ijk,
        W_total_ijk,
        membership,
        s_k
    ):
    """
    Calculate weighted S_k estimates from sampling.
    
    Implements: S_hat_k(i,L) = (1/n_walks) * sum_j sum_l W_total_{i,j,l} * x_{T_{i,j,l},k}
    
    Parameters
    ----------
    T_ijk : np.ndarray
        Attribute tensor of shape (n_nodes, n_walks, walk_length-1)
    W_total_ijk : np.ndarray
        Total weight tensor of shape (n_nodes, n_walks, walk_length-1)
    membership : np.ndarray
        Alliance membership of shape (n_attr,)
    s_k : np.ndarray
        Output weighted alliance scores of shape (n_nodes, n_communities)
    """
    n_nodes, n_walks, walk_length_minus_1 = T_ijk.shape
    n_nodes, n_communities = s_k.shape
    
    # Initialize
    for i in range(n_nodes):
        for k in range(n_communities):
            s_k[i, k] = 0.0
    
    # Aggregate weighted alliance scores
    for i in range(n_nodes):
        for j in range(n_walks):
            for l in range(walk_length_minus_1):
                current_attr = T_ijk[i, j, l]
                current_weight = W_total_ijk[i, j, l]
                
                # Check if the current attribute is valid
                if current_attr < 0:
                    break  # Stop processing this walk if invalid
                
                current_community = membership[current_attr]
                s_k[i, current_community] += current_weight
    
    # Apply the 1/n_walks normalization
    for i in range(n_nodes):
        for k in range(n_communities):
            s_k[i, k] /= n_walks


@numba.njit
def calc_sample_s_total_weighted_cpu(
        W_total_ijk,
        s_total
    ):
    """
    Calculate total weighted capacity estimates from sampling.
    
    Implements: S_hat_total(i,L) = (1/n_walks) * sum_j sum_k W_total{i,j,k}
    
    Parameters
    ----------
    W_total_ijk : np.ndarray
        Weight tensor of shape (n_nodes, n_walks, walk_length-1)
    s_total : np.ndarray
        Output total capacity scores of shape (n_nodes,)
    """
    n_nodes, n_walks, walk_length_minus_1 = W_total_ijk.shape
    
    # Initialize
    for i in range(n_nodes):
        s_total[i] = 0.0
    
    # Aggregate total weighted scores
    for i in range(n_nodes):
        for j in range(n_walks):
            for k in range(walk_length_minus_1):
                current_weight = W_total_ijk[i, j, k]
                s_total[i] += current_weight
    
    # Apply the 1/n_walks normalization
    for i in range(n_nodes):
        s_total[i] /= n_walks


@numba.njit
def calc_sample_normalized_p_t_weighted_cpu(
        s_t,
        s_total,
        p_t_normalized
    ):
    """
    Calculate normalized attribute probabilities from weighted sampling.
    
    Implements: p_hat(t | i, L) = S_hat_t(i, L) / S_hat_total(i, L)
    
    Parameters
    ----------
    s_t : np.ndarray
        Weighted attribute scores of shape (n_nodes, n_attr) (already normalized by n_walks)
    s_total : np.ndarray
        Total capacity scores of shape (n_nodes,) (already normalized by n_walks)
    p_t_normalized : np.ndarray
        Output normalized probabilities of shape (n_nodes, n_attr)
    """
    n_nodes, n_attr = s_t.shape
    
    for i in range(n_nodes):
        for t in range(n_attr):
            if s_total[i] > 0:
                p_t_normalized[i, t] = s_t[i, t] / s_total[i]
            else:
                p_t_normalized[i, t] = 0.0


@numba.njit 
def calc_sample_normalized_p_k_weighted_cpu(
        s_k,
        s_total,
        p_k_normalized
    ):
    """
    Calculate normalized alliance probabilities from weighted sampling.
    
    Implements: p_hat(k | i, L) = S_hat_k(i, L) / S_hat_total(i, L)
    
    Parameters
    ----------
    s_k : np.ndarray
        Weighted alliance scores of shape (n_nodes, n_communities) (already normalized by n_walks)
    s_total : np.ndarray
        Total capacity scores of shape (n_nodes,) (already normalized by n_walks)
    p_k_normalized : np.ndarray
        Output normalized probabilities of shape (n_nodes, n_communities)
    """
    n_nodes, n_communities = s_k.shape
    
    for i in range(n_nodes):
        for k in range(n_communities):
            if s_total[i] > 0:
                p_k_normalized[i, k] = s_k[i, k] / s_total[i]
            else:
                p_k_normalized[i, k] = 0.0


@numba.njit
def calc_sample_mpc_weighted_cpu(
        membership,
        p_t_normalized, 
        p_k_normalized,
        mpc
    ):
    """
    Calculate MPC from normalized weighted sampling probabilities.
    
    Implements: w_hat_{t,i} = sum_k x_{t,k} * p_hat(k|i,L) * p_hat(t|i,L)
    
    Parameters
    ----------
    membership : np.ndarray
        Alliance membership of shape (n_attr,)
    p_t_normalized : np.ndarray
        Normalized attribute probabilities of shape (n_nodes, n_attr)
    p_k_normalized : np.ndarray
        Normalized alliance probabilities of shape (n_nodes, n_communities)
    mpc : np.ndarray
        Output MPC array of shape (n_attr,)
    """
    n_nodes, n_attr = p_t_normalized.shape
    
    for t in range(n_attr):
        mpc[t] = 0.0
        k = membership[t]
        for i in range(n_nodes):
            mpc[t] += p_t_normalized[i, t] * p_k_normalized[i, k]
