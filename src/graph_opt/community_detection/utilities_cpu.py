import numpy as np
from typing import Optional
from random import Random
import numba  

from ..random_walk.utilities import naive_weighted_choices

# Sampling functions
def edge_sampler_cpu(adj3: np.ndarray, samples: np.ndarray, rs: np.random.RandomState) -> np.ndarray:
    # Initialize samples array

    n_unique_edges, n_samples = samples.shape

    # Loop through each unique edge and each sample
    for i in range(n_unique_edges):
        for j in range(n_samples):
            # Extract the weights for the edge
            weights = adj3[i, :]

            # Compute the cumulative sum of the weights
            cumsum = np.cumsum(weights)

            # Generate a random float in the range [0, cumsum[-1])
            threshold = rs.uniform(0, cumsum[-1])

            # Find the index where the cumulative sum exceeds the threshold
            chosen = np.searchsorted(cumsum, threshold)

            # Assign the chosen index to samples
            samples[i, j] = chosen

    return samples

def spatial_random_walk(
        adj: np.ndarray,
        adj_i: np.ndarray,
        V: np.ndarray,
        walks_edges: np.ndarray,
        rs: Random,
    ):
    """
    Perform `n` random walks of length `length` starting from root nodes `nodes`.

    Parameters
    ----------
    adj : np.ndarray
        dense adjacency matrix of shape n_nodes x n_nodes
    adj_i : np.ndarray
        dense matrix of shape n_nodes x n_nodes that contains edge indices
    V : np.ndarray
        array of shape n_nodes x n_samples x walk_length
        to store the generated walks
    walks_edges : np.ndarray
        array of shape n_nodes x n_samples x walk_length-1
        to store the edge indices of the generated walks
    rs : Random
        Random number generator used to generator the samples
    """

    n_nodes, n_samples, walk_length = V.shape

    for i in range(n_nodes):  # iterate over root nodes
        for j in range(n_samples):  # generate n walks per root node
            # the walk starts at the root
            current_node = i
            V[i,j,0] = current_node

            for k in range(walk_length - 1):
                # select one of the neighbors using the
                # appropriate transition probabilities
                weights = adj[current_node,:]
                edges = adj_i[current_node,:]

                if np.sum(weights) == 0:
                    break

                choice = naive_weighted_choices(rs, weights)
                if choice is None:
                    break

                current_node = choice
                current_edge = edges[choice]
                V[current_node,j,k+1] = choice
                walks_edges[i,j,k] = current_edge

def attr_sampler_cpu(
        walks_edges: np.ndarray,
        adj3: np.ndarray,
        T: np.ndarray,
        W: np.ndarray,
        rs: np.random.RandomState,
    ):
    """
    Sample attributes for each node in the random walks.

    Parameters
    ----------
    walks_edges : np.ndarray
        array of shape n_nodes x n_samples x walk_length-1
        that contains the edge indices of the generated walks
    adj3 : np.ndarray
        sparse matrix of shape n_unique_edges x n_attr
        that contains the weights for each edge attribute
    T : np.ndarray
        array of shape n_nodes x n_samples x walk_length
        to store the sampled attributes
    W : np.ndarray
        array of shape n_nodes x n_samples x walk_length
        to store the conditional probabilities of the sampled attributes
    rs : np.random.RandomState
        Random number generator used to generator the samples
    """

    n_nodes, n_samples, walk_length_1 = T.shape

    # Iterate over each step in the walk
    for k in range(walk_length_1):
        # Flatten the relevant slice for simplicity
        current_edges = walks_edges[:, :, k]
        valid_mask = current_edges >= 0

        # We need to handle valid edges only
        valid_edges = current_edges[valid_mask].astype(int)

        if len(valid_edges) > 0:
            # Extract weights from adj3 based on valid_edges for current step k
            weights = adj3[valid_edges]  # Shape: (num_valid_edges, n_attr)

            # Normalize weights to get probabilities
            weights_cumsum = np.cumsum(weights, axis=1)
            total_weights = weights_cumsum[:, -1, None]  # Ensure keepdims
            probs = weights_cumsum / total_weights

            # Sample according to the cumulative probability distribution
            random_samples = rs.rand(*probs.shape[:-1], 1)
            chosen_indices = np.argmax(probs >= random_samples, axis=1)

            # Assign the chosen indices to the sample array
            T[valid_mask, k] = chosen_indices

            # Compute W values using the selected attributes and adj values
            W[valid_mask, k] = adj3[current_edges[valid_mask], chosen_indices]

# Exact evaluation functions
@numba.njit
def edge_postprocess_cpu(
        edges_cp,
        membership,
        edges_communities,
    ):

    n_unique_edges, n_attr = edges_cp.shape

    for i in range(n_unique_edges):
        for j in range(n_attr):

            cp = edges_cp[i, j]
            k = membership[j]
            edges_communities[i,k] += cp

@numba.njit
def calc_hhi_cpu(
        edges_communities,
        edges_hhi
    ):

    n_unique_edges, n_communities = edges_communities.shape

    # Grid stride loop
    for i in range(n_unique_edges):
        for j in range(n_communities):
            
            cp = edges_communities[i, j]
            edges_hhi[i] += cp**2

@numba.njit
def calc_w_it_cpu(
        adj,
        adj3,
        w_it,
    ):

    n_unique_edges, n_attr = adj3.shape
    n_nodes, n_attr = w_it.shape

    # Grid stride loop
    for i in range(n_unique_edges):
        for j in range(n_attr):
            source = adj[i,0]
            w_it[source, j] += adj3[i, j]

@numba.njit
def calc_w_ik_cpu(
        membership,
        w_it,
        w_ik,
    ):

    n_nodes, n_attr = w_it.shape

    # Grid stride loop
    for i in range(n_nodes):
        for j in range(n_attr):
            k = membership[j]
            w_ik[i, k] += w_it[i, j]

@numba.njit
def calc_mpc_cpu(
        membership,
        w_it,
        w_ik,
        w_i,
        mpc,
    ):

    n_nodes, n_attr = w_it.shape
    n_nodes, n_communities = w_ik.shape

    # Grid stride loop
    for t in range(n_attr):
        for i in range(n_nodes):
            if w_i[i] > 0.0:
                k = membership[t]
                mpc[t] += w_it[i, t] * w_ik[i, k] / (w_i[i] ** 2)

# Sampling evaluation functions
@numba.njit
def calc_sample_w_i_cpu(T_ijk, w_i):
    n_nodes, n_samples, walk_length = T_ijk.shape

    # Iterate over each sample and walk step
    for i in range(n_nodes):
        for j in range(n_samples):
            for k in range(walk_length):  # Start from 1 since 0 is the source node
                current_attr = T_ijk[i, j, k]

                # Check if the current node is valid
                if current_attr < 0:
                    break  # Stop processing this walk if an invalid node is encountered

                w_i[i] += 1

@numba.njit
def calc_sample_w_it_cpu(T_ijk, w_it):
    n_nodes, n_samples, walk_length = T_ijk.shape
    n_nodes, n_attr = w_it.shape

    # Iterate over each sample and walk step
    for i in range(n_nodes):
        for j in range(n_samples):
            for t in range(n_attr):
                
                for k in range(walk_length):  # Start from 1 since 0 is the source node
                    current_attr = T_ijk[i, j, k]

                    # Check if the current node is valid
                    if current_attr < 0:
                        break  # Stop processing this walk if an invalid node is encountered

                    # Check if current node matches the source node condition
                    if t == current_attr:
                        w_it[i, t] += 1

def edge_sample_postprocess_cpu(samples, membership, n_communities, counts):
    n_unique_edges, n_samples_edges = samples.shape

    # Expand samples to index membership
    expanded_samples = membership[samples]

    # Use np.add.at for unbuffered in-place addition across the indices
    np.add.at(counts, (np.arange(n_unique_edges)[:, None], expanded_samples), 1)

@numba.njit
def calc_sample_w_ik_cpu(T_ijk, membership, w_ik):
    n_nodes, n_samples, walk_length = T_ijk.shape
    n_nodes, n_communities = w_ik.shape

    # Iterate over each sample and walk step
    for i in range(n_nodes):
        for j in range(n_samples):
            for k in range(n_communities):
                
                for l in range(walk_length):  # Start from 1 since 0 is the source node
                    current_attr = T_ijk[i, j, l]

                    # Check if the current node is valid
                    if current_attr < 0:
                        break  # Stop processing this walk if an invalid node is encountered
                    
                    current_community = membership[current_attr]
                    # Check if current node matches the source node condition
                    if k == current_community:
                        w_ik[i, k] += 1

@numba.njit
def calc_sample_mpc_cpu(w_it, w_ik, w_i, membership, mpc):
    n_nodes, n_attr = w_it.shape

    # Iterate over each sample and walk step
    for i in range(n_nodes):
        for j in range(n_attr):
            attr_community = membership[j]
            if w_i[i] > 0:
                mpc[i,j] = (w_ik[i,attr_community] * w_it[i,j]) / (w_i[i]*w_i[i])

# Optimization functions
def calc_membership_pairs_cpu(membership: np.ndarray, membership_pairs: np.ndarray, n_communities: int):
    """
    Generate all possible membership merge pairs.

    Parameters
    ----------
    membership : np.ndarray
        Array of shape (n_attr,) that contains the current community assignments

    membership_pairs : np.ndarray
        Array of shape (n_communities, n_communities, n_attr) that contains all possible
        pairs of community merges
    """

    communities = np.array(range(n_communities))

    n_attr = membership.shape[0]

    # Broadcast communities to create a 3D array where each layer is the comparison of community1 and community2
    community1 = communities[:, None, None]  # Shape (n, 1, 1)
    community2 = communities[None, :, None]  # Shape (1, n, 1)

    # Broadcast membership to match the third dimension
    membership_expanded = membership[None, None, :]  # Shape (1, 1, n_attr)
    membership_expanded = np.broadcast_to(membership_expanded, (n_communities, n_communities, n_attr))

    # Condition where community1 == community2
    mask_equal = community1 == community2
    mask_equal = np.broadcast_to(mask_equal, (n_communities, n_communities, n_attr))  # Make sure mask has the same shape as membership_pairs

    # Condition where community1 == community2
    membership_pairs[mask_equal] = membership_expanded[mask_equal]

    # Condition where community1 != community2
    mask_not_equal = community1 != community2
    mask_not_equal = np.broadcast_to(mask_not_equal, (n_communities, n_communities, n_attr))
    # Correct membership for the case where community1 != community2
    # Using np.where to select membership values or alternative community
    alternative_community = np.where(membership_expanded == community1, community2, membership_expanded)
    membership_pairs[mask_not_equal] = alternative_community[mask_not_equal]

    return membership_pairs