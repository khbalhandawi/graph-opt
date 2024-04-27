import numpy as np
from typing import Optional
from random import Random

from ..random_walk.utilities import naive_weighted_choices

def spatial_random_walk(
        rs: Random,
        adj: np.ndarray,
        adj_i: np.ndarray,
        n_nodes: int,
        n_samples: int,
        walk_length: int,
    ):
    """
    Perform `n` random walks of length `length` starting from root nodes `nodes`.

    Parameters
    ----------
    rs : Random
        Random number generator used to generator the samples
    adj : np.ndarray
        dense adjacency matrix of shape n_nodes x n_nodes
    adj_i : np.ndarray
        dense matrix of shape n_nodes x n_nodes that contains edge indices
    n_nodes : int
        total number of nodes in the graph
    n_samples : int
        numpber of samples per root node
    walk_length : int
        length of the random walk
    """

    walks = np.zeros((n_nodes, n_samples, walk_length), dtype=np.int32)
    walks_edges = np.full((n_nodes, n_samples, walk_length-1), -10000, dtype=np.int32)

    for i in range(n_nodes):  # iterate over root nodes
        for j in range(n_samples):  # generate n walks per root node
            # the walk starts at the root
            current_node = i
            walks[i,j,0] = current_node

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
                walks[current_node,j,k+1] = choice
                walks_edges[i,j,k] = current_edge

    return walks, walks_edges

def attr_sampler_cpu(n_samples, walk_length, walks_edges, adj, adj3, n_nodes):
    # Initialize output arrays
    attr_sample = np.full((n_nodes, n_samples, walk_length-1), -10000, np.int32)
    attr_cp = np.zeros((n_nodes, n_samples, walk_length-1))

    # Iterate over each step in the walk
    for k in range(walk_length-1):
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
            random_samples = np.random.rand(*probs.shape[:-1], 1)
            chosen_indices = np.argmax(probs >= random_samples, axis=1)

            # Assign the chosen indices to the sample array
            attr_sample[valid_mask, k] = chosen_indices

            # Compute attr_cp values using the selected attributes and adj values
            attr_cp[valid_mask, k] = adj[current_edges[valid_mask]] * (weights[np.arange(len(valid_edges)), chosen_indices] / 100.0)

    return attr_sample, attr_cp

def attr_source_postprocess_cpu(n_nodes, n_attr, attr_sample):

    source_attr_counts = np.zeros((n_nodes, n_attr), dtype=np.int32)  # Assuming 100 possible attributes

    # Iterate through the first attribute sample for each sample set
    # Extract the first attribute sample from each set
    first_attr_samples = attr_sample[:, :, 0]

    # Filter out invalid samples (NaN or negative)
    mask = first_attr_samples >= 0  # Create a mask for valid samples

    # Update counts for each node based on the first valid attribute sample
    for i in range(n_nodes):
        unique, counts = np.unique(first_attr_samples[i][mask[i]], return_counts=True)
        unique = unique.astype(int)  # Ensure indices are integers for indexing
        source_attr_counts[i, unique] += counts

    return source_attr_counts

def edge_sampler_cpu(n_samples, n_unique_edges, adj3):
    # Initialize samples array
    samples = np.zeros((n_unique_edges, n_samples), dtype=int)

    # Loop through each unique edge and each sample
    for i in range(n_unique_edges):
        for j in range(n_samples):
            # Extract the weights for the edge
            weights = adj3[i, :]

            # Compute the cumulative sum of the weights
            cumsum = np.cumsum(weights)

            # Generate a random float in the range [0, cumsum[-1])
            threshold = np.random.uniform(0, cumsum[-1])

            # Find the index where the cumulative sum exceeds the threshold
            chosen = np.searchsorted(cumsum, threshold)

            # Assign the chosen index to samples
            samples[i, j] = chosen

    return samples

def edge_postprocess_cpu(ci, cj, samples, membership, n_communities):
    n_unique_edges, n_samples_edges = samples.shape
    n_attr = samples.shape[-1]

    counts = np.zeros((n_unique_edges, n_communities), dtype=int)

    # Expand samples to index membership
    expanded_samples = membership[ci, cj, samples]

    # Use np.add.at for unbuffered in-place addition across the indices
    np.add.at(counts, (np.arange(n_unique_edges)[:, None], expanded_samples), 1)

    return counts

def attr_postprocess_cpu(ci, cj, samples, samples_weights, source_counts, membership, n_attr):
    n_nodes, n_samples, walk_length = samples.shape
    cum_weight = np.zeros((n_nodes, n_attr), dtype=np.float32)

    # Iterate over each sample and walk step
    for i in range(n_nodes):
        for j in range(n_samples):
            source_node = samples[i, j, 0]

            # Check if the source node is valid
            if not np.isnan(source_node) and source_node >= 0:
                source_node = int(source_node)
                source_c = membership[ci, cj, source_node]
                source_count = source_counts[i, source_node]

                # Only proceed if the source node has a valid count
                if source_count > 0:
                    for k in range(walk_length - 1):  # Start from 1 since 0 is the source node
                        current_node = samples[i, j, k]

                        # Check if the current node is valid
                        if np.isnan(current_node) or current_node < 0:
                            break  # Stop processing this walk if an invalid node is encountered

                        current_node = int(current_node)
                        current_c = membership[ci, cj, current_node]
                        current_weight = samples_weights[i, j, k]

                        # Check if current node matches the source node condition
                        if current_c == source_c:
                            # Update cumulative weight
                            cum_weight[i, source_node] += current_weight
                        else:
                            break  # Optional: Stop if the membership condition isn't met

    return cum_weight

def set_membership_cpu(communities, membership, n_attr):
    n = len(communities)

    # Initialize the membership_pairs array
    membership_pairs = np.zeros((n, n, n_attr), dtype=np.int32)

    # Broadcast communities to create a 3D array where each layer is the comparison of community1 and community2
    community1 = communities[:, None, None]  # Shape (n, 1, 1)
    community2 = communities[None, :, None]  # Shape (1, n, 1)

    # Broadcast membership to match the third dimension
    membership_expanded = membership[None, None, :]  # Shape (1, 1, n_attr)
    membership_expanded = np.broadcast_to(membership_expanded, (n, n, n_attr))

    # Condition where community1 == community2
    mask_equal = community1 == community2
    mask_equal = np.broadcast_to(mask_equal, (n, n, n_attr))  # Make sure mask has the same shape as membership_pairs

    # Condition where community1 == community2
    membership_pairs[mask_equal] = membership_expanded[mask_equal]

    # Condition where community1 != community2
    mask_not_equal = community1 != community2
    mask_not_equal = np.broadcast_to(mask_not_equal, (n, n, n_attr))
    # Correct membership for the case where community1 != community2
    # Using np.where to select membership values or alternative community
    alternative_community = np.where(membership_expanded == community1, community2, membership_expanded)
    membership_pairs[mask_not_equal] = alternative_community[mask_not_equal]

    return membership_pairs