

import math
from ..utilities_cuda import replace_values, cuda, DeviceNDArray

@cuda.jit()
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

@cuda.jit()
def compute_market_share(
        start_idx_i:int,
        start_idx_j:int,
        end_idx_i:int,
        end_idx_j:int,
        n_communities:int,
        n_edges:int,
        n_nodes:int,
        communities:DeviceNDArray,
        market_shares,
        market_shares_nodal:DeviceNDArray,
        edge_HHI,
        membership_pairs:DeviceNDArray,
        market_shares_pairs:DeviceNDArray,
        hhi_pairs
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
                    for i_comm in range(n_communities):
                        market_shares_pairs[i,j,i_edge,i_comm] = market_shares[i_edge,i_comm]
            else:
                for k in range(n_nodes):
                    for i_edge in range(n_edges):
                        market_shares_pairs[i,j,i_edge,membership_pairs[i,j,k]] += market_shares_nodal[i_edge,k]

@cuda.jit()
def calc_hhi(
        start_idx_i:int,
        start_idx_j:int,
        end_idx_i:int,
        end_idx_j:int,
        n_edges,
        n_communities,
        communities,
        market_shares_pairs,
        hhi_pairs
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
            for k in range(z, n_communities, dz):

                # Adjust indices for the current batch
                i_global = i + start_idx_i
                j_global = j + start_idx_j

                community1 = communities[i_global]
                community2 = communities[j_global]
                if community1 == community2:
                    pass
                else:
                    for i_edge in range(n_edges):
                        value = market_shares_pairs[i, j, i_edge, k] ** 2
                        cuda.atomic.add(hhi_pairs, (i, j, i_edge), value)
                        # hhi_pairs[i,j,i_edge] += market_shares_pairs[i,j,i_edge,k]**2

@cuda.jit()
def calc_ami(
        start_idx_i:int,
        start_idx_j:int,
        end_idx_i:int,
        end_idx_j:int,
        n_nodes,
        m_edges,
        communities,
        adj,
        membership_pairs,
        edge_nodes,
        edge_weights,
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
                    for k1 in range(m_edges):
                        if edge_nodes[k1] == k:
                            adj_k1_1 = adj[k1, 1]
                            edge_weight_k1 = edge_weights[k1]
                            for k2 in range(m_edges):
                                if adj[k2, 0] == adj_k1_1 and member_k == membership_pairs[i, j, edge_nodes[k2]]:
                                    edge_weight_k2 = edge_weights[k2]
                                    weights_pairs[i, j, k] += edge_weight_k2
                                    score_pairs[i, j, k] += edge_weight_k2