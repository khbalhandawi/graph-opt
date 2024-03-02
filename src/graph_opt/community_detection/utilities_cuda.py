

import math
from ..utilities_cuda import replace_values, cuda, DeviceNDArray

@cuda.jit(device=True)
def calc_hhi(
        i,j,
        n_edges,
        n_nodes,
        n_communities,
        membership_pairs,
        market_shares_nodal,
        market_shares_pairs,
        hhi_pairs
    ):
    for k in range(n_edges):
        for n in range(n_nodes):
            market_shares_pairs[i,j,k,membership_pairs[i,j,n]] += market_shares_nodal[k,n]
        
    for k in range(n_edges):
        for n in range(n_communities):
            hhi_pairs[i,j,k] += market_shares_pairs[i,j,k,n]**2

@cuda.jit(device=True)
def calc_ami(
        i,j,
        m_edges,
        n_nodes,
        adj,
        membership_pairs,
        edge_nodes,
        edge_weights,
        node_weight_individual,
        weights_pairs,
        score_pairs
    ):
    for n in range(n_nodes):
        score_pairs[i,j,n] += node_weight_individual[n]
        for k1 in range(m_edges):
            if edge_nodes[k1] == n:
                for k2 in range(m_edges):
                    if adj[k2,0] == adj[k1,1] and membership_pairs[i,j,n] == membership_pairs[i,j,edge_nodes[k2]]: 
                        # if destination edge is connected to current edge
                        weights_pairs[i,j,n] += edge_weights[k2]
                        score_pairs[i,j,n] += edge_weights[k2]
        score_pairs[i,j,n] = 0.0 if score_pairs[i,j,n] == 0.0 else math.log(score_pairs[i,j,n])

@cuda.jit(device=True)
def merge_score(
    i,j,
    i_global,
    j_global,
    n_communities:int,
    n_edges:int,
    n_nodes:int,
    m_edges:int,
    communities:DeviceNDArray,
    membership:DeviceNDArray,
    market_shares:DeviceNDArray,
    market_shares_nodal:DeviceNDArray,
    adj:DeviceNDArray,
    edge_nodes:DeviceNDArray,
    edge_weights:DeviceNDArray,
    edge_HHI:DeviceNDArray,
    node_weight:DeviceNDArray,
    node_weight_individual:DeviceNDArray,
    node_score:DeviceNDArray,
    membership_pairs:DeviceNDArray,
    market_shares_pairs:DeviceNDArray,
    hhi_pairs:DeviceNDArray,
    weights_pairs:DeviceNDArray,
    score_pairs:DeviceNDArray
):
    community1 = communities[i_global]
    community2 = communities[j_global]
    if community1 == community2:

        for k in range(n_nodes):
            membership_pairs[i,j,k] = membership[k]
            weights_pairs[i,j,k] = node_weight[k]
            score_pairs[i,j,k] = node_score[k]

        for k in range(n_edges):
            hhi_pairs[i,j,k] = edge_HHI[k]
            for l in range(n_communities):
                market_shares_pairs[i,j,k,l] = market_shares[k,l]

    else:
        for k in range(n_nodes):
            cur_member = membership[k] if membership[k] != community1 else community2
            membership_pairs[i,j,k] = cur_member

        calc_hhi(
            i,j,
            n_edges,
            n_nodes,
            n_communities,
            membership_pairs,
            market_shares_nodal,
            market_shares_pairs,
            hhi_pairs
        )

        calc_ami(
            i,j,
            m_edges,
            n_nodes,
            adj,
            membership_pairs,
            edge_nodes,
            edge_weights,
            node_weight_individual,
            weights_pairs,
            score_pairs
        )

@cuda.jit()
def merge_score_kernel(
        start_idx_i: int,
        start_idx_j: int,
        end_idx_i: int,
        end_idx_j: int,
        n_communities:int,
        n_edges:int,
        n_nodes:int,
        m_edges:int,
        communities:DeviceNDArray,
        membership:DeviceNDArray,
        market_shares:DeviceNDArray,
        market_shares_nodal:DeviceNDArray,
        adj:DeviceNDArray,
        edge_nodes:DeviceNDArray,
        edge_weights:DeviceNDArray,
        edge_HHI:DeviceNDArray,
        node_weight:DeviceNDArray,
        node_weight_individual:DeviceNDArray,
        node_score:DeviceNDArray,
        membership_pairs:DeviceNDArray,
        market_shares_pairs:DeviceNDArray,
        hhi_pairs:DeviceNDArray,
        weights_pairs:DeviceNDArray,
        score_pairs:DeviceNDArray
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

            merge_score(
                i,j,
                i_global,
                j_global,
                n_communities,
                n_edges,
                n_nodes,
                m_edges,
                communities,
                membership,
                market_shares,
                market_shares_nodal,
                adj,
                edge_nodes,
                edge_weights,
                edge_HHI,
                node_weight,
                node_weight_individual,
                node_score,
                membership_pairs,
                market_shares_pairs,
                hhi_pairs,
                weights_pairs,
                score_pairs
            )