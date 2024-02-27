

import math
from ..utilities_cuda import replace_values, cuda, DeviceNDArray

@cuda.jit(device=True)
def calc_hhi(
        i,j,
        n_edges,
        n_nodes,
        n_communities,
        membership,
        market_shares_nodal,
        market_shares_pairs,
        hhi_pairs
    ):
    for k in range(n_edges):
        for n in range(n_nodes):
            market_shares_pairs[i,j,k,membership[n]] += market_shares_nodal[k,n]
        
    for k in range(n_edges):
        for n in range(n_communities):
            hhi_pairs[i,j,k] += market_shares_pairs[i,j,k,n]**2

@cuda.jit(device=True)
def calc_ami(
        i,j,
        m_edges,
        n_nodes,
        adj,
        membership,
        edge_nodes,
        edge_weights,
        node_weight_individual,
        weights_pairs,
        score_pairs
    ):
    for n in range(n_nodes):
        score_pairs[i,j,n] += node_weight_individual[n]
        if membership[n] in [i,j]:
            for k1 in range(m_edges):
                if edge_nodes[k1] == n:
                    for k2 in range(m_edges):
                        if adj[k2,0] == adj[k1,1] and membership[edge_nodes[k2]] in [i,j]: 
                            # if destination edge is connected to current edge
                            weights_pairs[i,j,n] += edge_weights[k2]
                            score_pairs[i,j,n] += edge_weights[k2]
        score_pairs[i,j,n] = 0.0 if score_pairs[i,j,n] == 0.0 else math.log(score_pairs[i,j,n])


@cuda.jit()
def merge_score(
        n_communities:int,
        n_edges:int,
        n_nodes:int,
        m_edges:int,
        communities:DeviceNDArray,
        membership:DeviceNDArray,
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

    # Grid stride loop  
    for i in range(x, n_communities, dx):  
        for j in range(y, n_communities, dy):  
            
            if i == 0 and j == 1:

                community1 = communities[i]
                community2 = communities[j]
                if community1 == community2:
                    membership_pairs[i,j,:] = membership
                    market_shares_pairs[i,j,:,:] = market_shares_nodal
                    hhi_pairs[i,j,:] = edge_HHI
                    weights_pairs[i,j,:] = node_weight
                    score_pairs[i,j,:] = node_score
                else:
                    membership_pairs[i,j,:] = membership
                    replace_values(membership_pairs[i,j,:],community1,community2)

                    calc_hhi(
                        i,j,
                        n_edges,
                        n_nodes,
                        n_communities,
                        membership_pairs[i,j,:],
                        market_shares_nodal,
                        market_shares_pairs,
                        hhi_pairs
                    )

                    calc_ami(
                        i,j,
                        m_edges,
                        n_nodes,
                        adj,
                        membership,
                        edge_nodes,
                        edge_weights,
                        node_weight_individual,
                        weights_pairs,
                        score_pairs
                    )