from collections import defaultdict
import networkx as nx
from networkx import MultiDiGraph
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import random
import functools
import math
from pdb import set_trace
from scipy.sparse import spmatrix
from typing import Callable, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from .utilities import resort_communities
from ..utilities import _default_if_none, dict2array, cpu_gpu_dispatcher, cuda_blocks

from typing import Dict, List, Union, Tuple, Optional, NewType
from ..random_settings import random_state

type_share = NewType('type_share', Dict[int,float])
type_shares = NewType('type_shares', Dict[Tuple[int, int], type_share])
type_scores = NewType('type_scores', Dict[Tuple[int, int], Dict[int, float]])

def multiobjective(f1: Union[float,np.ndarray], f2: Union[float,np.ndarray], weights: Optional[Tuple[float,float]]=(0.5,0.5)):
    return weights[0]*f1 + weights[1]*f2


class AllianceBuilder():

    def __init__(
            self,
            MG: MultiDiGraph,
            communities_dict: Dict[str,List[str]] = None,
            seed: int = None,
            use_gpu: Optional[bool] = False,
        ) -> None:

        self._communities_dict = {}
        self.G: MultiDiGraph = deepcopy(MG)
        self.node_list = []
        self.edge_list = []
        self.edge_unique_list = []
        self.node_spatial_list = []
        self.node_spatial_weights = []
        self.node_key2i = {}
        self.node_i2key = {}
        self.node_key2w = defaultdict(float)
        self.node_spatial_key2i = {}
        self.node_spatial_key2w = {}
        self.node_spatial_i2key = {}
        self.edge_pairkey2ui = {}
        self.edge_pair2w = defaultdict(float)
        self.edge_pair2ui = {}
        self.edge_ui2pair = {}

        self._HHI = defaultdict(float)
        self._cond_probs_nodes = defaultdict(lambda: defaultdict(float))
        self._cond_probs_community = defaultdict(lambda: defaultdict(float))
        self._cond_probs_community_sparse = []
        self._adj3 = np.empty(0)
        self._adj_dense_index = np.empty(0)
        self._adj = np.empty(0)

        self._airline_score = {}
        self._weights_dict = defaultdict(list)
        self._weights_individual_dict = defaultdict(list)

        self.edge_node2keypair_list = defaultdict(list)
        self.edge_u2pairkey_list = defaultdict(list)
        self.edge_u2pair_list = defaultdict(list)
        self.edge_pairkey2atrr = {}
        self.edge_pair2pairkey_list = defaultdict(list)
        self.edge_pairkey2i = {}

        self.communities_history = []
        self.f_history = []
        self.multi_history = []

        self.seed = seed
        self.random_state: Tuple[random.Random,np.random.RandomState] = None

        self.use_gpu = use_gpu

        self._adj_dense = nx.to_numpy_array(self.G, weight="weight")

        self._parse_graph()
        self._initialize_communities(communities_dict)
        self._initialize_HHI()
        self._initialize_MAI()

        # Populate the adjacency matrix
        edge_nodes = []
        edge_weights = []
        edges_neighbour = []
        edge_unique_index = []
        processed_edges = set()

        adj = np.zeros((self.G.number_of_edges(), 2), dtype=np.int32)
        for i, (u, v, key, data) in enumerate(self.G.edges(data=True, keys=True)):
            adj[i,:] = np.array([self.node_spatial_key2i[u],self.node_spatial_key2i[v]])
            edge_nodes.append(self.node_key2i[data["edge_type"]])
            edge_weights.append(data["weight"])
            edge_unique_index.append(self.edge_pairkey2ui[(u, v, key)])
            if (v, key) not in processed_edges:
                for u_link, v_link, key_link in self.edge_u2pairkey_list[v]:
                    edges_neighbour.append((self.edge_pairkey2i[(u, v, key)], self.edge_pairkey2i[(u_link, v_link, key_link)]))
                processed_edges.add((v, key))

        # dense adjacency        
        self._edge_nodes = np.array(edge_nodes,dtype=np.int32)
        self._edge_weights = np.array(edge_weights,dtype=np.float32)
        self._edges_neighbour = np.array(edges_neighbour,dtype=np.int32)
        self._edge_unique_index = np.array(edge_unique_index,dtype=np.int32)
        self.m_edge_pairs = self._edges_neighbour.shape[0]

        self._market_shares_sparse_values = []
        self._market_shares_sparse_nodes = []
        for n,v in self._cond_probs_community_sparse:
            self._market_shares_sparse_values.append(v)
            self._market_shares_sparse_nodes.append(n)

        self._market_shares_sparse_values = np.array(self._market_shares_sparse_values, dtype=np.float32)
        self._market_shares_sparse_nodes = np.array(self._market_shares_sparse_nodes, dtype=np.int32)

        self.cuda = None

    @property
    def communities_dict(self)->Dict[str,int]:
        return {node: self._communities_dict[node] for node in self._communities_dict.keys()}

    @communities_dict.setter
    def communities_dict(self,value:Dict[str,int]):
        key, self._communities_dict, self._distinct_communities, self.n_communities = resort_communities(value)

    @property
    def market_shares_community(self):
        return {edge: self._cond_probs_community[edge].copy() for edge in self._cond_probs_community.keys()}

    @property
    def HHI(self):
        return {edge: self._HHI[edge] for edge in self._HHI.keys()}

    @property
    def weights_dict(self):
        return {node: self._weights_dict[node] for node in self._weights_dict.keys()}

    @property
    def weights_individual_dict(self):
        return {node: self._weights_individual_dict[node] for node in self._weights_individual_dict.keys()}


    @property
    def airline_score(self):
        return {node: self._airline_score[node] for node in self._airline_score.keys()}

    @airline_score.setter
    def airline_score(self, value):
        self._airline_score = value

    def _parse_graph(self):
        """
        Group edges by 'edge_type' and source node for faster lookup
        """

        # initialize nodes
        self.m_edges = self.G.number_of_edges()

        for i, (u, v, key, attr) in enumerate(self.G.edges(keys=True, data=True)):
            self.edge_node2keypair_list[attr['edge_type']].append((u, v, key))
            self.edge_u2pairkey_list[u].append((u, v, key))
            self.edge_pairkey2atrr[(u, v, key)] = attr
            self.edge_pairkey2i[(u, v, key)] = i
            self.edge_pair2pairkey_list[(u, v)].append((u, v, key))


        for i, u in enumerate(self.G.nodes()):
            node_weight = self.G.out_degree(nbunch=u,weight="weight")
            self.node_spatial_key2w[u] = node_weight
            self.node_spatial_key2i[u] = i
            self.node_spatial_key2i[u] = i
            self.node_spatial_i2key[i] = u
            self.node_spatial_list.append(u)
            self.node_spatial_weights.append(node_weight)

        self.n_nodes_spatial = len(self.node_spatial_list)

    def _initialize_communities(self, communities_dict: Optional[Dict[str,str]] = None):
        """
        Initialize each node to be its own community

        Parameters
        ----------
        node_list : List[str]
            _description_
        communities_dict : Dict[str,str]
            _description_
        """

        if communities_dict is None:

            edge_types = set()
            for _, _, data in self.G.edges(data=True):
                edge_type = data.get('edge_type', None)
                if edge_type:
                    edge_types.add(edge_type)

            node_list = sorted(edge_types)

            communities_dict = {}
            for i,node in enumerate(node_list):
                communities_dict[node] = i
                self.node_list.append(node)
                self.node_key2i[node] = i
                self.node_i2key[i] = node

            self.communities_dict = communities_dict
        else:
            self.communities_dict = communities_dict
            node_list = sorted(list(self._communities_dict.keys()))
            for i,node in enumerate(node_list):
                self.node_list.append(node)
                self.node_key2i[node] = i

        self.n_nodes = len(self.node_list)

    def _initialize_HHI(self):
        """
        Initializes the HHI attribute of each edge of the graph
        """
        for u, v, data in self.G.edges(data=True):
            # Accumulate the squares for each route
            self.edge_pair2w[(u, v)] += data['weight']
            self._cond_probs_nodes[(u, v)][data['edge_type']] += data['market_share_percent']
            self.node_key2w[data['edge_type']] += data['weight']
            self._cond_probs_community[(u, v)][self._communities_dict[data['edge_type']]] += data['market_share_percent']
            self._cond_probs_community_sparse.append([self._communities_dict[data['edge_type']],data['market_share_percent']])

        # Calculate the HHI for each edge
        for i, (edge, communities) in enumerate(self._cond_probs_community.items()):
            self._HHI[edge] = sum(market_share ** 2 for market_share in communities.values())

        self.edge_unique_list = list(self._cond_probs_nodes.keys())
        self.n_unique_edges = len(self.edge_unique_list)
        adj3 = np.zeros((self.n_unique_edges,self.n_nodes))
        adj = np.zeros((self.n_unique_edges))
        adj_dense_index = np.full((self.n_nodes_spatial, self.n_nodes_spatial), np.nan)
        for i, (edge, nodes) in enumerate(self._cond_probs_nodes.items()):
            self.edge_u2pair_list[edge[0]].append(edge)
            self.edge_pair2ui[edge] = i
            self.edge_ui2pair[i] = edge
            adj_dense_index[self.node_spatial_key2i[edge[0]],self.node_spatial_key2i[edge[1]]] = i
            adj[i] = self._adj_dense[self.node_spatial_key2i[edge[0]],self.node_spatial_key2i[edge[1]]]
            for node, value in nodes.items():
                adj3[i,self.node_key2i[node]] = value
            for u,v,key in self.edge_pair2pairkey_list[edge]:
                self.edge_pairkey2ui[(u,v,key)] = i

        self._adj3 = adj3
        self._adj = adj
        self._adj_dense_index = adj_dense_index

    def _initialize_MAI(self):
        """
        Initializes the HHI attribute of each edge of the graph
        """
        for node in self.node_list:
            target_community = self._communities_dict[node]

            # Get edges by 'edge_type' from the pre-grouped dictionary
            edges_node = self.edge_node2keypair_list[node]
            G_sub = self.G.edge_subgraph(edges_node)

            # Get 'to' nodes
            to_nodes = {v for _, v, _ in edges_node}

            # Calculate weights
            if len(to_nodes) > 0:
                edges_neighbors = set(sum((self.edge_u2pairkey_list[node] for node in to_nodes), []))
                weights = [self.edge_pairkey2atrr[(u,v,key)]['weight'] for u,v,key in edges_neighbors if self._communities_dict[self.edge_pairkey2atrr[(u,v,key)]['edge_type']] == target_community]
                weights_individual = [attr['weight'] for _, _, attr in G_sub.edges(data=True)]

                self._weights_dict[node] = sum(weights)
                self._weights_individual_dict[node] = sum(weights_individual)

                if np.sum(weights_individual) > 0:
                    self._airline_score[node] = np.log(self._weights_dict[node] + self._weights_individual_dict[node])

    def update_scores(self, community_to_merge, new_community):
        edges_to_update, nodes_to_update = self._update_communities(community_to_merge,new_community)
        self._update_HHI_scores(edges_to_update)
        self._update_scores(nodes_to_update)

    def _greedy_optimize_gpu(
            self,
            weights: Optional[Tuple[float,float]]=(0.5,0.5),
            max_iterations: Optional[int]=1000,
            batch_size: Optional[Union[int,Tuple[int,int]]] = 2,
            num_streams: Optional[int] = 1,
            threads_per_block: Optional[Union[int,Tuple[int,int,int]]] = 2,
            debug: bool = False
        ) -> Tuple[Dict[int,int], float]:

        # Calculate the initial mean HHI over all routes
        f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
        f2 = sum(self._airline_score.values()) / len(self._airline_score)
        f = multiobjective(f1,f2,weights)

        # log history
        self.communities_history.append(self.communities_dict)
        self.f_history.append(f)
        self.multi_history.append((f1,f2))

        # Initialize flag for improvement
        improvement = True

        # Loop while improvement is possible
        n_iterations = 0
        with tqdm(total=max_iterations, desc="Outer Loop") as pbar_outer:
            while improvement and n_iterations < max_iterations:

                improvement = False
                max_increase = 0
                best_pair = None

                # move iteration constants to memory
                communities_d = self.cuda.to_device(self._distinct_communities)
                membership_d = self.cuda.to_device(list((self._communities_dict.values())))
                edge_HHI_d = self.cuda.to_device(list(self._HHI.values()))
                node_weight_d = self.cuda.to_device(list(self._weights_dict.values()))
                node_score_d = self.cuda.to_device(list(self._weights_individual_dict.values()))

                # Define the batch size based on memory requirements and available GPU memory
                if isinstance(batch_size, int):
                    batch_size_i = batch_size
                    batch_size_j = batch_size
                else:
                    batch_size_i = batch_size[0]
                    batch_size_j = batch_size[1]

                if isinstance(threads_per_block, int):
                    TPB_i = threads_per_block
                    TPB_j = threads_per_block
                    TPB_k = threads_per_block
                else:
                    TPB_i = threads_per_block[0]
                    TPB_j = threads_per_block[1]
                    TPB_k = threads_per_block[2]

                # Calculate the number of batches needed to process all communities
                num_batches_i = math.ceil(self.n_communities / batch_size_i)
                num_batches_j = math.ceil(self.n_communities / batch_size_j)

                # Define the number of streams (can be adjusted based on the desired level of concurrency)
                num_streams = min(num_streams, num_batches_i, num_batches_j)

                # Create streams
                streams = [self.cuda.stream() for _ in range(num_streams)]

                # Initialize host arrays to store results (adjust the size based on the actual number of batches)
                best_pairs = np.zeros((num_batches_i, num_batches_j, 2), dtype=np.int32)
                hhi_pairs = np.zeros((num_batches_i, num_batches_j, self.n_unique_edges), dtype=np.float32)
                weights_pairs = np.zeros((num_batches_i, num_batches_j, self.n_nodes), dtype=np.float32)
                score_pairs = np.zeros((num_batches_i, num_batches_j, self.n_nodes), dtype=np.float32)
                membership_pairs = np.zeros((num_batches_i, num_batches_j, self.n_nodes), dtype=np.int32)
                
                # Batch processing with streams
                num_total_batches = num_batches_i * num_batches_j
                with tqdm(total=num_total_batches, desc="Batches processed") as pbar_inner:
                    for batch_index in range(num_total_batches):
                        batch_i = batch_index // num_batches_j
                        batch_j = batch_index % num_batches_j

                        start_idx_i = batch_i * batch_size_i
                        end_idx_i = min((batch_i + 1) * batch_size_i, self.n_communities)

                        start_idx_j = batch_j * batch_size_j
                        end_idx_j = min((batch_j + 1) * batch_size_j, self.n_communities)

                        # Select a stream for this batch
                        stream = streams[(batch_i * num_batches_j + batch_j) % num_streams]

                        # Initialize device arrays for the batch
                        membership_pairs_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_nodes), dtype=np.int8)
                        local_array_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_unique_edges), dtype=np.float32)
                        hhi_pairs_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_unique_edges), dtype=np.float32)
                        weights_pairs_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_nodes), dtype=np.float32)
                        score_pairs_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_nodes), dtype=np.float32)

                        # preprocess arrays on host
                        min_dim = min(score_pairs_h.shape) # Determine the length of the shorter dimension
                        diag_indices = np.arange(min_dim) # Create an index array for the diagonal elements
                        score_pairs_h[diag_indices, diag_indices, :] = self._node_weight_individual # Use advanced indexing to set the values

                        # move final arrays to device
                        membership_pairs_d = self.cuda.to_device(membership_pairs_h,stream=stream)
                        local_array_d = self.cuda.to_device(local_array_h,stream=stream)
                        hhi_pairs_d = self.cuda.to_device(hhi_pairs_h,stream=stream)
                        weights_pairs_d = self.cuda.to_device(weights_pairs_h,stream=stream)
                        score_pairs_d = self.cuda.to_device(score_pairs_h,stream=stream)

                        # Define the thread and block dimensions
                        threadsperblock = (TPB_i, TPB_j, TPB_k)
                        blockspergrid_x = math.ceil(batch_size_i / threadsperblock[0])
                        blockspergrid_y = math.ceil(batch_size_j / threadsperblock[1])
                        blockspergrid_z = math.ceil(self.n_nodes / threadsperblock[2])
                        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

                        self.set_batch_membership[blockspergrid,threadsperblock,stream](
                                start_idx_i,
                                start_idx_j,
                                end_idx_i,
                                end_idx_j,
                                self.n_nodes,
                                node_weight_d,
                                node_score_d,
                                communities_d,
                                membership_d,
                                membership_pairs_d,
                                weights_pairs_d,
                                score_pairs_d
                            )

                        stream.synchronize()
                        print('membership done')

                        # Define the thread and block dimensions
                        threadsperblock = (TPB_i, TPB_j)
                        blockspergrid_x = math.ceil(batch_size_i / threadsperblock[0])
                        blockspergrid_y = math.ceil(batch_size_j / threadsperblock[1])
                        blockspergrid = (blockspergrid_x, blockspergrid_y)

                        self.calc_hhi[blockspergrid,threadsperblock,stream](
                                start_idx_i,
                                start_idx_j,
                                end_idx_i,
                                end_idx_j,
                                self.n_unique_edges,
                                self.m_edges,
                                self.n_communities,
                                communities_d,
                                self._market_shares_sparse_nodes_d,
                                self._market_shares_sparse_values_d,
                                self._edge_unique_index_d,
                                edge_HHI_d,
                                membership_pairs_d,
                                hhi_pairs_d,
                                local_array_d
                            )
                        
                        # stream.synchronize()
                        # print('hhi done')

                        # Define the thread and block dimensions
                        threadsperblock = (TPB_i, TPB_j, TPB_k)
                        blockspergrid_x = math.ceil(batch_size_i / threadsperblock[0])
                        blockspergrid_y = math.ceil(batch_size_j / threadsperblock[1])
                        blockspergrid_z = math.ceil(self.n_nodes / threadsperblock[2])
                        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

                        self.calc_ami[blockspergrid,threadsperblock,stream](
                                start_idx_i,
                                start_idx_j,
                                end_idx_i,
                                end_idx_j,
                                self.n_nodes,
                                self.m_edge_pairs,
                                communities_d,
                                membership_pairs_d,
                                self._edge_nodes_d,
                                self._edge_weights_d,
                                self._edges_neighbour_d,
                                weights_pairs_d,
                                score_pairs_d
                            )

                        stream.synchronize()
                        print('ami done')

                        # Copy results from device to host for the current batch
                        hhi_pairs_h = hhi_pairs_d.copy_to_host(stream=stream)
                        weights_pairs_h = weights_pairs_d.copy_to_host(stream=stream)
                        score_pairs_h = score_pairs_d.copy_to_host(stream=stream)   
                        score_pairs_h = np.where(score_pairs_h > 0.0, np.log(score_pairs_h), 0.0)
                        membership_pairs_h = membership_pairs_d.copy_to_host(stream=stream)
                        
                        # free memory
                        membership_pairs_d = None
                        local_array_d = None
                        hhi_pairs_d = None
                        weights_pairs_d = None
                        score_pairs_d = None

                        f1_pairs = -np.mean(hhi_pairs_h,axis=2) / 100
                        f2_pairs = np.mean(score_pairs_h,axis=2)
                        f_pairs = multiobjective(f1_pairs,f2_pairs,weights=weights)

                        # Create a mask where i + start_idx_i > j + start_idx_j
                        mask = np.fromfunction(lambda i, j: i + start_idx_i < j + start_idx_j, (end_idx_i - start_idx_i, end_idx_j - start_idx_j))
                        f_pairs = np.where(mask, f_pairs, -np.inf)
                        hhi_pairs_h = np.where(mask[:,:,np.newaxis], hhi_pairs_h, np.sign(weights[0])*10000.0)
                        weights_pairs_h = np.where(mask[:,:,np.newaxis], weights_pairs_h, 0.0)
                        score_pairs_h = np.where(mask[:,:,np.newaxis], score_pairs_h, 0.0)
                        membership_pairs_h = np.where(mask[:,:,np.newaxis], membership_pairs_h, 0)

                        max_index_flat = np.argmax(f_pairs)
                        best_pair = list(np.unravel_index(max_index_flat, f_pairs.shape))
                        best_pair_global = (best_pair[0] + start_idx_i, best_pair[1] + start_idx_j)

                        # Copy results from device to host for the current batch
                        best_pairs[batch_i,batch_j] = np.array(best_pair_global,dtype=np.int32)
                        hhi_pairs[batch_i,batch_j] = hhi_pairs_h[best_pair[0], best_pair[1]]
                        weights_pairs[batch_i,batch_j] = weights_pairs_h[best_pair[0], best_pair[1]]
                        score_pairs[batch_i,batch_j] = score_pairs_h[best_pair[0], best_pair[1]]
                        membership_pairs[batch_i,batch_j] = membership_pairs_h[best_pair[0], best_pair[1]]

                        pbar_inner.update(1)

                # Synchronize all streams
                for stream in streams:
                    stream.synchronize()

                f1_pairs = -np.mean(hhi_pairs,axis=2) / 100
                f2_pairs = np.mean(score_pairs,axis=2)

                f_pairs = multiobjective(f1_pairs,f2_pairs,weights=weights)

                # Find the index of the maximum element
                max_index_flat = np.argmax(f_pairs)
                best_batch = np.unravel_index(max_index_flat, f_pairs.shape)

                # Calculate the increase in objective
                best_pair = tuple(best_pairs[best_batch[0], best_batch[1]])
                f_new = f_pairs[best_batch[0], best_batch[1]]
                f1 = f1_pairs[best_batch[0], best_batch[1]]
                f2 = f2_pairs[best_batch[0], best_batch[1]]
                max_increase = f_new - f

                # free memory
                communities_d = None
                membership_d = None
                edge_HHI_d = None
                node_weight_d = None
                node_score_d = None
                membership_pairs_d = None
                hhi_pairs_d = None
                weights_pairs_d = None
                score_pairs_d = None

                # If the maximum increase is greater than 0, merge these two communities and update the mean HHI
                if max_increase > 0.0:

                    self.update_scores(*best_pair)
                    self.communities_dict = self._communities_dict
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
                    f2 = sum(self._airline_score.values()) / len(self._airline_score)
                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        if debug:
            return membership_pairs, hhi_pairs, weights_pairs, score_pairs

    def _greedy_optimize_cpu(
            self,
            weights: Optional[Tuple[float,float]]=(0.5,0.5),
            max_iterations: Optional[int]=1000,
            n_samples: Optional[int] = None
        ) -> Tuple[Dict[int,int], float]:

        self.random_state = _default_if_none(self.random_state, random_state(self.seed), "rng_state")

        rng = self.random_state[0]
        rng_np = self.random_state[1]

        # Calculate the initial mean HHI over all routes
        f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
        f2 = sum(self._airline_score.values()) / len(self._airline_score)
        f = multiobjective(f1,f2,weights)

        # log history
        self.communities_history.append(self.communities_dict)
        self.f_history.append(f)
        self.multi_history.append((f1,f2))

        # Initialize flag for improvement
        improvement = True

        # Loop while improvement is possible
        n_iterations = 0
        with tqdm(total=max_iterations, desc="Outer Loop") as pbar_outer:
            while improvement and n_iterations < max_iterations:
                improvement = False
                max_increase = 0
                best_pair = None

                # cache current partition
                communities_dict = self.communities_dict
                market_shares_community, HHI = self._get_cache_HHI()
                weights_dict, airline_score = self._get_cache_MAI()

                if n_samples is None or n_samples >= len(self._distinct_communities):
                    search_communities = self._distinct_communities
                else:
                    assert n_samples >= 2, "number of samples must be at least 2"
                    search_communities = rng.sample(list(self._distinct_communities),k=n_samples)

                # For each pair of communities
                n_pairs = len(search_communities)**2
                with tqdm(total=n_pairs, desc="Inner Loop", leave=False) as pbar_inner:
                    for community1 in search_communities:
                        for community2 in search_communities:
                            if community1 != community2:
                                # Calculate the mean HHI if these two communities were to be merged
                                self.update_scores(community1,community2)

                                f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
                                f2 = sum(self._airline_score.values()) / len(self._airline_score)
                                f_new = multiobjective(f1,f2,weights)

                                # Calculate the increase in mean HHI
                                increase = f_new - f

                                # If this pair gives a better increase, update max_increase and best_pair
                                if increase > max_increase:
                                    max_increase = increase
                                    best_pair = (community1, community2)

                                # Revert the merge
                                self.communities_dict = communities_dict
                                self._set_cache_HHI(market_shares_community, HHI)
                                self._set_cache_MAI(weights_dict, airline_score)

                            pbar_inner.update(1)

                # If the maximum increase is greater than 0, merge these two communities and update the mean HHI
                if max_increase > 0:
                    self.update_scores(*best_pair)
                    self.communities_dict = self._communities_dict
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
                    f2 = sum(self._airline_score.values()) / len(self._airline_score)
                    self.communities_history.append(communities_dict.copy())
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        # Print the final community structure and the final mean HHI
        print("Final community structure:", self._communities_dict)
        print("Final objective:", f)

        return self.communities_dict, f

    @cpu_gpu_dispatcher('_greedy_optimize_cpu', '_greedy_optimize_gpu')
    def greedy_optimize(
            self,
            weights: Optional[Tuple[float,float]]=(0.5,0.5),
            max_iterations: Optional[int]=1000,
            n_samples: Optional[int] = None,
            **kwargs
        ) -> Tuple[Dict[int,int], float]:
        """
        performs greedy optimization on the graph community
        structure such that HHI score is minimized and
        MAI score is maximized

        Parameters
        ----------
        weights : Optional[Tuple[float,float]], optional
            The weighting used for the importance of
            HHI vs MAI, by default (0.5,0.5)
        max_iterations : Optional[int], optional
            maximium number of iterations to terminate at, by default 1000
        n_samples : Optional[int], optional
            number of random community pair samples to merge during
            each iteration of the greedy algorithm, by default None

        Returns
        -------
        Tuple[Dict[int,int], float]
            returns a pair of the optimal community partitions
            and the objective function value
        """
        pass
    
    def init_gpu(
            self,
            tile_size: Optional[int] = 42, 
            block_size: Optional[int] = 24
        ):

        assert self.use_gpu, "error: use_gpu flag set to False"

        from .utilities_cuda import cuda, set_membership, calc_ami, calc_hhi, \
            create_xoroshiro128p_states, define_cuda_kernels
        
        from ..utilities_cuda import elementwise_divide

        self.cuda = cuda
        # self.DeviceNDArray = DeviceNDArray
        self.set_batch_membership: Callable = set_membership
        self.calc_ami: Callable = calc_ami
        self.calc_hhi: Callable = calc_hhi
        self.create_xoroshiro128p_states: Callable = create_xoroshiro128p_states
        self.elementwise_divide: Callable = elementwise_divide

        # move constants to memory
        self._adj_d = self.cuda.to_device(self._adj)
        self._adj_dense_d = self.cuda.to_device(self._adj_dense)
        self._adj3_d = self.cuda.to_device(self._adj3)
        self._adj_dense_index_d = self.cuda.to_device(self._adj_dense_index)
        self._edge_nodes_d = self.cuda.to_device(self._edge_nodes)
        self._edge_weights_d = self.cuda.to_device(self._edge_weights)
        self._edges_neighbour_d = self.cuda.to_device(self._edges_neighbour)
        self._edge_unique_index_d = self.cuda.to_device(self._edge_unique_index)
        self._node_weight_individual = np.array(list(self._weights_individual_dict.values()),dtype=np.float32)
        self._market_shares_sparse_values_d = self.cuda.to_device(self._market_shares_sparse_values)
        self._market_shares_sparse_nodes_d = self.cuda.to_device(self._market_shares_sparse_nodes)
        
        node_spatial_list = np.array(list(self.node_spatial_list),np.int32)
        node_spatial_weights = np.array(list(self.node_spatial_weights),np.float32)

        self._node_spatial_list_d = self.cuda.to_device(node_spatial_list)
        self._node_spatial_weights_d = self.cuda.to_device(node_spatial_weights)


        self.set_membership, self.spatial_random_walk, self.weighted_spatial_random_walk, \
            self.attr_sampler, self.attr_source_postprocess, self.edge_sampler, \
            self.init_edge_counts, self.edge_postprocess, \
            self.load_counts, self.edge_histogram, self.copy_reduced_counts, \
            self.init_attr, self.attr_postprocess = define_cuda_kernels(self.n_nodes,self.n_nodes_spatial, tile_size, block_size)

    def weighted_sampling(
            self,
            n_samples_source_node: Optional[int] = 20,
            n_samples_edge: Optional[int] = 5,
            walk_length: Optional[int] = 10,
            TPB: Optional[Union[int,Tuple[int,int,int]]] = 2,
            testing: Optional[bool] = False
        ):

        walks_weighted_d = self.cuda.device_array((n_samples_source_node,n_samples_edge,walk_length), dtype=np.int32)
        walks_weighted_edges_d = self.cuda.device_array((n_samples_source_node,n_samples_edge,walk_length-1), dtype=np.int32)

        cumsum_weighted_d = self.cuda.device_array((n_samples_source_node,n_samples_edge,self.n_nodes_spatial,),dtype=np.float32)
        weights_weighted_d = self.cuda.device_array((n_samples_source_node,n_samples_edge,self.n_nodes_spatial,),dtype=np.float32)

        # Define the thread and block dimensions
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(n_samples_source_node, n_samples_edge), TPB = TPB)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)
        self.weighted_spatial_random_walk[blockspergrid,threadsperblock](
            n_samples_source_node,
            n_samples_edge,
            walk_length,
            self._node_spatial_weights_d,
            self._adj_dense_d,
            self._adj_dense_index_d,
            walks_weighted_d,
            walks_weighted_edges_d,
            cuda_random_state,
            cumsum_weighted_d,
            weights_weighted_d
        )

        walks_weighted_h = walks_weighted_d.copy_to_host()
        walks_weighted_edges_h = walks_weighted_edges_d.copy_to_host()

        if testing:
            return walks_weighted_h, walks_weighted_edges_h

    def _stochastic_eval_gpu(
            self, 
            community_i, 
            community_j, 
            edge_samples_d,
            walks_nodes_weights_d,
            walks_nodes_d,
            source_counts_d,
            membership_pairs_d,
            n_samples_edge,
            walk_length,
            TPB,
            stream,
            testing = False,
            parallel_reduce = False
        ):

        #################################
        # merge values

        reduced_counts_d = self.cuda.device_array((self.n_unique_edges,self.n_communities), dtype=np.int32, stream=stream)
        cum_weight_d = self.cuda.device_array((self.n_nodes_spatial,self.n_nodes), dtype=np.float32, stream=stream)

        if parallel_reduce:

            edge_counts_d = self.cuda.device_array((self.n_unique_edges,n_samples_edge,self.n_communities), dtype=np.int16, stream=stream)

            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, n_samples_edge), TPB = TPB)

            self.load_counts[blockspergrid,threadsperblock,stream](
                community_i,
                community_j,
                self.n_communities,
                n_samples_edge,
                self.n_unique_edges,
                edge_samples_d,
                edge_counts_d,
                membership_pairs_d
            )

            stream.synchronize()

            # Kernel launch  
            self.edge_histogram[blockspergrid,threadsperblock,stream](
                self.n_communities,
                n_samples_edge,
                self.n_unique_edges,
                edge_counts_d,
                edge_counts_d
            )
            
            # Reduction part
            while blockspergrid[1] > 1:  
                n_samples_edge_r = blockspergrid[1]
                blockspergrid = list(blockspergrid)
                blockspergrid[1] = math.ceil(blockspergrid[1] / threadsperblock[1])
                blockspergrid = tuple(blockspergrid)

                # Kernel launch  
                self.edge_histogram[blockspergrid,threadsperblock,stream](
                    self.n_communities,
                    n_samples_edge_r,
                    self.n_unique_edges,
                    edge_counts_d,
                    edge_counts_d
                )

            stream.synchronize()

            # Transfer the first row of the matrix to the vector  
            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_communities), TPB = TPB)

            self.copy_reduced_counts[blockspergrid,threadsperblock,stream](
                self.n_communities,
                self.n_unique_edges,
                edge_counts_d,
                reduced_counts_d
            )

            stream.synchronize()

        else:

            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_communities), TPB = TPB)

            self.init_edge_counts[blockspergrid,threadsperblock,stream](
                self.n_communities,
                self.n_unique_edges,
                reduced_counts_d,
            )

            stream.synchronize()

            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, n_samples_edge), TPB = TPB)

            self.edge_postprocess[blockspergrid,threadsperblock,stream](
                community_i,
                community_j,
                n_samples_edge,
                self.n_unique_edges,
                edge_samples_d,
                reduced_counts_d,
                membership_pairs_d
            )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_nodes), TPB = TPB)

        self.init_attr[blockspergrid,threadsperblock,stream](cum_weight_d,)

        stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, n_samples_edge), TPB = TPB)

        self.attr_postprocess[blockspergrid,threadsperblock,stream](
            community_i,
            community_j,
            n_samples_edge,
            walk_length,
            walks_nodes_weights_d,
            walks_nodes_d,
            source_counts_d,
            cum_weight_d,
            membership_pairs_d
        )

        stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, n_samples_edge), TPB = TPB)

        self.elementwise_divide[blockspergrid,threadsperblock,stream](
            cum_weight_d,
            source_counts_d
        )

        stream.synchronize()

        edge_samples_h = edge_samples_d.copy_to_host()

        walks_nodes_h = walks_nodes_d.copy_to_host()
        walks_nodes_weights_h = walks_nodes_weights_d.copy_to_host()

        edge_counts_h = reduced_counts_d.copy_to_host() / n_samples_edge
        cum_weight_h = cum_weight_d.copy_to_host()

        hhi_estimate = edge_counts_h**2
        hhi_estimate = np.nansum(hhi_estimate,axis=1)

        mai_estimate = cum_weight_h
        mai_estimate = np.nansum(mai_estimate,axis=0)
        log_mai = np.where(mai_estimate > 0.0, np.log(mai_estimate), 0.0)

        if testing:
            return edge_samples_h, walks_nodes_h, walks_nodes_weights_h, edge_counts_h, cum_weight_h, hhi_estimate, mai_estimate, log_mai
        else:
            return hhi_estimate, mai_estimate, log_mai

    def _randomwalk_gpu(
            self,
            n_samples_edge,
            walk_length,
            TPB_2D,
            TPB_3D,
            testing = False
        ):

        #################################
        # perform random walks
        edge_samples_d = self.cuda.device_array((self.n_unique_edges,n_samples_edge), dtype=np.int32)

        walks_d = self.cuda.device_array((self.n_nodes_spatial,n_samples_edge,walk_length), dtype=np.int32)
        walks_edges_d = self.cuda.device_array((self.n_nodes_spatial,n_samples_edge,walk_length-1), dtype=np.int32)

        walks_nodes_d = self.cuda.device_array((self.n_nodes_spatial,n_samples_edge,walk_length-1), dtype=np.int32)
        walks_nodes_weights_d = self.cuda.device_array((self.n_nodes_spatial,n_samples_edge,walk_length-1), dtype=np.float32)

        source_counts_h = np.zeros((self.n_nodes_spatial,self.n_nodes), dtype=np.int32)
        source_counts_d = self.cuda.to_device(source_counts_h)

        cumsum_d = self.cuda.device_array((self.n_nodes_spatial,n_samples_edge,self.n_nodes_spatial,),dtype=np.float32)
        weights_d = self.cuda.device_array((self.n_nodes_spatial,n_samples_edge,self.n_nodes_spatial,),dtype=np.float32)
        
        # Define the thread and block dimensions
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, n_samples_edge), TPB = TPB_2D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.edge_sampler[blockspergrid,threadsperblock](
            n_samples_edge,
            self.n_unique_edges,
            self._adj3_d,
            edge_samples_d,
            cuda_random_state,
        )

        # Define the thread and block dimensions
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, n_samples_edge), TPB = TPB_2D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.spatial_random_walk[blockspergrid,threadsperblock](
            n_samples_edge,
            walk_length,
            self._node_spatial_list_d,
            self._adj_dense_d,
            self._adj_dense_index_d,
            walks_d,
            walks_edges_d,
            cuda_random_state,
            cumsum_d,
            weights_d
        )

        self.cuda.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, n_samples_edge, walk_length-1), TPB = TPB_3D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.attr_sampler[blockspergrid,threadsperblock](
            n_samples_edge,
            walk_length,
            walks_edges_d,
            self._adj_d,
            self._adj3_d,
            walks_nodes_weights_d,
            walks_nodes_d,
            cuda_random_state
        )

        self.cuda.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, n_samples_edge), TPB = TPB_2D)
        self.attr_source_postprocess[blockspergrid,threadsperblock](
            n_samples_edge,
            walks_nodes_d,
            source_counts_d
        )

        self.cuda.synchronize()

        if testing:
            return edge_samples_d, walks_d, walks_edges_d, walks_nodes_d, walks_nodes_weights_d, source_counts_d
        else:
            return edge_samples_d, walks_nodes_d, walks_nodes_weights_d, source_counts_d

    def _calculate_members_gpu(self,TPB):
        
        # iteration constants
        communities_d = self.cuda.to_device(self._distinct_communities)
        membership_d = self.cuda.to_device(list((self._communities_dict.values())))

        # iteration values
        membership_pairs_d = self.cuda.device_array((self.n_communities, self.n_communities, self.n_nodes), dtype=np.int16)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_communities, self.n_communities, self.n_nodes), TPB = TPB)
        self.set_membership[blockspergrid,threadsperblock](
            communities_d,
            membership_d,
            membership_pairs_d,
        )

        self.cuda.synchronize()

        return membership_pairs_d

    def _process_batch(
            self, 
            batch_index, 
            streams, 
            edge_samples_d, 
            walks_nodes_weights_d, 
            walks_nodes_d, 
            source_counts_d, 
            membership_pairs_d, 
            n_samples_edge, 
            walk_length, 
            TPB_2D, 
            weights, 
            f,
            parallel_reduce
        ):

        community_i = batch_index // self.n_communities
        community_j = batch_index % self.n_communities

        if community_i > community_j:
            stream = streams[batch_index % len(streams)]

            # Perform stochastic evaluation using the selected stream
            hhi_estimate, mai_estimate, log_mai = self._stochastic_eval_gpu(
                community_i,
                community_j,
                edge_samples_d,
                walks_nodes_weights_d,
                walks_nodes_d,
                source_counts_d,
                membership_pairs_d,
                n_samples_edge,
                walk_length,
                stream=stream,
                TPB=TPB_2D,
                parallel_reduce = parallel_reduce
            )

            f1 = -np.mean(hhi_estimate) * 100
            f2 = np.mean(log_mai)
            f_new = multiobjective(f1, f2, weights=weights)

            increase = f_new - f

            return increase, community_i, community_j, hhi_estimate, mai_estimate

        return 0, community_i, community_j, None, None

    def greedy_optimize_stochastic(
            self,
            weights: Optional[Tuple[float,float]]=(0.5,0.5),
            max_iterations: Optional[int]=1000,
            n_samples_edge: Optional[int] = 100,
            walk_length: Optional[int] = 10,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 24,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            num_streams: Optional[int] = 1,
            testing_opt: Optional[bool] = False,
            parallel_reduce: Optional[bool] = False
        ) -> Tuple[Dict[int,int], float]:

        # Calculate the initial mean HHI over all routes
        f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
        f2 = sum(self._airline_score.values()) / len(self._airline_score)
        f = multiobjective(f1,f2,weights)

        # log history
        self.communities_history.append(self.communities_dict)
        self.f_history.append(f)
        self.multi_history.append((f1,f2))

        # Initialize flag for improvement
        improvement = True

        # Loop while improvement is possible
        n_iterations = 0
        with tqdm(total=max_iterations, desc="Outer Loop") as pbar_outer:
            while improvement and n_iterations < max_iterations:

                improvement = False
                max_increase = 0
                best_pair = None

                TPB_2D = threads_per_block_2D
                TPB_3D = threads_per_block_3D
                
                if testing_opt:
                    hhi_pairs_h = np.zeros((self.n_communities, self.n_communities, self.n_unique_edges), dtype=np.float32)
                    weights_pairs_h = np.zeros((self.n_communities, self.n_communities, self.n_nodes), dtype=np.float32)

                membership_pairs_d = self._calculate_members_gpu(TPB_3D)

                edge_samples_d, walks_nodes_d, walks_nodes_weights_d, source_counts_d = self._randomwalk_gpu(
                    n_samples_edge,
                    walk_length,
                    TPB_2D,
                    TPB_3D
                )

                # Create streams
                streams = [self.cuda.stream() for _ in range(num_streams)]
                num_total_batches = self.n_communities * self.n_communities

                # Use ThreadPoolExecutor to parallelize the loop
                with ThreadPoolExecutor(max_workers=num_streams) as executor:
                    futures = []
                    for batch_index in range(num_total_batches):
                        future = executor.submit(
                            self._process_batch, 
                            batch_index, 
                            streams, 
                            edge_samples_d, 
                            walks_nodes_weights_d, 
                            walks_nodes_d, 
                            source_counts_d, 
                            membership_pairs_d, 
                            n_samples_edge, 
                            walk_length, 
                            TPB_2D, 
                            weights, 
                            f,
                            parallel_reduce
                        )
                        futures.append(future)

                    # Collect results and update max_increase and best_pair
                    for future in tqdm(futures, total=num_total_batches, desc="merges processed"):
                        increase, community_i, community_j, hhi_estimate, mai_estimate = future.result()

                        if community_i > community_j:
                            if increase > max_increase:
                                max_increase = increase
                                best_pair = (community_i, community_j)
        
                            if testing_opt:
                                hhi_pairs_h[community_i,community_j,:] = hhi_estimate
                                weights_pairs_h[community_i,community_j,:] = mai_estimate

                # Synchronize all streams
                for stream in streams:
                    stream.synchronize()

                # If the maximum increase is greater than 0, merge these two communities and update the mean HHI
                if max_increase > 0:
                    self.update_scores(*best_pair)
                    self.communities_dict = self._communities_dict
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
                    f2 = sum(self._airline_score.values()) / len(self._airline_score)
                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))
                
                n_iterations += 1
                pbar_outer.update(1)

        if testing_opt:
            return membership_pairs_d.copy_to_host(), hhi_pairs_h, weights_pairs_h

    def _update_communities(self, community_to_merge, new_community, update_communities=True):
        # Find the nodes in the community to merge
        nodes_to_update_HHI = [k for k, v in self._communities_dict.items() if v in [community_to_merge,]]
        nodes_to_update_MAI = [k for k, v in self._communities_dict.items() if v in [community_to_merge,new_community]]

        # Update the community in communities_dict ands reset the market shares for this route
        edges_to_update = set()
        for node in nodes_to_update_HHI:
            if update_communities:
                self._communities_dict[node] = new_community

            edges_to_update.update(set(((u,v) for u,v,key in self.edge_node2keypair_list[node])))

        return edges_to_update, nodes_to_update_MAI

    def _update_HHI_scores(self, edges_to_update):

        self._cond_probs_community_sparse = []
        for u, v in edges_to_update:
            self._cond_probs_community[(u, v)] = defaultdict(float)

        for u, v in edges_to_update:
            data = self.G.get_edge_data(u,v)
            for key,data_node in data.items():
                # Recalculate the market shares and HHI score for this route
                self._cond_probs_community[(u, v)][self._communities_dict[data_node['edge_type']]] += data_node['market_share_percent']

        # Recalculate the community HHI for each affected route
        for edge, communities in self._cond_probs_community.items():
            self._HHI[edge] = sum(market_share ** 2 for market_share in communities.values())

    def _update_scores(self, nodes_to_update):

        # Recalculate scores for the affected airlines
        for node in nodes_to_update:
            target_community = self._communities_dict[node]

            # Get edges by 'edge_type' from the pre-grouped dictionary
            edges = self.edge_node2keypair_list[node]

            # Get 'to' nodes
            to_nodes = {v for _, v, _ in edges}

            # Calculate weights
            if len(to_nodes) > 0:
                edges = set(sum((self.edge_u2pairkey_list[node] for node in to_nodes), []))
                weights = [self.edge_pairkey2atrr[(u,v,key)]['weight'] for u,v,key in edges if self._communities_dict[self.edge_pairkey2atrr[(u,v,key)]['edge_type']] == target_community]

                self._weights_dict[node] = sum(weights)

                if self._weights_individual_dict[node] > 0:
                    self._airline_score[node] = np.log(self._weights_dict[node] + self._weights_individual_dict[node])

    def _get_cache_HHI(self, edges:Optional[List[Tuple[int,int]]]=None) -> Tuple[type_shares, type_scores]:

        if edges is None:
            edges = self._cond_probs_community.keys()

        market_shares_community = {}
        HHI = {}
        for edge in edges:
            market_shares_community[edge] = self._cond_probs_community[edge].copy()
            HHI[edge] = self._HHI[edge]

        return market_shares_community, HHI

    def _get_cache_MAI(self, nodes: List[int]=None) -> Tuple[type_scores, type_scores]:

        if nodes is None:
            nodes = self._weights_dict.keys()

        weights_dict = {}
        airline_score = {}
        for node in nodes:
            weights_dict[node] = self._weights_dict[node]
            airline_score[node] = self._airline_score[node]

        return weights_dict, airline_score

    def _set_cache_HHI(self, market_shares_community: type_shares, HHI: type_scores):
        for edge in market_shares_community.keys():
            self._cond_probs_community[edge] = market_shares_community[edge].copy()
            self._HHI[edge] = HHI[edge]

    def _set_cache_MAI(self, weights_dict: type_scores, airline_score: type_scores):
        for node in weights_dict.keys():
            self._weights_dict[node] = weights_dict[node]
            self._airline_score[node] = airline_score[node]

    def __del__(self):
        if self.use_gpu:
            print("Destructor: Instance destroyed")
            del self._adj_d
            del self._adj_dense_d
            del self._adj3_d
            del self._adj_dense_index_d
            del self._edge_nodes_d
            del self._edge_weights_d
            del self._edges_neighbour_d
            del self._edge_unique_index_d
            del self._node_weight_individual
            del self._market_shares_sparse_values_d
            del self._market_shares_sparse_nodes_d
            del self._node_spatial_list_d
            del self._node_spatial_weights_d
