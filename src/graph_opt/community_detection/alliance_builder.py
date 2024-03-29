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
import itertools

from abc import ABC

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from .utilities import resort_communities
from ..utilities import _default_if_none, dict2array, cpu_gpu_dispatcher, cuda_blocks

from typing import Dict, List, Union, Tuple, Optional, NewType
from ..random_settings import random_state

type_cp = NewType('type_cp', Dict[int,float])
type_cps = NewType('type_cps', Dict[Tuple[int, int], type_cp])
type_hhi = NewType('type_hhi', Dict[Tuple[int, int], Dict[int, float]])
type_bfs = NewType('type_bfs', Dict[int, float])

def multiobjective(f1: Union[float,np.ndarray], f2: Union[float,np.ndarray], weights: Optional[Tuple[float,float]]=(0.5,0.5)):
    return weights[0]*f1 + weights[1]*f2

class BaseCommunity(ABC):

    def __init__(
            self,
            MG: MultiDiGraph,
            communities_dict: Dict[str,List[str]] = None,
            seed: int = None,
            use_gpu: Optional[bool] = False,
        ) -> None:

        self._communities_dict = {}
        self.G: MultiDiGraph = deepcopy(MG)
        self.attr_list = []
        self.edge_unique_list = []
        self.node_spatial_list = []
        self.node_spatial_degree = []
        self.attr_key2i = {}
        self.attr_i2key = {}
        self.attr_key2w = defaultdict(float)
        self.node_spatial_key2i = {}
        self.node_spatial_key2w = {}
        self.node_spatial_i2key = {}
        self.edge_pairkey2ui = {}
        self.edge_pair2w = defaultdict(float)
        self.edge_pair2ui = {}
        self.edge_ui2pair = {}

        self.edge_attr2keypair_list = defaultdict(list)
        self.edge_u2pairkey_list = defaultdict(list)
        self.edge_u2pair_list = defaultdict(list)
        self.edge_pairkey2atrr = {}
        self.edge_pair2pairkey_list = defaultdict(list)
        self.edge_pairkey2i = {}

        self._hhi = defaultdict(float)
        self._cond_probs_attr = defaultdict(lambda: defaultdict(float))
        self._cond_probs_community = defaultdict(lambda: defaultdict(float))
        self._cond_probs_community_sparse = []

        self._adj_dense = nx.to_numpy_array(self.G, weight="weight")
        self._adj3 = np.empty(0)
        self._adj_dense_index = np.empty(0)
        self._adj = np.empty(0)

        self._log_bfs_degree = {}
        self._bfs_degree = defaultdict(list)
        self.bfs_degree_community = np.empty(0)
        self._sampled_communities_pairs = set()

        self.communities_history = []
        self.f_history = []
        self.multi_history = []

        self.seed = seed
        self.random_state: Tuple[random.Random,np.random.RandomState] = random_state(self.seed)

        self.use_gpu = use_gpu

        self._parse_graph()
        self._initialize_communities(communities_dict)

    def init_gpu(
            self,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
        ):
        assert self.use_gpu, "error: use_gpu flag set to False"
        from .utilities_cuda import cuda
        self.cuda = cuda
        self._adj_d = self.cuda.to_device(self._adj)

        self.TPB_2D = threads_per_block_2D
        self.TPB_3D = threads_per_block_3D

    @property
    def communities_dict(self)->Dict[str,int]:
        return {attr: self._communities_dict[attr] for attr in self._communities_dict.keys()}

    @communities_dict.setter
    def communities_dict(self,value:Dict[str,int]):
        key, self._communities_dict, self._distinct_communities, self.n_communities = resort_communities(value)

    @property
    def market_shares_community(self) -> type_cps:
        return {edge: self._cond_probs_community[edge].copy() for edge in self._cond_probs_community.keys()}

    @property
    def hhi(self) -> Dict[str,float]:
        return {edge: self._hhi[edge] for edge in self._hhi.keys()}

    @property
    def bfs_degree(self) -> Dict[str,float]:
        return {attr: self._bfs_degree[attr] for attr in self._bfs_degree.keys()}

    @property
    def log_bfs_degree(self) -> Dict[str,float]:
        return {attr: self._log_bfs_degree[attr] for attr in self._log_bfs_degree.keys()}

    @log_bfs_degree.setter
    def log_bfs_degree(self, value: Dict[str,float]):
        self._log_bfs_degree = value

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
        structure such that hhi score is minimized and
        bfs_degree score is maximized

        Parameters
        ----------
        weights : Optional[Tuple[float,float]], optional
            The weighting used for the importance of
            hhi vs bfs_degree, by default (0.5,0.5)
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

    def update_scores(self, community_to_merge: int, new_community: int) -> Tuple[List[Tuple[str,str]], List[str]]:
        return self._update_communities(community_to_merge,new_community)
    
    def _update_scores_community(self):
        self.bfs_degree_community = np.zeros(self.n_communities)
        for i, v in self.bfs_degree.items():
            self.bfs_degree_community[self.communities_dict[i]] += v

    def _parse_graph(self):
        """
        Group edges by 'edge_type' and source attr for faster lookup
        """

        # initialize attr
        self.m_edges = self.G.number_of_edges()
        edge_types = set()
        for i, (u, v, key, data) in enumerate(self.G.edges(keys=True, data=True)):
            self.edge_attr2keypair_list[data['edge_type']].append((u, v, key))
            self.edge_u2pairkey_list[u].append((u, v, key))
            self.edge_pairkey2atrr[(u, v, key)] = data
            self.edge_pairkey2i[(u, v, key)] = i
            self.edge_pair2pairkey_list[(u, v)].append((u, v, key))
            self.edge_pair2w[(u, v)] += data['weight']
            self.attr_key2w[data['edge_type']] += data['weight']

            # conditional probabilities
            self._cond_probs_attr[(u, v)][data['edge_type']] += data['market_share_percent']

            # attribute lists
            edge_type = data.get('edge_type', None)
            if edge_type:
                edge_types.add(edge_type)

        # attribute lists
        attr_list = sorted(edge_types)
        for i,attr in enumerate(attr_list):
            self.attr_list.append(attr)
            self.attr_key2i[attr] = i
            self.attr_i2key[i] = attr

        self.edge_unique_list = list(self._cond_probs_attr.keys())
        self.n_attr = len(self.attr_list)
        self.n_unique_edges = len(self.edge_unique_list)

        for i, u in enumerate(self.G.nodes()):
            node_degree = self.G.out_degree(nbunch=u,weight="weight")
            self.node_spatial_key2w[u] = node_degree
            self.node_spatial_key2i[u] = i
            self.node_spatial_key2i[u] = i
            self.node_spatial_i2key[i] = u
            self.node_spatial_list.append(u)
            self.node_spatial_degree.append(node_degree)

        self.n_nodes_spatial = len(self.node_spatial_list)

        adj3 = np.zeros((self.n_unique_edges,self.n_attr))
        adj = np.zeros((self.n_unique_edges))
        adj_dense_index = np.full((self.n_nodes_spatial, self.n_nodes_spatial), np.nan)
        for i, (edge, attr) in enumerate(self._cond_probs_attr.items()):
            self.edge_u2pair_list[edge[0]].append(edge)
            self.edge_pair2ui[edge] = i
            self.edge_ui2pair[i] = edge
            adj_dense_index[self.node_spatial_key2i[edge[0]],self.node_spatial_key2i[edge[1]]] = i
            adj[i] = self._adj_dense[self.node_spatial_key2i[edge[0]],self.node_spatial_key2i[edge[1]]]
            for attr, value in attr.items():
                adj3[i,self.attr_key2i[attr]] = value
            for u,v,key in self.edge_pair2pairkey_list[edge]:
                self.edge_pairkey2ui[(u,v,key)] = i

        self._adj3 = adj3
        self._adj = adj
        self._adj_dense_index = adj_dense_index

    def _initialize_communities(self, communities_dict: Optional[Dict[str,str]] = None):
        """
        Initialize each attr to be its own community

        Parameters
        ----------
        communities_dict : Dict[str,str]
            _description_
        """

        if communities_dict is None:

            communities_dict = {}
            for i,attr in enumerate(self.attr_list):
                communities_dict[attr] = i

            self.communities_dict = communities_dict
        else:
            self.communities_dict = communities_dict

    def _update_communities(self, community_to_merge: int, new_community: int, update_communities: Optional[bool]=True) -> Tuple[List[Tuple[str,str]], List[str]]:
        # Find the attr in the community to merge
        attr_to_update_hhi = [k for k, v in self._communities_dict.items() if v in [community_to_merge,]]
        attr_to_update_bfs_degree = [k for k, v in self._communities_dict.items() if v in [community_to_merge,new_community]]

        # Update the community in communities_dict ands reset the market shares for this route
        edges_to_update = set()
        for attr in attr_to_update_hhi:
            if update_communities:
                self._communities_dict[attr] = new_community

            edges_to_update.update(set(((u,v) for u,v,key in self.edge_attr2keypair_list[attr])))

        return edges_to_update, attr_to_update_bfs_degree

    def _get_cache_hhi(self, edges:Optional[List[Tuple[int,int]]]=None) -> Tuple[type_cps, type_hhi]:

        if edges is None:
            edges = self._cond_probs_community.keys()

        cond_probs_community = {}
        hhi = {}
        for edge in edges:
            cond_probs_community[edge] = self._cond_probs_community[edge].copy()
            hhi[edge] = self._hhi[edge]

        return cond_probs_community, hhi

    def _get_cache_bfs_degree(self, attr: List[int]=None) -> Tuple[type_bfs, type_bfs]:

        if attr is None:
            attr = self._bfs_degree.keys()

        bfs_degree = {}
        log_bfs_degree = {}
        for attr in attr:
            bfs_degree[attr] = self._bfs_degree[attr]
            log_bfs_degree[attr] = self._log_bfs_degree[attr]

        return bfs_degree, log_bfs_degree

    def _set_cache_hhi(self, cond_probs_community: type_cps, hhi: type_hhi):
        for edge in cond_probs_community.keys():
            self._cond_probs_community[edge] = cond_probs_community[edge].copy()
            self._hhi[edge] = hhi[edge]

    def _set_cache_bfs_degree(self, bfs_degree: type_bfs, log_bfs_degree: type_bfs):
        for attr in bfs_degree.keys():
            self._bfs_degree[attr] = bfs_degree[attr]
            self._log_bfs_degree[attr] = log_bfs_degree[attr]

    def _sample_community_pairs(self, n_samples: int = 5, replace: Optional[bool] = False) -> Union[list,bool]:
        
        rng_np = self.random_state[1]
        all_samples = list(itertools.product(range(self.n_communities), range(self.n_communities)))  

        sample_set = set(all_samples) - self._sampled_communities_pairs
        sample_list = list(sample_set)  # convert set to list  

        if not replace:
            n_samples = min(len(sample_list),n_samples)

        diag_mask = np.array([True if i==j else False for i,j in sample_list])

        probabilities = self.bfs_degree_community[sample_list] / np.sum(self.bfs_degree_community)
        probabilities_pairs = probabilities[:,0] * probabilities[:,1]
        probabilities_pairs[diag_mask] = 0.0
        probabilities_pairs = probabilities_pairs / np.sum(probabilities_pairs)

        if len(sample_list) > 0:
            random_indices = rng_np.choice(len(sample_list), n_samples, p=probabilities_pairs, replace=replace)
            sample_pairs = [sample_list[i] for i in random_indices]
            if not replace:
                self._sampled_communities_pairs |= set(sample_pairs)
            return sample_pairs

        return False

class DeterministicCommunity(BaseCommunity):

    def __init__(self, MG: MultiDiGraph, communities_dict: Dict[str, List[str]] = None, seed: int = None, use_gpu: bool | None = False) -> None:
        super().__init__(MG, communities_dict, seed, use_gpu)

        self._update_hhi_scores()
        self._update_bfs_scores()
        self._update_scores_community()

    def init_gpu(
            self,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
        ):
        super().init_gpu(threads_per_block_2D,threads_per_block_3D)

        from .utilities_cuda import define_cuda_kernels_deterministic

        # Populate the adjacency matrix
        edge_attr = []
        edge_weights = []
        edges_neighbour = []
        edge_unique_index = []
        processed_edges = set()

        for i, (u, v, key, data) in enumerate(self.G.edges(data=True, keys=True)):
            edge_attr.append(self.attr_key2i[data["edge_type"]])
            edge_weights.append(data["weight"])
            edge_unique_index.append(self.edge_pairkey2ui[(u, v, key)])
            if (v, key) not in processed_edges:
                for u_link, v_link, key_link in self.edge_u2pairkey_list[v]:
                    edges_neighbour.append((self.edge_pairkey2i[(u, v, key)], self.edge_pairkey2i[(u_link, v_link, key_link)]))
                processed_edges.add((v, key))

        # dense adjacency
        self._edge_attr = np.array(edge_attr,dtype=np.int32)
        self._edge_weights = np.array(edge_weights,dtype=np.float32)
        self._edges_neighbour = np.array(edges_neighbour,dtype=np.int32)
        self._edge_unique_index = np.array(edge_unique_index,dtype=np.int32)
        self.m_edge_pairs = self._edges_neighbour.shape[0]

        self._cp_sparse_values = []
        self._cp_sparse_attr = []
        for n,v in self._cond_probs_community_sparse:
            self._cp_sparse_values.append(v)
            self._cp_sparse_attr.append(n)

        self._cp_sparse_values = np.array(self._cp_sparse_values, dtype=np.float32)
        self._cp_sparse_attr = np.array(self._cp_sparse_attr, dtype=np.int32)

        # load kernels
        set_batch_membership, calc_hhi, calc_bfs = define_cuda_kernels_deterministic()
        self.set_batch_membership: Callable = set_batch_membership
        self.calc_hhi: Callable = calc_hhi
        self.calc_bfs: Callable = calc_bfs

        # move constants to memory
        self._edge_attr_d = self.cuda.to_device(self._edge_attr)
        self._edge_weights_d = self.cuda.to_device(self._edge_weights)
        self._edges_neighbour_d = self.cuda.to_device(self._edges_neighbour)
        self._edge_unique_index_d = self.cuda.to_device(self._edge_unique_index)
        self._cp_sparse_values_d = self.cuda.to_device(self._cp_sparse_values)
        self._cp_sparse_attr_d = self.cuda.to_device(self._cp_sparse_attr)

    def update_scores(self, community_to_merge: int, new_community: int):
        edges_to_update, attr_to_update = super().update_scores(community_to_merge, new_community)
        self._update_hhi_scores(edges_to_update)
        self._update_bfs_scores(attr_to_update)
        self._update_scores_community()

    def _greedy_optimize_gpu(
            self,
            weights: Optional[Tuple[float,float]]=(0.5,0.5),
            max_iterations: Optional[int]=1000,
            batch_size: Optional[Union[int,Tuple[int,int]]] = 2,
            num_streams: Optional[int] = 1,
            debug: bool = False
        ) -> Tuple[Dict[int,int], float]:

        # Calculate the initial mean hhi over all routes
        f1 = -(sum(self._hhi.values())*100) / (len(self._hhi)*10000)
        f2 = sum(self._log_bfs_degree.values()) / len(self._log_bfs_degree)
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
                edge_hhi_d = self.cuda.to_device(list(self._hhi.values()))
                attr_weight_d = self.cuda.to_device(list(self._bfs_degree.values()))

                # Define the batch size based on memory requirements and available GPU memory
                if isinstance(batch_size, int):
                    batch_size_i = batch_size
                    batch_size_j = batch_size
                else:
                    batch_size_i = batch_size[0]
                    batch_size_j = batch_size[1]

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
                bfs_pairs = np.zeros((num_batches_i, num_batches_j, self.n_attr), dtype=np.float32)
                score_pairs = np.zeros((num_batches_i, num_batches_j, self.n_attr), dtype=np.float32)
                membership_pairs = np.zeros((num_batches_i, num_batches_j, self.n_attr), dtype=np.int32)

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
                        membership_pairs_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_attr), dtype=np.int8)
                        local_array_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_unique_edges), dtype=np.float32)
                        hhi_pairs_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_unique_edges), dtype=np.float32)
                        bfs_pairs_h = np.zeros((end_idx_i - start_idx_i, end_idx_j - start_idx_j, self.n_attr), dtype=np.float32)

                        # move final arrays to device
                        membership_pairs_d = self.cuda.to_device(membership_pairs_h,stream=stream)
                        local_array_d = self.cuda.to_device(local_array_h,stream=stream)
                        hhi_pairs_d = self.cuda.to_device(hhi_pairs_h,stream=stream)
                        bfs_pairs_d = self.cuda.to_device(bfs_pairs_h,stream=stream)

                        # Define the thread and block dimensions
                        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(batch_size_i, batch_size_j, self.n_attr), TPB = self.TPB_3D)
                        self.set_batch_membership[blockspergrid,threadsperblock,stream](
                                start_idx_i,
                                start_idx_j,
                                end_idx_i,
                                end_idx_j,
                                self.n_attr,
                                attr_weight_d,
                                communities_d,
                                membership_d,
                                membership_pairs_d,
                                bfs_pairs_d,
                            )

                        stream.synchronize()
                        print('membership done')

                        # Define the thread and block dimensions
                        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(batch_size_i, batch_size_j), TPB = self.TPB_2D)
                        self.calc_hhi[blockspergrid,threadsperblock,stream](
                                start_idx_i,
                                start_idx_j,
                                end_idx_i,
                                end_idx_j,
                                self.n_unique_edges,
                                self.m_edges,
                                self.n_communities,
                                communities_d,
                                self._cp_sparse_attr_d,
                                self._cp_sparse_values_d,
                                self._edge_unique_index_d,
                                edge_hhi_d,
                                membership_pairs_d,
                                hhi_pairs_d,
                                local_array_d
                            )

                        # stream.synchronize()
                        # print('hhi done')

                        # Define the thread and block dimensions
                        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(batch_size_i, batch_size_j, self.n_attr), TPB = self.TPB_3D)
                        self.calc_bfs[blockspergrid,threadsperblock,stream](
                                start_idx_i,
                                start_idx_j,
                                end_idx_i,
                                end_idx_j,
                                self.n_attr,
                                self.m_edge_pairs,
                                communities_d,
                                membership_pairs_d,
                                self._edge_attr_d,
                                self._edge_weights_d,
                                self._edges_neighbour_d,
                                bfs_pairs_d,
                            )

                        stream.synchronize()
                        print('bfs done')

                        # Copy results from device to host for the current batch
                        hhi_pairs_h = hhi_pairs_d.copy_to_host(stream=stream)
                        bfs_pairs_h = bfs_pairs_d.copy_to_host(stream=stream)
                        score_pairs_h = np.where(bfs_pairs_h > 0.0, np.log(bfs_pairs_h), 0.0)
                        membership_pairs_h = membership_pairs_d.copy_to_host(stream=stream)

                        # free memory
                        membership_pairs_d = None
                        local_array_d = None
                        hhi_pairs_d = None
                        bfs_pairs_d = None

                        f1_pairs = -np.mean(hhi_pairs_h,axis=2) / 100
                        f2_pairs = np.mean(score_pairs_h,axis=2)
                        f_pairs = multiobjective(f1_pairs,f2_pairs,weights=weights)

                        # Create a mask where i + start_idx_i > j + start_idx_j
                        mask = np.fromfunction(lambda i, j: i + start_idx_i < j + start_idx_j, (end_idx_i - start_idx_i, end_idx_j - start_idx_j))
                        f_pairs = np.where(mask, f_pairs, -np.inf)
                        hhi_pairs_h = np.where(mask[:,:,np.newaxis], hhi_pairs_h, np.sign(weights[0])*10000.0)
                        bfs_pairs_h = np.where(mask[:,:,np.newaxis], bfs_pairs_h, 0.0)
                        score_pairs_h = np.where(mask[:,:,np.newaxis], score_pairs_h, 0.0)
                        membership_pairs_h = np.where(mask[:,:,np.newaxis], membership_pairs_h, 0)

                        max_index_flat = np.argmax(f_pairs)
                        best_pair = list(np.unravel_index(max_index_flat, f_pairs.shape))
                        best_pair_global = (best_pair[0] + start_idx_i, best_pair[1] + start_idx_j)

                        # Copy results from device to host for the current batch
                        best_pairs[batch_i,batch_j] = np.array(best_pair_global,dtype=np.int32)
                        hhi_pairs[batch_i,batch_j] = hhi_pairs_h[best_pair[0], best_pair[1]]
                        bfs_pairs[batch_i,batch_j] = bfs_pairs_h[best_pair[0], best_pair[1]]
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
                del communities_d
                del membership_d
                del edge_hhi_d
                del attr_weight_d
                del membership_pairs_d
                del hhi_pairs_d
                del bfs_pairs_d

                # If the maximum increase is greater than 0, merge these two communities and update the mean hhi
                if max_increase > 0.0:

                    self.update_scores(*best_pair)
                    self.communities_dict = self._communities_dict
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -(sum(self._hhi.values())*100) / (len(self._hhi)*10000)
                    f2 = sum(self._log_bfs_degree.values()) / len(self._log_bfs_degree)
                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        if debug:
            return membership_pairs, hhi_pairs, bfs_pairs, score_pairs

    def _greedy_optimize_cpu(
            self,
            weights: Optional[Tuple[float,float]]=(0.5,0.5),
            max_iterations: Optional[int]=1000,
        ) -> Tuple[Dict[int,int], float]:

        # Calculate the initial mean hhi over all routes
        f1 = -(sum(self._hhi.values())*100) / (len(self._hhi)*10000)
        f2 = sum(self._log_bfs_degree.values()) / len(self._log_bfs_degree)
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
                market_shares_community, hhi = self._get_cache_hhi()
                bfs_degree, log_bfs_degree = self._get_cache_bfs_degree()

                # For each pair of communities
                n_pairs = len(self._distinct_communities)**2
                with tqdm(total=n_pairs, desc="Inner Loop", leave=False) as pbar_inner:
                    for community1 in self._distinct_communities:
                        for community2 in self._distinct_communities:
                            if community1 != community2:
                                # Calculate the mean hhi if these two communities were to be merged
                                self.update_scores(community1,community2)

                                f1 = -(sum(self._hhi.values())*100) / (len(self._hhi)*10000)
                                f2 = sum(self._log_bfs_degree.values()) / len(self._log_bfs_degree)
                                f_new = multiobjective(f1,f2,weights)

                                # Calculate the increase in mean hhi
                                increase = f_new - f

                                # If this pair gives a better increase, update max_increase and best_pair
                                if increase > max_increase:
                                    max_increase = increase
                                    best_pair = (community1, community2)

                                # Revert the merge
                                self.communities_dict = communities_dict
                                self._set_cache_hhi(market_shares_community, hhi)
                                self._set_cache_bfs_degree(bfs_degree, log_bfs_degree)

                            pbar_inner.update(1)

                # If the maximum increase is greater than 0, merge these two communities and update the mean hhi
                if max_increase > 0:
                    self.update_scores(*best_pair)
                    self.communities_dict = self._communities_dict
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -(sum(self._hhi.values())*100) / (len(self._hhi)*10000)
                    f2 = sum(self._log_bfs_degree.values()) / len(self._log_bfs_degree)
                    self.communities_history.append(communities_dict.copy())
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        # Print the final community structure and the final mean hhi
        print("Final community structure:", self._communities_dict)
        print("Final objective:", f)

        return self.communities_dict, f

    def _update_hhi_scores(self, edges_to_update: Optional[List[Tuple[str,str]]] = None):

        if edges_to_update is None:
            edges_to_update = self.edge_unique_list

        # reset selected edges
        self._cond_probs_community_sparse = []
        for u, v in edges_to_update:
            self._cond_probs_community[(u, v)] = defaultdict(float)

        for u, v in edges_to_update:
            data_list = self.G.get_edge_data(u,v)
            for key,data in data_list.items():
                # Accumulate the squares for each route
                self._cond_probs_community[(u, v)][self._communities_dict[data['edge_type']]] += data['market_share_percent']
                self._cond_probs_community_sparse.append([self._communities_dict[data['edge_type']],data['market_share_percent']])

        # Recalculate the community hhi for each affected route
        for edge, communities in self._cond_probs_community.items():
            self._hhi[edge] = sum(market_share ** 2 for market_share in communities.values())

    def _update_bfs_scores(self, attr_to_update: Optional[List[str]] = None):

        if attr_to_update is None:
            attr_to_update = self.attr_list

        # Recalculate scores for the affected airlines
        for attr in attr_to_update:
            target_community = self._communities_dict[attr]

            # Get edges by 'edge_type' from the pre-grouped dictionary
            edges = self.edge_attr2keypair_list[attr]

            # Get 'to' attr
            to_attr = {v for _, v, _ in edges}

            # Calculate weights
            if len(to_attr) > 0:
                edges = set(sum((self.edge_u2pairkey_list[attr] for attr in to_attr), []))
                bfs = [self.edge_pairkey2atrr[(u,v,key)]['weight'] for u,v,key in edges if self._communities_dict[self.edge_pairkey2atrr[(u,v,key)]['edge_type']] == target_community]

                self._bfs_degree[attr] = sum(bfs) + self.attr_key2w[attr]

                if self.attr_key2w[attr] > 0:
                    self._log_bfs_degree[attr] = np.log(self._bfs_degree[attr])


class StochasticCommunity(BaseCommunity):

    def __init__(
            self,
            MG: MultiDiGraph,
            communities_dict: Dict[str, List[str]] = None,
            seed: int = None,
            use_gpu: bool | None = False,
            n_samples_edge: Optional[int] = 100,
            walk_length: Optional[int] = 10,
        ) -> None:
        super().__init__(MG, communities_dict, seed, use_gpu)

        self.n_samples_edge = n_samples_edge
        self.walk_length = walk_length

    def init_gpu(
            self,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            tile_size: Optional[int] = 42,
            block_size: Optional[int] = 24
        ):
        super().init_gpu(threads_per_block_2D,threads_per_block_3D)

        from .utilities_cuda import define_cuda_kernels_stochastic
        from ..utilities_cuda import pairwise_product, elementwise_divide, create_xoroshiro128p_states

        self.pairwise_product: Callable = pairwise_product
        self.elementwise_divide: Callable = elementwise_divide

        # move constants to memory
        self._adj_dense_d = self.cuda.to_device(self._adj_dense)
        self._adj3_d = self.cuda.to_device(self._adj3)
        self._adj_dense_index_d = self.cuda.to_device(self._adj_dense_index)

        node_spatial_list = np.array(list(self.node_spatial_list),np.int32)
        node_spatial_degree = np.array(list(self.node_spatial_degree),np.float32)

        self._node_spatial_list_d = self.cuda.to_device(node_spatial_list)
        self._node_spatial_degree_d = self.cuda.to_device(node_spatial_degree)

        kernels = define_cuda_kernels_stochastic(self.n_attr,self.n_nodes_spatial, tile_size, block_size)

        self.set_membership                     : Callable = kernels[0]
        self.spatial_random_walk                : Callable = kernels[1]
        self.weighted_spatial_random_walk       : Callable = kernels[2]
        self.attr_sampler                       : Callable = kernels[3]
        self.attr_source_postprocess            : Callable = kernels[4]
        self.edge_sampler                       : Callable = kernels[5]
        self.init_edge_counts                   : Callable = kernels[6]
        self.edge_postprocess                   : Callable = kernels[7]
        self.load_counts                        : Callable = kernels[8]
        self.edge_histogram                     : Callable = kernels[9]
        self.copy_reduced_counts                : Callable = kernels[10]
        self.init_attr                          : Callable = kernels[11]
        self.attr_postprocess                   : Callable = kernels[12]

        self.create_xoroshiro128p_states: Callable = create_xoroshiro128p_states

        # random walk data
        self._edge_samples_d = self.cuda.device_array((self.n_unique_edges,self.n_samples_edge), dtype=np.int32)
        self._walks_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.walk_length), dtype=np.int32)
        self._walks_edges_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.int32)
        self._walks_attr_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.int32)
        self._walks_attr_cp_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.float32)
        self._source_attr_counts_d = self.cuda.device_array((self.n_nodes_spatial,self.n_attr), dtype=np.int32)

        self._randomwalk_gpu()

        # Calculate the initial mean hhi over all routes
        self._update_scores_gpu(0,0)
        self._update_scores_community()

    def _randomwalk_gpu(self,testing = False):

        #################################
        # perform random walks


        cumsum_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.n_nodes_spatial,),dtype=np.float32)
        weights_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.n_nodes_spatial,),dtype=np.float32)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_attr), TPB = self.TPB_2D)

        self.init_attr[blockspergrid,threadsperblock](self._source_attr_counts_d,)

        # Define the thread and block dimensions
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_samples_edge), TPB = self.TPB_2D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.edge_sampler[blockspergrid,threadsperblock](
            self.n_samples_edge,
            self.n_unique_edges,
            self._adj3_d,
            self._edge_samples_d,
            cuda_random_state,
        )

        # Define the thread and block dimensions
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_samples_edge), TPB = self.TPB_2D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.spatial_random_walk[blockspergrid,threadsperblock](
            self.n_samples_edge,
            self.walk_length,
            self._node_spatial_list_d,
            self._adj_dense_d,
            self._adj_dense_index_d,
            self._walks_d,
            self._walks_edges_d,
            cuda_random_state,
            cumsum_d,
            weights_d
        )

        self.cuda.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_samples_edge, self.walk_length-1), TPB = self.TPB_3D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.attr_sampler[blockspergrid,threadsperblock](
            self.n_samples_edge,
            self.walk_length,
            self._walks_edges_d,
            self._adj_d,
            self._adj3_d,
            self._walks_attr_cp_d,
            self._walks_attr_d,
            cuda_random_state
        )

        self.cuda.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_samples_edge), TPB = self.TPB_2D)
        self.attr_source_postprocess[blockspergrid,threadsperblock](
            self.n_samples_edge,
            self._walks_attr_d,
            self._source_attr_counts_d
        )

        self.cuda.synchronize()

        if testing:
            return self._edge_samples_d, self._walks_d, self._walks_edges_d, self._walks_attr_d, self._walks_attr_cp_d, self._source_attr_counts_d

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
            self._node_spatial_degree_d,
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
            stream,
            testing = False,
            parallel_reduce = False
        ):

        #################################
        # merge values

        reduced_counts_d = self.cuda.device_array((self.n_unique_edges,self.n_communities), dtype=np.int32, stream=stream)
        cum_weight_d = self.cuda.device_array((self.n_nodes_spatial,self.n_attr), dtype=np.float32, stream=stream)

        if parallel_reduce:

            edge_counts_d = self.cuda.device_array((self.n_unique_edges,self.n_samples_edge,self.n_communities), dtype=np.int16, stream=stream)

            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_samples_edge), TPB = self.TPB_2D)

            self.load_counts[blockspergrid,threadsperblock,stream](
                community_i,
                community_j,
                self.n_communities,
                self.n_samples_edge,
                self.n_unique_edges,
                edge_samples_d,
                edge_counts_d,
                membership_pairs_d
            )

            stream.synchronize()

            # Kernel launch
            self.edge_histogram[blockspergrid,threadsperblock,stream](
                self.n_communities,
                self.n_samples_edge,
                self.n_unique_edges,
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
                    edge_counts_d
                )

            stream.synchronize()

            # Transfer the first row of the matrix to the vector
            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_communities), TPB = self.TPB_2D)

            self.copy_reduced_counts[blockspergrid,threadsperblock,stream](
                self.n_communities,
                self.n_unique_edges,
                edge_counts_d,
                reduced_counts_d
            )

            stream.synchronize()

        else:

            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_communities), TPB = self.TPB_2D)

            self.init_edge_counts[blockspergrid,threadsperblock,stream](
                self.n_communities,
                self.n_unique_edges,
                reduced_counts_d,
            )

            stream.synchronize()

            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_samples_edge), TPB = self.TPB_2D)

            self.edge_postprocess[blockspergrid,threadsperblock,stream](
                community_i,
                community_j,
                self.n_samples_edge,
                self.n_unique_edges,
                edge_samples_d,
                reduced_counts_d,
                membership_pairs_d
            )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_attr), TPB = self.TPB_2D)

        self.init_attr[blockspergrid,threadsperblock,stream](cum_weight_d,)

        stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_samples_edge), TPB = self.TPB_2D)

        self.attr_postprocess[blockspergrid,threadsperblock,stream](
            community_i,
            community_j,
            self.n_samples_edge,
            self.walk_length,
            walks_nodes_weights_d,
            walks_nodes_d,
            source_counts_d,
            cum_weight_d,
            membership_pairs_d
        )

        stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, self.n_samples_edge), TPB = self.TPB_2D)

        self.elementwise_divide[blockspergrid,threadsperblock,stream](
            cum_weight_d,
            source_counts_d
        )

        stream.synchronize()

        edge_samples_h = edge_samples_d.copy_to_host()

        walks_nodes_h = walks_nodes_d.copy_to_host()
        walks_nodes_weights_h = walks_nodes_weights_d.copy_to_host()

        edge_counts_h = reduced_counts_d.copy_to_host() / self.n_samples_edge
        cum_weight_h = cum_weight_d.copy_to_host()

        hhi_estimate = edge_counts_h**2
        hhi_estimate = np.nansum(hhi_estimate,axis=1)

        bfs_estimate = cum_weight_h
        bfs_estimate = np.nansum(bfs_estimate,axis=0)
        log_bfs = np.where(bfs_estimate > 0.0, np.log(bfs_estimate), 0.0)

        if testing:
            return edge_samples_h, walks_nodes_h, walks_nodes_weights_h, edge_counts_h, cum_weight_h, hhi_estimate, bfs_estimate, log_bfs
        else:
            return hhi_estimate, bfs_estimate, log_bfs

    def _calculate_members_gpu(self):

        # iteration constants
        communities_d = self.cuda.to_device(self._distinct_communities)
        membership_d = self.cuda.to_device(list((self._communities_dict.values())))

        # iteration values
        membership_pairs_d = self.cuda.device_array((self.n_communities, self.n_communities, self.n_attr), dtype=np.int16)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_communities, self.n_communities, self.n_attr), TPB = self.TPB_3D)
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
            weights,
            f,
            parallel_reduce
        ):

        community_i = batch_index // self.n_communities
        community_j = batch_index % self.n_communities

        if community_i > community_j:
            stream = streams[batch_index % len(streams)]

            # Perform stochastic evaluation using the selected stream
            hhi_estimate, bfs_estimate, log_bfs = self._stochastic_eval_gpu(
                community_i,
                community_j,
                edge_samples_d,
                walks_nodes_weights_d,
                walks_nodes_d,
                source_counts_d,
                membership_pairs_d,
                stream=stream,
                parallel_reduce = parallel_reduce
            )

            f1 = -np.mean(hhi_estimate) * 100
            f2 = np.mean(log_bfs)
            f_new = multiobjective(f1, f2, weights=weights)

            increase = f_new - f

            return increase, community_i, community_j, hhi_estimate, bfs_estimate

        return 0, community_i, community_j, None, None

    def _update_scores_gpu(self,community_i,community_j):

        stream = self.cuda.stream()

        membership_pairs_d = self._calculate_members_gpu()

        hhi_estimate, bfs_estimate, log_bfs = self._stochastic_eval_gpu(
            community_i,
            community_j,
            self._edge_samples_d,
            self._walks_attr_cp_d,
            self._walks_attr_d,
            self._source_attr_counts_d,
            membership_pairs_d,
            stream,
            testing = False,
            parallel_reduce = False
        )

        # Recalculate the community hhi for each affected route
        for i, hhi in enumerate(hhi_estimate):
            self._hhi[self.edge_ui2pair[i]] = hhi

        for i, log_bfs in enumerate(log_bfs):
            self._log_bfs_degree[self.attr_i2key[i]] = log_bfs

        return hhi_estimate, bfs_estimate, log_bfs

    def _greedy_optimize_gpu(
            self,
            weights: Optional[Tuple[float,float]]=(0.5,0.5),
            max_iterations: Optional[int]=1000,
            num_streams: Optional[int] = 1,
            testing_opt: Optional[bool] = False,
            parallel_reduce: Optional[bool] = False
        ) -> Tuple[Dict[int,int], float]:

        f1 = -(sum(self._hhi.values())*100) / (len(self._hhi)*10000)
        f2 = sum(self._log_bfs_degree.values()) / len(self._log_bfs_degree)
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

                if testing_opt:
                    hhi_pairs_h = np.zeros((self.n_communities, self.n_communities, self.n_unique_edges), dtype=np.float32)
                    bfs_pairs_h = np.zeros((self.n_communities, self.n_communities, self.n_attr), dtype=np.float32)

                membership_pairs_d = self._calculate_members_gpu()

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
                            self._edge_samples_d,
                            self._walks_attr_cp_d,
                            self._walks_attr_d,
                            self._source_attr_counts_d,
                            membership_pairs_d,
                            weights,
                            f,
                            parallel_reduce
                        )
                        futures.append(future)

                    # Collect results and update max_increase and best_pair
                    for future in tqdm(futures, total=num_total_batches, desc="merges processed"):
                        increase, community_i, community_j, hhi_estimate, bfs_estimate = future.result()

                        if community_i > community_j:
                            if increase > max_increase:
                                max_increase = increase
                                best_pair = (community_i, community_j)

                            if testing_opt:
                                hhi_pairs_h[community_i,community_j,:] = hhi_estimate
                                bfs_pairs_h[community_i,community_j,:] = bfs_estimate

                # Synchronize all streams
                for stream in streams:
                    stream.synchronize()

                # If the maximum increase is greater than 0, merge these two communities and update the mean hhi
                if max_increase > 0:
                    self._update_scores_gpu(*best_pair)
                    self.communities_dict = self._communities_dict
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -(sum(self._hhi.values())*100) / (len(self._hhi)*10000)
                    f2 = sum(self._log_bfs_degree.values()) / len(self._log_bfs_degree)

                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        if testing_opt:
            return membership_pairs_d.copy_to_host(), hhi_pairs_h, bfs_pairs_h

    def __del__(self):
        if self.use_gpu:
            print("Destructor: Instance destroyed")
            del self._adj_d
            del self._adj_dense_d
            del self._adj3_d
            del self._adj_dense_index_d
            del self._edge_attr_d
            del self._edge_bfs_d
            del self._edges_neighbour_d
            del self._edge_unique_index_d
            del self._cp_sparse_values_d
            del self._cp_sparse_attr_d
            del self._node_spatial_list_d
            del self._node_spatial_degree_d
            del self._edge_samples_d
            del self._walks_d
            del self._walks_edges_d
            del self._walks_attr_d
            del self._walks_attr_cp_d
            del self._source_attr_counts_d
