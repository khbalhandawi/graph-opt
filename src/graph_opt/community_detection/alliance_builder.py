from collections import defaultdict
import networkx as nx
from networkx import MultiDiGraph
import numpy as np
import os
import json
from copy import deepcopy
import sys
import random
import functools
import math
from pdb import set_trace
from scipy.sparse import spmatrix
from typing import Callable, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools

from abc import ABC, abstractmethod

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from .utilities import resort_communities
from .utilities_cpu import set_membership_cpu, edge_sampler_cpu, spatial_random_walk, attr_sampler_cpu, \
    attr_source_postprocess_cpu, attr_weight_cpu, node_weight_cpu, community_weight_cpu, edge_postprocess_cpu, attr_postprocess_cpu
from ..utilities import _default_if_none, dict2array, cpu_gpu_dispatcher, cuda_blocks, safe_log, serialize, deserialize
from ..random_walk.utilities import naive_weighted_choices

from typing import Dict, List, Union, Tuple, Optional, NewType
from ..random_settings import random_state

type_cp = NewType('type_cp', Dict[int,float])
type_cps = NewType('type_cps', Dict[Tuple[int, int], type_cp])
type_hhi = NewType('type_hhi', Dict[Tuple[int, int], Dict[int, float]])
type_bfs = NewType('type_bfs', Dict[int, float])

def multiobjective(f1: Union[float,np.ndarray], f2: Union[float,np.ndarray], weights: Optional[Tuple[float,float]]=(0.5,0.5)):
    return weights[0]*f1 + weights[1]*f2

def load_dict(data):
    return data

def load_tuple_dict(data):
    return defaultdict(float, {tuple(k.strip('()').replace("'", "").split(', ')): v for k, v in data.items()})

def load_np_array(data):
    return np.array(data)

dtype_load_func = {
    Dict[Tuple[str,str],int]: load_tuple_dict,
    np.ndarray: load_np_array,
    # Add more dtype load functions as needed
}


class BaseCommunity(ABC):

    def _jsonmode(func):
        def wrapper(self, directory='models', **kwargs):
            if not os.path.exists(directory):
                os.makedirs(directory)

            func(self, directory=directory)

        return wrapper

    def __init__(
            self,
            MG: MultiDiGraph,
            communities_dict: Dict[str,List[str]] = None,
            seed: int = None,
            use_gpu: Optional[bool] = False,
            num_streams: Optional[int] = 1
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
        self.edge_pairkey2data = {}
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
        self._init_gpu = False
        self.num_streams = num_streams

        self._parse_graph()
        self._initialize_communities(communities_dict)

    @cpu_gpu_dispatcher('_init_random_cpu', '_init_random_gpu')
    def init_random(self, **kwargs):
        pass
    
    def _init_random_cpu(self, **kwargs):
        pass

    def _init_random_gpu(
            self,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8
        ):
        assert self.use_gpu, "error: use_gpu flag set to False"
        from .utilities_cuda import cuda
        self.cuda = cuda
        self._adj_d = self.cuda.to_device(self._adj)

        self.TPB_1D = threads_per_block_1D
        self.TPB_2D = threads_per_block_2D
        self.TPB_3D = threads_per_block_3D
        self._init_gpu = True

        # Create streams
        self.streams = [self.cuda.stream() for _ in range(self.num_streams)]

    @property
    def communities_dict(self)->Dict[str,int]:
        return {attr: self._communities_dict[attr] for attr in self._communities_dict.keys()}

    @communities_dict.setter
    def communities_dict(self,value:Dict[str,int]):
        sorted2orig, orig2sorted, self._communities_dict, self._distinct_communities, self.n_communities = resort_communities(value)

    @property
    def cond_probs_community(self) -> type_cps:
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

    def update_scores(self, community_to_merge: int, new_community: int, update_communities: Optional[bool]=True) -> Tuple[List[Tuple[str,str]], List[str]]:
        edges_to_update, attr_to_update = self._update_communities(community_to_merge,new_community,update_communities)
        self.communities_dict = self._communities_dict
        return edges_to_update, attr_to_update

    def _update_communities(self, community_to_merge: int, new_community: int, update_communities: bool) -> Tuple[List[Tuple[str,str]], List[str]]:
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

    def _update_community_weights(self):
        self.bfs_degree_community = np.zeros(self.n_communities)
        for i, v in self.bfs_degree.items():
            self.bfs_degree_community[self._communities_dict[i]] += v

        self._sampled_communities_pairs = set()

        # populate excluded pairs
        for i in range(self.n_communities):
            for j in range(self.n_communities):
                if i >= j:
                    self._sampled_communities_pairs.add((i,j))

                weight_i = self.bfs_degree_community[i]
                weight_j = self.bfs_degree_community[j]

                # pop empty attributes
                if weight_i == 0.0 or weight_j == 0.0:
                    self._sampled_communities_pairs.add((i,j))

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
            self.edge_pairkey2data[(u, v, key)] = data
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
            n_not_defined = len(communities_dict)
            _communities_dict = {}
            for i,attr in enumerate(self.attr_list):
                value = communities_dict.get(attr,None)

                if value is not None:
                    _communities_dict[attr] = value
                else:
                    _communities_dict[attr] = n_not_defined
                    n_not_defined += 1

            self.communities_dict = _communities_dict

    def _sample_community_pairs(self, n_samples: int = 5, epsilon: float = 0.1, replace: Optional[bool] = False) -> Union[list,bool]:

        rng_np = self.random_state[1]
        all_samples = list(itertools.product(range(self.n_communities), range(self.n_communities)))

        sample_set = set(all_samples) - self._sampled_communities_pairs
        sample_list = list(sample_set)  # convert set to list

        if len(sample_list) == 0:
            return False

        probabilities = self.bfs_degree_community[sample_list] / np.sum(self.bfs_degree_community)
        probabilities_pairs = probabilities[:,0] * probabilities[:,1]
        probabilities_pairs = probabilities_pairs / np.sum(probabilities_pairs)

        if not replace:
            n_samples = min(len(sample_list),n_samples)

        if len(sample_list) == 0:
            return False

        n_samples_is = math.ceil(n_samples * (1-epsilon))
        n_samples_uni = n_samples - n_samples_is
        is_indices = rng_np.choice(len(sample_list), n_samples_is, p=probabilities_pairs, replace=replace)
        sample_is_pairs = [sample_list[i] for i in is_indices]
        if not replace:
            self._sampled_communities_pairs |= set(sample_is_pairs)

        sample_set = set(all_samples) - self._sampled_communities_pairs
        sample_list = list(sample_set)  # convert set to list

        if not replace:
            n_samples_uni = min(len(sample_list),n_samples_uni)

        uni_indices = rng_np.choice(len(sample_list), n_samples_uni, p=None, replace=replace)
        sample_uni_pairs = [sample_list[i] for i in uni_indices]
        if not replace:
            self._sampled_communities_pairs |= set(sample_uni_pairs)


        return sample_is_pairs + sample_uni_pairs

    def __del__(self):
        if self.use_gpu and self._init_gpu:
            # print("Destructor: Instance destroyed")
            del self._adj_d

    @_jsonmode
    def save(self, directory=None, **kwargs):
        data_to_save = {
            'communities_dict': self._communities_dict,
            'log_bfs_degree': self._log_bfs_degree,
            'bfs_degree': self._bfs_degree,
            'hhi': {str(k): v for k, v in self._hhi.items()},
            'bfs_degree_community': self.bfs_degree_community.tolist() if isinstance(self.bfs_degree_community, np.ndarray) else self.bfs_degree_community,
            'communities_history': self.communities_history,
            'f_history': self.f_history,
            'multi_history': self.multi_history
        }

        for name, data in data_to_save.items():
            with open(f'{directory}/{name}.json', 'w') as f:
                json.dump(serialize(data), f, indent=4)

    @_jsonmode
    def load(self, directory=None, **kwargs):
        data_to_load = {
            'communities_dict':     {"name" : '_communities_dict', "dtype" : dict},
            'log_bfs_degree':       {"name" : '_log_bfs_degree', "dtype" : dict},
            'bfs_degree':           {"name" : '_bfs_degree', "dtype" : dict},
            'hhi':                  {"name" : '_hhi', "dtype" : Dict[Tuple[str,str],int]},
            'bfs_degree_community': {"name" : 'bfs_degree_community', "dtype" : np.ndarray},
            'communities_history':  {"name" : 'communities_history', "dtype" : List[Dict[str,int]]},
            'f_history':            {"name" : 'f_history', "dtype" : list},
            'multi_history':        {"name" : 'multi_history', "dtype" : List[Tuple[float,float]]},
        }

        for name, data_dict in data_to_load.items():
            with open(f'{directory}/{name}.json', 'r') as f:
                data = deserialize(json.load(f))
                dtype = data_dict["dtype"]
                data = dtype_load_func.get(dtype, load_dict)(data)
                setattr(self, data_dict["name"], data)

class DeterministicCommunity(BaseCommunity):

    def __init__(
            self, 
            MG: MultiDiGraph, 
            communities_dict: Dict[str, List[str]] = None, 
            seed: int = None, 
            use_gpu: bool | None = False,
            num_streams: Optional[int] = 1
        ) -> None:
        super().__init__(MG, communities_dict, seed, use_gpu, num_streams)

        self._update_hhi_scores()
        self._update_bfs_scores()
        self._update_community_weights()

    def _init_random_gpu(
            self,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
        ):
        super()._init_random_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D)

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

    def update_scores(self, community_to_merge: int, new_community: int, update_communities: bool | None = True) -> Tuple[List[Tuple[str]] | List[str]]:
        edges_to_update, attr_to_update = super().update_scores(community_to_merge, new_community, update_communities)
        self._update_hhi_scores(edges_to_update)
        self._update_bfs_scores(attr_to_update)
        self._update_community_weights()

    def _greedy_optimize_gpu(
            self,
            weights: Optional[Tuple[float,float]]=(0.5,0.5),
            max_iterations: Optional[int]=1000,
            batch_size: Optional[Union[int,Tuple[int,int]]] = 2,
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
        with tqdm(total=max_iterations, desc="Outer Loop", position=0, leave=False) as pbar_outer:
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
                num_streams = min(self.num_streams, num_batches_i, num_batches_j)

                # Initialize host arrays to store results (adjust the size based on the actual number of batches)
                best_pairs = np.zeros((num_batches_i, num_batches_j, 2), dtype=np.int32)
                hhi_pairs = np.zeros((num_batches_i, num_batches_j, self.n_unique_edges), dtype=np.float32)
                bfs_pairs = np.zeros((num_batches_i, num_batches_j, self.n_attr), dtype=np.float32)
                score_pairs = np.zeros((num_batches_i, num_batches_j, self.n_attr), dtype=np.float32)
                membership_pairs = np.zeros((num_batches_i, num_batches_j, self.n_attr), dtype=np.int32)

                # Batch processing with streams
                num_total_batches = num_batches_i * num_batches_j
                with tqdm(total=num_total_batches, desc="Batches processed", position=1, leave=False) as pbar_inner:
                    for batch_index in range(num_total_batches):
                        batch_i = batch_index // num_batches_j
                        batch_j = batch_index % num_batches_j

                        start_idx_i = batch_i * batch_size_i
                        end_idx_i = min((batch_i + 1) * batch_size_i, self.n_communities)

                        start_idx_j = batch_j * batch_size_j
                        end_idx_j = min((batch_j + 1) * batch_size_j, self.n_communities)

                        # Select a stream for this batch
                        stream = self.streams[(batch_i * num_batches_j + batch_j) % num_streams]

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
                        score_pairs_h = safe_log(bfs_pairs_h)
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
                for stream in self.streams:
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
                if max_increase > 0:
                    self.update_scores(*best_pair)
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
        with tqdm(total=max_iterations, desc="Outer Loop", position=0, leave=False) as pbar_outer:
            while improvement and n_iterations < max_iterations:
                improvement = False
                max_increase = 0
                best_pair = None

                # cache current partition
                communities_dict = self.communities_dict
                cond_probs_community, hhi = self._get_cache_hhi()
                bfs_degree, log_bfs_degree = self._get_cache_bfs_degree()

                # For each pair of communities
                n_pairs = len(self._distinct_communities)**2
                with tqdm(total=n_pairs, desc="Inner Loop", position=1, leave=False) as pbar_inner:
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
                                self._set_cache_hhi(cond_probs_community, hhi)
                                self._set_cache_bfs_degree(bfs_degree, log_bfs_degree)

                            pbar_inner.update(1)

                # If the maximum increase is greater than 0, merge these two communities and update the mean hhi
                if max_increase > 0:
                    self.update_scores(*best_pair)
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
                bfs = [self.edge_pairkey2data[(u,v,key)]['weight'] for u,v,key in edges if self._communities_dict[self.edge_pairkey2data[(u,v,key)]['edge_type']] == target_community]

                self._bfs_degree[attr] = sum(bfs) + self.attr_key2w[attr]

                if self.attr_key2w[attr] > 0:
                    self._log_bfs_degree[attr] = np.log(self._bfs_degree[attr])

    def __del__(self):
        super().__del__()

        if self.use_gpu and self._init_gpu:
            del self._edge_attr_d
            del self._edge_weights_d
            del self._edges_neighbour_d
            del self._edge_unique_index_d
            del self._cp_sparse_values_d
            del self._cp_sparse_attr_d


class StochasticCommunity(BaseCommunity):

    def __init__(
            self,
            MG: MultiDiGraph,
            communities_dict: Dict[str, List[str]] = None,
            seed: int = None,
            use_gpu: bool | None = False,
            n_samples_edge: Optional[int] = 100,
            walk_length: Optional[int] = 10,
            num_streams: Optional[int] = 1
        ) -> None:
        super().__init__(MG, communities_dict, seed, use_gpu, num_streams)

        self.n_samples_edge = n_samples_edge
        self.walk_length = walk_length

        self._walks = np.zeros((self.n_nodes_spatial,self.n_samples_edge,self.walk_length), dtype=np.int32)
        self._walks_edges = np.zeros((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.int32)
        self._walks_attr = np.zeros((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.int32)
        self._walks_attr_cp = np.zeros((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.float32)
        self._source_attr_counts = np.zeros((self.n_nodes_spatial,self.n_attr), dtype=np.int32)
        self._attr_weights = np.zeros((self.n_nodes_spatial,self.n_attr), dtype=np.float32)
        self._node_weights = np.zeros((self.n_nodes_spatial), dtype=np.float32)

    def _init_random_cpu(self, **kwargs):
        super()._init_random_cpu(**kwargs)

        self.distributions_changed_streams = []
        self.cum_weight_streams = []
        self.reduced_counts_streams = []
        for _ in range(self.num_streams):
            self.distributions_changed_streams += [np.zeros(self.n_unique_edges, dtype=np.int32)]
            self.cum_weight_streams += [np.zeros((self.n_nodes_spatial,self.n_attr), dtype=np.float32)]
            self.reduced_counts_streams += [np.zeros((self.n_unique_edges,self.n_communities), dtype=np.float32)]

        self._randomwalk_cpu()
        self.update_scores(0,0)

    def _init_random_gpu(
            self,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            tile_size: Optional[int] = 42,
            block_size: Optional[int] = 24
        ):
        super()._init_random_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D)

        from .utilities_cuda import define_cuda_kernels_stochastic
        from ..utilities_cuda import pairwise_product, elementwise_divide, create_xoroshiro128p_states

        self.pairwise_product: Callable = pairwise_product
        self.elementwise_divide: Callable = elementwise_divide

        # move constants to memory
        self._adj_dense_d = self.cuda.to_device(self._adj_dense)
        self._adj3_d = self.cuda.to_device(self._adj3)
        self._adj_dense_index_d = self.cuda.to_device(self._adj_dense_index)

        node_spatial_list = np.array([self.node_spatial_key2i[node] for node in self.node_spatial_list],np.int32)
        node_spatial_degree = np.array(list(self.node_spatial_degree),np.float32)

        self._node_spatial_list_d = self.cuda.to_device(node_spatial_list)
        self._node_spatial_degree_d = self.cuda.to_device(node_spatial_degree)

        kernels = define_cuda_kernels_stochastic(self.n_attr, self.n_nodes_spatial, self.n_samples_edge, tile_size, block_size)

        self.set_membership                     : Callable = kernels[0]
        self.spatial_random_walk                : Callable = kernels[1]
        self.attr_sampler                       : Callable = kernels[2]
        self.attr_source_postprocess            : Callable = kernels[3]
        self.edge_sampler                       : Callable = kernels[4]
        self.init_edge_counts                   : Callable = kernels[5]
        self.edge_postprocess                   : Callable = kernels[6]
        self.check_edge_distribution            : Callable = kernels[7]
        self.load_counts                        : Callable = kernels[8]
        self.edge_histogram                     : Callable = kernels[9]
        self.copy_reduced_counts                : Callable = kernels[10]
        self.init_attr                          : Callable = kernels[11]
        self.init_node_weights                  : Callable = kernels[12]
        self.init_community                     : Callable = kernels[13]
        self.node_weight                        : Callable = kernels[14]
        self.attr_weight                        : Callable = kernels[15]
        self.community_weight                   : Callable = kernels[16]
        self.attr_postprocess                   : Callable = kernels[17]
        
        self.create_xoroshiro128p_states: Callable = create_xoroshiro128p_states

        # random walk data
        self._edge_samples_d = self.cuda.device_array((self.n_unique_edges,self.n_samples_edge), dtype=np.int32)
        self._walks_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.walk_length), dtype=np.int32)
        self._walks_edges_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.int32)
        self._walks_attr_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.int32)
        self._walks_attr_cp_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.walk_length-1), dtype=np.float32)
        self._source_attr_counts_d = self.cuda.device_array((self.n_nodes_spatial,self.n_attr), dtype=np.int32)
        self._attr_weights_d = self.cuda.device_array((self.n_nodes_spatial,self.n_attr), dtype=np.float32)
        self._node_weights_d = self.cuda.device_array((self.n_nodes_spatial), dtype=np.float32)

        self.distributions_changed_streams = []
        self.cum_weight_streams = []
        self.community_weight_streams = []
        self.reduced_counts_streams = []
        for stream in self.streams:
            self.distributions_changed_streams += [self.cuda.device_array(self.n_unique_edges, dtype=np.int32, stream=stream)]
            self.cum_weight_streams += [self.cuda.device_array((self.n_nodes_spatial,self.n_attr), dtype=np.float32, stream=stream)]
            self.community_weight_streams += [self.cuda.device_array((self.n_nodes_spatial,self.n_communities), dtype=np.float32, stream=stream)]
            self.reduced_counts_streams += [self.cuda.device_array((self.n_unique_edges,self.n_communities), dtype=np.float32, stream=stream)]

        self._randomwalk_gpu()
        self.update_scores(0,0)

    def update_scores(
            self,
            community_to_merge: int,
            new_community: int,
            reduced_counts_base: Optional[np.ndarray] = None,
            cum_weight_base: Optional[np.ndarray] = None,
            stream_id: Optional[int] = 0
        ):
        super().update_scores(community_to_merge, new_community)
        if self.use_gpu and self._init_gpu:
            hhi_estimate, bfs_estimate, log_bfs = self._update_scores_gpu(
                reduced_counts_base_d = reduced_counts_base, 
                cum_weight_base_d = cum_weight_base, 
                stream_id = stream_id
            )
            self._update_community_weights()
        else:
            hhi_estimate, bfs_estimate, log_bfs = self._update_scores_cpu(
                reduced_counts_base = reduced_counts_base, 
                cum_weight_base = cum_weight_base,
                stream_id = stream_id
            )
            self._update_community_weights()

        return hhi_estimate, bfs_estimate, log_bfs

    def _randomwalk_cpu(
            self,
            testing:Optional[bool] = False
        ):
        """
        Perform `n` random walks of length `length` starting from root nodes `nodes`. 

        Parameters
        ----------
        testing : Optional[bool], optional
            whether to return intermediate values for testing and debugging, by default False
        """
        rng = self.random_state[0]

        self._edge_samples = edge_sampler_cpu(
            n_samples = self.n_samples_edge,
            n_unique_edges = self.n_unique_edges,
            adj3 = self._adj3
        )

        self._walks, self._walks_edges = spatial_random_walk(
            rs = rng,
            adj = self._adj_dense,
            adj_i = self._adj_dense_index,
            n_nodes = self.n_nodes_spatial,
            n_samples = self.n_samples_edge,
            walk_length = self.walk_length,
        )

        self._walks_attr, self._walks_attr_cp = attr_sampler_cpu(
            n_samples = self.n_samples_edge,
            walk_length = self.walk_length,
            walks_edges = self._walks_edges,
            adj = self._adj,
            adj3 = self._adj3,
            n_nodes = self.n_nodes_spatial
        )

        self._source_attr_counts = attr_source_postprocess_cpu(
            n_nodes = self.n_nodes_spatial,
            n_attr = self.n_attr,
            attr_sample = self._walks_attr,
        )

        self._attr_weights = attr_weight_cpu(self._walks_attr, self._walks_attr_cp, self.n_attr)
        self._node_weights = node_weight_cpu(self._walks_attr, self._walks_attr_cp)

        if testing:
            return self._edge_samples, self._walks, self._walks_edges, self._walks_attr, self._walks_attr_cp, self._source_attr_counts, \
                self._attr_weights, self._node_weights

    def _stochastic_eval_cpu(
            self,
            community_i,
            community_j,
            edge_samples,
            walks_nodes_weights,
            walks_nodes,
            membership_pairs,
            stream_id = 0,
            testing = False,
            reduced_counts = None,
            cum_weight = None
        ):

        #################################
        # merge values

        reduced_counts_h = edge_postprocess_cpu(
            ci = community_i,
            cj = community_j,
            samples = edge_samples,
            membership = membership_pairs,
            n_communities = self.n_communities
        )

        community_weight = community_weight_cpu(
            ci = community_i, 
            cj = community_j, 
            samples = walks_nodes, 
            samples_weights = walks_nodes_weights,
            membership = membership_pairs,
            n_communities = self.n_communities
        )

        cum_weight_h = attr_postprocess_cpu(
            ci = community_i,
            cj = community_j,
            attr_weight = self._attr_weights,
            community_weight = community_weight,
            node_weight = self._node_weights,
            membership = membership_pairs,
            n_nodes = self.n_nodes_spatial,
            n_attr = self.n_attr
        )

        # Initialize the cum_weight_norm array with zeros (to avoid underflows)
        cum_weight_norm = cum_weight_h

        edge_counts = reduced_counts_h / self.n_samples_edge

        hhi_estimate = edge_counts**2
        hhi_estimate = np.nansum(hhi_estimate,axis=1)

        bfs_estimate = cum_weight_norm
        bfs_estimate = np.nansum(bfs_estimate,axis=0) / bfs_estimate.shape[0]
        log_bfs = safe_log(bfs_estimate)

        # update record
        if reduced_counts is None and cum_weight is None:
            np.copyto(self.reduced_counts_streams[stream_id], reduced_counts_h)
            np.copyto(self.cum_weight_streams[stream_id], cum_weight_h)
        else:
            np.copyto(reduced_counts, reduced_counts_h)
            np.copyto(cum_weight, cum_weight_h)       

        if testing:
            return edge_samples, walks_nodes, walks_nodes_weights, community_weight, edge_counts, cum_weight_norm, hhi_estimate, bfs_estimate, log_bfs
        else:
            return hhi_estimate, bfs_estimate, log_bfs

    def _calculate_members_cpu(self):

        # iteration constants
        communities = np.array(self._distinct_communities)
        membership = np.array(list(self._communities_dict.values()))

        # iteration values
        membership_pairs = set_membership_cpu(
            communities = communities,
            membership = membership,
            n_attr = self.n_attr
        )
        return membership_pairs

    def _process_batch_cpu(
            self,
            batch_index,
            community_i,
            community_j,
            edge_samples,
            walks_nodes_weights,
            walks_nodes,
            source_counts,
            membership_pairs,
            reduced_counts_base,
            weights,
            f,
        ):

        if community_i < community_j:
            stream_id = batch_index % self.num_streams

            # Perform stochastic evaluation using the selected stream
            hhi_estimate, bfs_estimate, log_bfs = self._stochastic_eval_cpu(
                community_i,
                community_j,
                edge_samples,
                walks_nodes_weights,
                walks_nodes,
                membership_pairs,
                stream_id = stream_id
            )
            f1 = -np.mean(hhi_estimate) * 100
            f2 = np.mean(log_bfs)
            f_new = multiobjective(f1, f2, weights=weights)

            reduced_counts = self.reduced_counts_streams[stream_id]
            # Assuming your matrices are named matrix1 and matrix2
            hhi_new = np.sort(reduced_counts, axis=1)
            hhi_base = np.sort(reduced_counts_base, axis=1)
            distributions_changed = np.any(hhi_new != hhi_base, axis=1)

            increase = f_new - f if np.any(distributions_changed) else 0.0

            return increase, community_i, community_j, hhi_estimate, bfs_estimate

        return 0, community_i, community_j, None, None

    def _update_scores_cpu(self, reduced_counts_base, cum_weight_base, stream_id):

        membership_pairs = self._calculate_members_cpu()

        self.distributions_changed_streams = []
        self.cum_weight_streams = []
        self.reduced_counts_streams = []
        for _ in range(self.num_streams):
            self.distributions_changed_streams += [np.zeros(self.n_unique_edges, dtype=np.int32)]
            self.cum_weight_streams += [np.zeros((self.n_nodes_spatial,self.n_attr), dtype=np.float32)]
            self.reduced_counts_streams += [np.zeros((self.n_unique_edges,self.n_communities), dtype=np.float32)]

        hhi_estimate, bfs_estimate, log_bfs = self._stochastic_eval_cpu(
            0,
            0,
            self._edge_samples,
            self._walks_attr_cp,
            self._walks_attr,
            membership_pairs,
            stream_id = stream_id,
            testing = False,
            reduced_counts = reduced_counts_base,
            cum_weight = cum_weight_base
        )

        # Recalculate the community hhi for each affected route
        for i, hhi in enumerate(hhi_estimate):
            self._hhi[self.edge_ui2pair[i]] = hhi

        for i, (b, log_b) in enumerate(zip(bfs_estimate,log_bfs)):
            self._bfs_degree[self.attr_i2key[i]] = b
            self._log_bfs_degree[self.attr_i2key[i]] = log_b

        return hhi_estimate, bfs_estimate, log_bfs

    def _greedy_optimize_cpu(
            self,
            weights: Optional[Tuple[float,float]] = (0.5,0.5),
            max_iterations: Optional[int] = 1000,
            n_sample_pairs: Optional[int] = None,
            epsilon: Optional[float] = 0.1,
            testing_opt: Optional[bool] = False,
            rtol: Optional[float] = 1e-12,
        ) -> Tuple[Dict[int,int], float]:

        # Calculate the initial mean hhi over all routes
        reduced_counts_base = np.zeros((self.n_unique_edges,self.n_communities), dtype=np.int32)
        cum_weight_base = np.zeros((self.n_nodes_spatial,self.n_attr), dtype=np.float32)

        hhi_estimate, bfs_estimate, log_bfs = self.update_scores(0,0,reduced_counts_base,cum_weight_base)

        f1 = -np.mean(list(self._hhi.values())) * 100
        f2 = np.mean(list(self._log_bfs_degree.values()))
        f = multiobjective(f1,f2,weights)

        # log history
        self.communities_history.append(self.communities_dict)
        self.f_history.append(f)
        self.multi_history.append((f1,f2))

        # Initialize flag for improvement
        improvement = True

        # Loop while improvement is possible
        n_iterations = 0
        with tqdm(total=max_iterations, desc="Outer Loop", position=0, leave=False) as pbar_outer:
            while improvement and n_iterations < max_iterations:

                improvement = False
                max_increase = 0
                best_pair = None

                if testing_opt:
                    hhi_pairs = np.zeros((self.n_communities, self.n_communities, self.n_unique_edges), dtype=np.float32)
                    bfs_pairs = np.zeros((self.n_communities, self.n_communities, self.n_attr), dtype=np.float32)

                membership_pairs = self._calculate_members_cpu()

                # Create streams
                improvement_sample = False
                n_feasible_pairs = self.n_communities**2 - len(self._sampled_communities_pairs)
                n_sample_pairs = _default_if_none(n_sample_pairs,n_feasible_pairs,'n_sample_pairs')

                with tqdm(total=n_feasible_pairs, desc="Inner Loop", position=1, leave=False) as pbar_inner:
                    while True:
                        sample_pairs = self._sample_community_pairs(n_sample_pairs, epsilon)

                        if not sample_pairs:
                            break

                        # Use ThreadPoolExecutor to parallelize the loop
                        with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
                            futures = []
                            for batch_index, (community_i, community_j) in enumerate(sample_pairs):
                                future = executor.submit(
                                    self._process_batch_cpu,
                                    batch_index,
                                    community_i,
                                    community_j,
                                    self._edge_samples,
                                    self._walks_attr_cp,
                                    self._walks_attr,
                                    self._source_attr_counts,
                                    membership_pairs,
                                    reduced_counts_base,
                                    weights,
                                    f
                                )
                                futures.append(future)

                            # Collect results and update max_increase and best_pair
                            for future in futures:
                                increase, community_i, community_j, hhi_estimate, bfs_estimate = future.result()
                                pbar_inner.update(1)
                                if community_i < community_j:
                                    if increase - max_increase > rtol:
                                        max_increase = increase
                                        best_pair = (community_i, community_j)
                                        improvement_sample = True

                                    if testing_opt:
                                        hhi_pairs[community_i,community_j,:] = hhi_estimate
                                        bfs_pairs[community_i,community_j,:] = bfs_estimate

                        if improvement_sample:
                            break

                # If the maximum increase is greater than 0, merge these two communities and update the mean hhi
                if max_increase > 0:

                    self.n_communities -= 1
                    reduced_counts_base = np.zeros((self.n_unique_edges,self.n_communities), dtype=np.int32)
                    cum_weight_base = np.zeros((self.n_nodes_spatial,self.n_attr), dtype=np.float32)
                    hhi_estimate, bfs_estimate, log_bfs = self.update_scores(*best_pair, reduced_counts_base, cum_weight_base)
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -np.mean(list(self._hhi.values())) * 100
                    f2 = np.mean(list(self._log_bfs_degree.values()))

                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        if testing_opt:
            return membership_pairs, hhi_pairs, bfs_pairs

    def _randomwalk_gpu(self,testing = False):

        #################################
        # perform random walks


        cumsum_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.n_nodes_spatial,),dtype=np.float32)
        weights_d = self.cuda.device_array((self.n_nodes_spatial,self.n_samples_edge,self.n_nodes_spatial,),dtype=np.float32)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_attr), TPB = self.TPB_2D)

        self.init_attr[blockspergrid,threadsperblock](self._source_attr_counts_d,)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial,), TPB = self.TPB_1D)

        self.init_node_weights[blockspergrid,threadsperblock](self._node_weights_d,)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial,self.n_attr), TPB = self.TPB_2D)

        self.init_community[blockspergrid,threadsperblock](self.n_attr,self._attr_weights_d)

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

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_samples_edge), TPB = self.TPB_2D)
        self.attr_weight[blockspergrid,threadsperblock](
            self.n_samples_edge, 
            self.walk_length, 
            self.n_attr, 
            self._walks_attr_d, 
            self._walks_attr_cp_d, 
            self._attr_weights_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_samples_edge), TPB = self.TPB_2D)
        self.node_weight[blockspergrid,threadsperblock](
            self.n_samples_edge, 
            self.walk_length, 
            self._walks_attr_d, 
            self._walks_attr_cp_d, 
            self._node_weights_d
        )
        
        self.cuda.synchronize()

        if testing:
            return self._edge_samples_d, self._walks_d, self._walks_edges_d, self._walks_attr_d, self._walks_attr_cp_d, self._source_attr_counts_d, \
                self._attr_weights_d, self._node_weights_d

    def _stochastic_eval_gpu(
            self,
            community_i,
            community_j,
            edge_samples_d,
            walks_nodes_weights_d,
            walks_nodes_d,
            membership_pairs_d,
            stream,
            stream_id = 0,
            testing = False,
            parallel_reduce = False,
            reduced_counts_d = None,
            cum_weight_d = None,
            community_weight_d = None,
        ):

        #################################
        # merge values

        reduced_counts_d = _default_if_none(reduced_counts_d, self.reduced_counts_streams[stream_id], "reduced_counts_d")
        cum_weight_d = _default_if_none(cum_weight_d, self.cum_weight_streams[stream_id], "cum_weight_d")
        community_weight_d = _default_if_none(community_weight_d, self.community_weight_streams[stream_id], "community_weight_d")

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

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_communities), TPB = self.TPB_2D)

        self.init_community[blockspergrid,threadsperblock,stream](self.n_communities,community_weight_d)

        stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_samples_edge), TPB = self.TPB_2D)

        self.community_weight[blockspergrid,threadsperblock,stream](
            community_i, 
            community_j, 
            self.n_samples_edge, 
            self.walk_length, 
            self.n_communities,
            self._walks_attr_d, 
            self._walks_attr_cp_d, 
            membership_pairs_d, 
            community_weight_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_nodes_spatial, self.n_attr), TPB = self.TPB_2D)

        self.attr_postprocess[blockspergrid,threadsperblock,stream](
            community_i,
            community_j,
            self.n_attr,
            self._attr_weights_d,
            community_weight_d,
            self._node_weights_d,
            membership_pairs_d,
            cum_weight_d
        )

        stream.synchronize()

        edge_samples_h = edge_samples_d.copy_to_host()

        walks_nodes_h = walks_nodes_d.copy_to_host()
        walks_nodes_weights_h = walks_nodes_weights_d.copy_to_host()

        reduced_counts_h = reduced_counts_d.copy_to_host()
        edge_counts_h = reduced_counts_h / self.n_samples_edge
        cum_weight_h = cum_weight_d.copy_to_host()

        hhi_estimate = edge_counts_h**2
        hhi_estimate = np.nansum(hhi_estimate,axis=1)

        bfs_estimate = cum_weight_h
        bfs_estimate = np.nansum(bfs_estimate,axis=0) / bfs_estimate.shape[0]
        log_bfs = safe_log(bfs_estimate)

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

    def _process_batch_gpu(
            self,
            batch_index,
            community_i,
            community_j,
            edge_samples_d,
            walks_nodes_weights_d,
            walks_nodes_d,
            membership_pairs_d,
            reduced_counts_base_d,
            weights,
            f,
            parallel_reduce
        ):

        if community_i < community_j:
            stream_id = batch_index % len(self.streams)
            stream = self.streams[stream_id]

            # Perform stochastic evaluation using the selected stream
            hhi_estimate, bfs_estimate, log_bfs = self._stochastic_eval_gpu(
                community_i,
                community_j,
                edge_samples_d,
                walks_nodes_weights_d,
                walks_nodes_d,
                membership_pairs_d,
                stream=stream,
                stream_id=stream_id,
                parallel_reduce = parallel_reduce
            )
            f1 = -np.mean(hhi_estimate) * 100
            f2 = np.mean(log_bfs)
            f_new = multiobjective(f1, f2, weights=weights)

            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, ), TPB = self.TPB_1D)

            distributions_changed_d = self.distributions_changed_streams[stream_id]
            reduced_counts_d = self.reduced_counts_streams[stream_id]

            self.check_edge_distribution[threadsperblock,blockspergrid,stream](
                reduced_counts_base_d,
                reduced_counts_d,
                distributions_changed_d
            )

            stream.synchronize()

            distributions_changed = distributions_changed_d.copy_to_host()
            distributions_changed = distributions_changed > 0

            increase = f_new - f if np.any(distributions_changed) else 0.0

            return increase, community_i, community_j, hhi_estimate, bfs_estimate

        return 0, community_i, community_j, None, None

    def _update_scores_gpu(self, reduced_counts_base_d, cum_weight_base_d, stream_id):

        stream = self.cuda.stream()

        membership_pairs_d = self._calculate_members_gpu()

        hhi_estimate, bfs_estimate, log_bfs = self._stochastic_eval_gpu(
            0,
            0,
            self._edge_samples_d,
            self._walks_attr_cp_d,
            self._walks_attr_d,
            membership_pairs_d,
            stream,
            stream_id = stream_id,
            testing = False,
            parallel_reduce = False,
            reduced_counts_d = reduced_counts_base_d,
            cum_weight_d = cum_weight_base_d
        )

        # Recalculate the community hhi for each affected route
        for i, hhi in enumerate(hhi_estimate):
            self._hhi[self.edge_ui2pair[i]] = hhi

        for i, (b, log_b) in enumerate(zip(bfs_estimate,log_bfs)):
            self._bfs_degree[self.attr_i2key[i]] = b
            self._log_bfs_degree[self.attr_i2key[i]] = log_b

        return hhi_estimate, bfs_estimate, log_bfs

    def _greedy_optimize_gpu(
            self,
            weights: Optional[Tuple[float,float]] = (0.5,0.5),
            max_iterations: Optional[int] = 1000,
            n_sample_pairs: Optional[int] = None,
            epsilon: Optional[float] = 0.1,
            testing_opt: Optional[bool] = False,
            parallel_reduce: Optional[bool] = False,
            rtol: Optional[float] = 1e-12,
        ) -> Tuple[Dict[int,int], float]:

        # Calculate the initial mean hhi over all routes
        reduced_counts_base_d = self.cuda.device_array((self.n_unique_edges,self.n_communities), dtype=np.int32)
        cum_weight_base_d = self.cuda.device_array((self.n_nodes_spatial,self.n_attr), dtype=np.float32)
        self.reduced_counts_streams = [self.cuda.device_array((self.n_unique_edges,self.n_communities), dtype=np.int32, stream=stream) for stream in self.streams]

        hhi_estimate, bfs_estimate, log_bfs = self.update_scores(0,0,reduced_counts_base_d,cum_weight_base_d)

        f1 = -np.mean(list(self._hhi.values())) * 100
        f2 = np.mean(list(self._log_bfs_degree.values()))
        f = multiobjective(f1,f2,weights)

        # log history
        self.communities_history.append(self.communities_dict)
        self.f_history.append(f)
        self.multi_history.append((f1,f2))

        # Initialize flag for improvement
        improvement = True

        # Loop while improvement is possible
        n_iterations = 0
        with tqdm(total=max_iterations, desc="Outer Loop", position=0, leave=False) as pbar_outer:
            while improvement and n_iterations < max_iterations:

                improvement = False
                max_increase = 0
                best_pair = None

                if testing_opt:
                    hhi_pairs_h = np.zeros((self.n_communities, self.n_communities, self.n_unique_edges), dtype=np.float32)
                    bfs_pairs_h = np.zeros((self.n_communities, self.n_communities, self.n_attr), dtype=np.float32)

                membership_pairs_d = self._calculate_members_gpu()

                # Create streams
                improvement_sample = False
                n_feasible_pairs = self.n_communities**2 - len(self._sampled_communities_pairs)
                n_sample_pairs = _default_if_none(n_sample_pairs,n_feasible_pairs,'n_sample_pairs')

                with tqdm(total=n_feasible_pairs, desc="Inner Loop", position=1, leave=False) as pbar_inner:
                    while True:
                        sample_pairs = self._sample_community_pairs(n_sample_pairs, epsilon)

                        if not sample_pairs:
                            break

                        # Use ThreadPoolExecutor to parallelize the loop
                        with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
                            futures = []
                            for batch_index, (community_i, community_j) in enumerate(sample_pairs):
                                future = executor.submit(
                                    self._process_batch_gpu,
                                    batch_index,
                                    community_i,
                                    community_j,
                                    self._edge_samples_d,
                                    self._walks_attr_cp_d,
                                    self._walks_attr_d,
                                    membership_pairs_d,
                                    reduced_counts_base_d,
                                    weights,
                                    f,
                                    parallel_reduce
                                )
                                futures.append(future)

                            # Collect results and update max_increase and best_pair
                            for future in futures:
                                increase, community_i, community_j, hhi_estimate, bfs_estimate = future.result()
                                pbar_inner.update(1)
                                if community_i < community_j:
                                    if increase - max_increase > rtol:
                                        max_increase = increase
                                        best_pair = (community_i, community_j)
                                        improvement_sample = True

                                    if testing_opt:
                                        hhi_pairs_h[community_i,community_j,:] = hhi_estimate
                                        bfs_pairs_h[community_i,community_j,:] = bfs_estimate

                        # Synchronize all streams
                        for stream in self.streams:
                            stream.synchronize()

                        if improvement_sample:
                            break

                # If the maximum increase is greater than 0, merge these two communities and update the mean hhi
                if max_increase > 0:
                    del reduced_counts_base_d
                    reduced_counts_base_d = self.cuda.device_array((self.n_unique_edges,self.n_communities-1), dtype=np.int32)
                    cum_weight_base_d = self.cuda.device_array((self.n_nodes_spatial,self.n_attr), dtype=np.float32)
                    self.reduced_counts_streams = [self.cuda.device_array((self.n_unique_edges,self.n_communities-1), dtype=np.int32, stream=stream) for stream in self.streams]
                    hhi_estimate, bfs_estimate, log_bfs = self.update_scores(*best_pair, reduced_counts_base_d, cum_weight_base_d)
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -np.mean(list(self._hhi.values())) * 100
                    f2 = np.mean(list(self._log_bfs_degree.values()))

                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        if testing_opt:
            return membership_pairs_d.copy_to_host(), hhi_pairs_h, bfs_pairs_h

    def __del__(self):
        super().__del__()

        if self.use_gpu and self._init_gpu:
            del self._adj_dense_d
            del self._adj3_d
            del self._adj_dense_index_d
            del self._node_spatial_list_d
            del self._node_spatial_degree_d
            del self._edge_samples_d
            del self._walks_d
            del self._walks_edges_d
            del self._walks_attr_d
            del self._walks_attr_cp_d
            del self._source_attr_counts_d

            for _ in range(self.num_streams):
                distributions_changed = self.distributions_changed_streams.pop()
                cum_weight = self.cum_weight_streams.pop()
                reduced_counts = self.reduced_counts_streams.pop()

                del distributions_changed
                del cum_weight
                del reduced_counts