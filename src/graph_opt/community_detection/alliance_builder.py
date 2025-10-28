from collections import defaultdict
import networkx as nx
from networkx import MultiDiGraph
import numpy as np
import os
import json
from copy import deepcopy
import sys
import random
import math
from typing import Callable, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
from gurobipy import Model, GRB, quicksum, read, Var, GurobiError
import csv
import threading
import time

import logging

from ..utilities import parallel_sampling

from abc import ABC, abstractmethod

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from .utilities import distinct_communities, generate_membership_combinations, generate_possible_memberships
from .utilities_cpu import edge_sampler_cpu, spatial_random_walk, attr_sampler_cpu
from .utilities_cpu import calc_sample_w_i_cpu, calc_sample_w_it_cpu, edge_sample_p_et_cpu, edge_sample_p_ek_cpu, calc_sample_w_ik_cpu, calc_sample_mpc_cpu
from .utilities_cpu import edge_postprocess_cpu, calc_hhi_cpu, calc_w_it_cpu, calc_w_ik_cpu, calc_mpc_cpu, calc_p_iv_cpu, calc_p_iv_t_cpu, calc_p_it_L, calc_p_ik_L
from .utilities_weighted_cpu import (
    calc_s_t_weighted_cpu, calc_s_k_weighted_cpu, calc_s_total_weighted_cpu,
    calc_normalized_p_t_weighted_cpu, calc_normalized_p_k_weighted_cpu,
    calc_mpc_weighted_cpu,
    calc_sample_s_t_weighted_cpu, calc_sample_s_k_weighted_cpu, calc_sample_s_total_weighted_cpu,
    calc_sample_normalized_p_t_weighted_cpu, calc_sample_normalized_p_k_weighted_cpu,
    calc_sample_mpc_weighted_cpu
)
from .utilities_weighted_cuda import define_cuda_kernels_weighted_exact, define_cuda_kernels_weighted_sampling

from ..utilities import _default_if_none, cpu_gpu_dispatcher, cuda_blocks, safe_log, serialize, deserialize

from typing import Dict, List, Union, Tuple, Optional, NewType
from ..random_settings import random_state

__all__ = [
    'GraphLoader',
    'NeighbourSampler',
    'ExactEvaluator',
    'ExactWeightedEvaluator',
    'SampleEvaluator',
    'SampleWeightedEvaluator',
    'GreedyOptimizer',
    'BruteForce',
    'ExactOptimizer'
]

logger = logging.getLogger(__name__)

type_cp = NewType('type_cp', Dict[int,float])
type_cps = NewType('type_cps', Dict[Tuple[int, int], type_cp])
type_hhi = NewType('type_hhi', Dict[Tuple[int, int], Dict[int, float]])
type_mpc = NewType('type_mpc', Dict[int, float])

def multiobjective(f1: Union[float,np.ndarray], f2: Union[float,np.ndarray], weights: Optional[Tuple[float,float]]=(0.5,0.5)):
    return weights[0]*f1 + weights[1]*f2

def load_dict(data):
    return data

def load_tuple_dict(data):
    return defaultdict(float, {tuple(k.strip('()').replace("'", "").split(', ')): v for k, v in data.items()})

def load_np_array(data):
    return np.array(data)

def safe_delete(obj:Dict[str,Any],attr:str):
    if obj.get(attr,None) is not None:
        del obj[attr]

dtype_load_func = {
    Dict[Tuple[str,str],int]: load_tuple_dict,
    np.ndarray: load_np_array,
    # Add more dtype load functions as needed
}

def _jsonmode(func):
    def wrapper(self, directory='models', **kwargs):
        if not os.path.exists(directory):
            os.makedirs(directory)

        func(self, directory=directory)

    return wrapper

def default_float():
    return float()

def default_float_dict():
    return defaultdict(default_float)


class GraphLoader:

    def __init__(
            self,
            MG: MultiDiGraph,
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
        self.edge_pairkey2data = {}
        self.edge_pair2pairkey_list = defaultdict(list)
        self.edge_pairkey2i = {}

        self._cond_probs_attr = defaultdict(default_float_dict)
        self._adj_dense = nx.to_numpy_array(self.G, weight="weight")
        self._adj_dense_index = np.full((self.G.number_of_nodes(), self.G.number_of_nodes()), np.nan)
        self._cp = np.empty(0)
        self._adj_sparse = np.empty(0)
        self._adj3 = np.empty(0)
        self._adj_dense_index = np.empty(0)
        self._adj = np.empty(0)
        self._adj_multi_attr = np.zeros((nx.number_of_edges(self.G),), dtype=np.float32)
        self._adj_multi_attr_type = np.zeros((nx.number_of_edges(self.G),), dtype=np.int32)
        self._adj_multi_attr_index = np.zeros((nx.number_of_edges(self.G),), dtype=np.int32)
        self._adj_multi_attr_sparse = np.zeros((nx.number_of_edges(self.G), 2), dtype=np.int32)

        self.use_gpu = use_gpu
        self._init_gpu = False

        self._parse_graph()

        if self.use_gpu:
            self._post_init_gpu()

    def _post_init_gpu(self):

        assert self.use_gpu, "error: use_gpu flag set to False"
        from .utilities_cuda import cuda
        self.cuda = cuda
        self._cp_d = self.cuda.to_device(self._cp)
        self._adj_d = self.cuda.to_device(self._adj)
        self._adj3_d = self.cuda.to_device(self._adj3)
        self._adj_sparse_d = self.cuda.to_device(self._adj_sparse)
        self._adj_dense_d = self.cuda.to_device(self._adj_dense)
        self._adj_dense_index_d = self.cuda.to_device(self._adj_dense_index)

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
            self._cond_probs_attr[(u, v)][data['edge_type']] += data['market_share_percent'] / 100.0

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
            self.node_spatial_i2key[i] = u
            self.node_spatial_list.append(u)
            self.node_spatial_degree.append(node_degree)

        self.n_nodes_spatial = len(self.node_spatial_list)

        self._adj_sparse = np.zeros((self.n_unique_edges, 2), dtype=np.int32)
        for i, (u,v) in enumerate(self.edge_pair2w.keys()):
            self._adj_sparse[i] = np.array([self.node_spatial_key2i[u], self.node_spatial_key2i[v]])

        adj3 = np.zeros((self.n_unique_edges,self.n_attr))
        cp = np.zeros((self.n_unique_edges,self.n_attr))
        adj = np.zeros((self.n_unique_edges))
        adj_dense_index = np.full((self.n_nodes_spatial, self.n_nodes_spatial), np.nan)
        for i, (edge, attrs) in enumerate(self._cond_probs_attr.items()):
            self.edge_u2pair_list[edge[0]].append(edge)
            self.edge_pair2ui[edge] = i
            self.edge_ui2pair[i] = edge
            adj_dense_index[self.node_spatial_key2i[edge[0]],self.node_spatial_key2i[edge[1]]] = i
            adj[i] = self._adj_dense[self.node_spatial_key2i[edge[0]],self.node_spatial_key2i[edge[1]]]

            for attr in attrs.keys():
                cp[i,self.attr_key2i[attr]] = attrs[attr]

            for u, v, key in self.edge_pair2pairkey_list[edge]:
                data = self.edge_pairkey2data[(u,v,key)]
                attr = data['edge_type']
                weight = data['weight']
                adj3[i,self.attr_key2i[attr]] = weight

            for u,v,key in self.edge_pair2pairkey_list[edge]:
                self.edge_pairkey2ui[(u,v,key)] = i

        adj_multi_attr = np.zeros((nx.number_of_edges(self.G),), dtype=np.float32)
        adj_multi_attr_type = np.zeros((nx.number_of_edges(self.G),), dtype=np.int32)
        adj_multi_attr_index = np.zeros((nx.number_of_edges(self.G),), dtype=np.int32)
        adj_multi_attr_sparse = np.zeros((nx.number_of_edges(self.G), 2), dtype=np.int32)
        for i, (u, v, key, data) in enumerate(self.G.edges(keys=True, data=True)):
            adj_multi_attr[i] = data['weight']
            adj_multi_attr_type[i] = self.attr_key2i[data['edge_type']]
            adj_multi_attr_index[i] = self.edge_pair2ui[(u,v)]
            adj_multi_attr_sparse[i] = np.array([self.node_spatial_key2i[u], self.node_spatial_key2i[v]])


        self._cp = cp
        self._adj3 = adj3
        self._adj = adj
        self._adj_dense_index = adj_dense_index
        self._node_spatial_degree = np.array(self.node_spatial_degree)
        self._adj_multi_attr = adj_multi_attr
        self._adj_multi_attr_type = adj_multi_attr_type
        self._adj_multi_attr_index = adj_multi_attr_index
        self._adj_multi_attr_sparse = adj_multi_attr_sparse

    def __del__(self):
        if self.use_gpu and self._init_gpu:
            # print("Destructor: Instance destroyed")
            del self._cp_d
            del self._adj_d
            del self._adj3_d
            del self._adj_sparse_d
            del self._adj_dense_d
            del self._adj_dense_index_d

    def __getstate__(self):
        state = self.__dict__.copy()

        if self.use_gpu:
            state['_cp'] = self._cp
            state['_adj'] = self._adj
            state['_adj3'] = self._adj3
            state['_adj_dense'] = self._adj_dense
            state['_adj_dense_index'] = self._adj_dense_index

            del state['cuda']
            del state['_cp_d']
            del state['_adj_d']
            del state['_adj3_d']
            del state['_adj_sparse_d']
            del state['_adj_dense_d']
            del state['_adj_dense_index_d']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.use_gpu:
            self._post_init_gpu()


class BaseSampler(ABC):

    def __init__(
            self,
            gloader: GraphLoader,
            seed: int = None,
            use_gpu: bool | None = False,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            stream: Optional[Any] = None
        ) -> None:

        self._gloader = gloader
        self.seed = seed
        self.use_gpu = use_gpu

        self.random_state: Tuple[random.Random,np.random.RandomState] = random_state(self.seed)
        self._post_init(
            threads_per_block_1D=threads_per_block_1D,
            threads_per_block_2D=threads_per_block_2D,
            threads_per_block_3D=threads_per_block_3D,
            stream=stream
        )

    @cpu_gpu_dispatcher('_post_init_cpu', '_post_init_gpu')
    def _post_init(self, **kwargs):
        pass

    def _post_init_cpu(self, **kwargs):
        pass

    def _post_init_gpu(
            self,
            threads_per_block_1D: int,
            threads_per_block_2D: Union[int,Tuple[int,int]],
            threads_per_block_3D: Union[int,Tuple[int,int,int]],
            stream: Optional[Any] = None
        ):
        assert self.use_gpu, "error: use_gpu flag set to False"
        from .utilities_cuda import cuda
        self.cuda = cuda

        self.TPB_1D = threads_per_block_1D
        self.TPB_2D = threads_per_block_2D
        self.TPB_3D = threads_per_block_3D
        self._init_gpu = True

        if stream is None:
            self.stream = self.cuda.stream()
        else:
            self.stream = stream

    def __del__(self):
        pass


class NeighbourSampler(BaseSampler):

    def __init__(
            self,
            gloader: GraphLoader,
            seed: int = None,
            use_gpu: bool | None = False,
            n_samples_edge: Optional[int] = 100,
            n_walks: Optional[int] = 50,
            walk_length: Optional[int] = 10,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
        ) -> None:
        super().__init__(gloader, seed, use_gpu, threads_per_block_1D, threads_per_block_2D, threads_per_block_3D)

        self.n_samples_edge = n_samples_edge
        self.n_walks = n_walks
        self.walk_length = walk_length

        self._V = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length+1), dtype=np.int32)
        self._walks_edges = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.int32)
        self._T = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.int32)
        self._W = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.float32)
        self._W_total = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.float32)

        self._init_sampling()

    @property
    def N_uv(self):
        if self.use_gpu:
            return self._N_uv_d.copy_to_host()
        else:
            return self._N_uv.copy()

    @property
    def V(self):
        if self.use_gpu:
            return self._V_d.copy_to_host()
        else:
            return self._V.copy()

    @property
    def walks_edges(self):
        if self.use_gpu:
            return self._walks_edges_d.copy_to_host()
        else:
            return self._walks_edges.copy()

    @property
    def T(self):
        if self.use_gpu:
            return self._T_d.copy_to_host()
        else:
            return self._T.copy()

    @property
    def W(self):
        if self.use_gpu:
            return self._W_d.copy_to_host()
        else:
            return self._W.copy()

    @property
    def W_total(self):
        if self.use_gpu:
            return self._W_total_d.copy_to_host()
        else:
            return self._W_total.copy()

    @cpu_gpu_dispatcher('_init_sampling_cpu', '_init_sampling_gpu')
    def _init_sampling(self):
        pass

    def _init_sampling_cpu(self):

        # random walk data
        self._N_uv = np.zeros((self._gloader.n_unique_edges,self.n_samples_edge), dtype=np.int32)
        self._V = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length+1), dtype=np.int32)
        self._walks_edges = np.full((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), -10000, dtype=np.int32)
        self._T = np.full((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), -10000, dtype=np.int32)
        self._W = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.float32)
        self._W_total = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.float32)

    def _init_sampling_gpu(self):

        from .utilities_cuda import define_cuda_kernels_sampling
        from ..utilities_cuda import create_xoroshiro128p_states

        kernels = define_cuda_kernels_sampling(self._gloader.n_attr, self._gloader.n_nodes_spatial)

        self.edge_sampler_gpu       : Callable = kernels[0]
        self.spatial_random_walk    : Callable = kernels[1]
        self.attr_sampler_gpu       : Callable = kernels[2]

        self.create_xoroshiro128p_states: Callable = create_xoroshiro128p_states

        # random walk data
        self._N_uv_d = self.cuda.device_array((self._gloader.n_unique_edges,self.n_samples_edge), dtype=np.int32)
        self._V_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length+1), dtype=np.int32)
        self._walks_edges_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.int32)
        self._T_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.int32)
        self._W_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.float32)
        self._W_total_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.float32)

    @cpu_gpu_dispatcher('_randomwalk_cpu', '_randomwalk_gpu')
    def execute_randomwalk(self):
        pass

    def _randomwalk_cpu(self):
        """
        Perform `n` random walks of length `length` starting from root nodes `nodes`.

        Parameters
        ----------
        testing : Optional[bool], optional
            whether to return intermediate values for testing and debugging, by default False
        """
        rng, rng_np = self.random_state

        edge_sampler_cpu(
            adj3 = self._gloader._adj3,
            samples = self._N_uv,
            rs = rng_np
        )

        spatial_random_walk(
            adj = self._gloader._adj_dense,
            adj_i = self._gloader._adj_dense_index,
            V = self._V,
            W_total = self._W_total,
            walks_edges = self._walks_edges,
            rs = rng,
        )

        attr_sampler_cpu(
            walks_edges = self._walks_edges,
            adj3 = self._gloader._adj3,
            T = self._T,
            W = self._W,
            rs = rng_np
        )

        return None

    def _randomwalk_gpu(self):

        #################################
        # perform random walks
        cumsum_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self.n_walks,self._gloader.n_nodes_spatial,),dtype=np.float32)
        weights_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self.n_walks,self._gloader.n_nodes_spatial,),dtype=np.float32)

        # Define the thread and block dimensions
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self.n_samples_edge), TPB = self.TPB_2D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.edge_sampler_gpu[blockspergrid,threadsperblock,self.stream](
            self._gloader._adj3_d,
            self._N_uv_d,
            cuda_random_state,
        )

        # Define the thread and block dimensions
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self.n_walks), TPB = self.TPB_2D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.spatial_random_walk[blockspergrid,threadsperblock,self.stream](
            self._gloader._adj_dense_d,
            self._gloader._adj_dense_index_d,
            self._V_d,
            self._W_total_d,
            self._walks_edges_d,
            cuda_random_state,
            cumsum_d,
            weights_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self.n_walks, self.walk_length), TPB = self.TPB_3D)
        cuda_random_state = self.create_xoroshiro128p_states(nstates,self.seed)

        self.attr_sampler_gpu[blockspergrid,threadsperblock,self.stream](
            self._walks_edges_d,
            self._gloader._adj_d,
            self._gloader._adj3_d,
            self._T_d,
            self._W_d,
            cuda_random_state
        )

        self.stream.synchronize()

    def save(self, directory=None, **kwargs):
        data_to_save = {
            'N_uv': self.N_uv,
            'V': self.V,
            'walks_edges': self.walks_edges,
            'T': self.T,
            'W': self.W,
        }

        for name, data in data_to_save.items():
            with open(f'{directory}/{name}.npy', 'wb') as f:
                np.save(f, data)

    def load(self, directory=None, **kwargs):
        data_to_load = {
            'N_uv': {"name" : '_N_uv', "dtype" : np.ndarray},
            'V': {"name" : '_V', "dtype" : np.ndarray},
            'walks_edges': {"name" : '_walks_edges', "dtype" : np.ndarray},
            'T': {"name" : '_T', "dtype" : np.ndarray},
            'W': {"name" : '_W', "dtype" : np.ndarray},
        }

        for name, data_dict in data_to_load.items():
            with open(f'{directory}/{name}.npy', 'rb') as f:
                data = np.load(f)
                dtype = data_dict["dtype"]
                data = dtype_load_func.get(dtype, load_np_array)(data)
                setattr(self, data_dict["name"], data)

        if self.use_gpu:
            self._N_uv_d = self.cuda.to_device(self._N_uv)
            self._V_d = self.cuda.to_device(self._V)
            self._walks_edges_d = self.cuda.to_device(self._walks_edges)
            self._T_d = self.cuda.to_device(self._T)
            self._W_d = self.cuda.to_device(self._W)

    def __del__(self):
        super().__del__()

        if self.use_gpu and self._init_gpu:
            del self._N_uv_d
            del self._V_d
            del self._walks_edges_d
            del self._T_d
            del self._W_d


    def __getstate__(self):
        state = self.__dict__.copy()

        if self.use_gpu:
            state['_N_uv'] = self.N_uv
            state['_V'] = self.V
            state['_walks_edges'] = self.walks_edges
            state['_T'] = self.T
            state['_W'] = self.W
            state['_W_total'] = self.W_total

            del state['cuda']
            del state['stream']

            safe_delete(state, '_N_uv_d')
            safe_delete(state, '_V_d')
            safe_delete(state, '_walks_edges_d')
            safe_delete(state, '_T_d')
            safe_delete(state, '_W_d')
            safe_delete(state, '_W_total_d')

            del state['edge_sampler_gpu']
            del state['spatial_random_walk']
            del state['attr_sampler_gpu']
            del state['create_xoroshiro128p_states']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.use_gpu:
            self._post_init_gpu(
                threads_per_block_1D=self.TPB_1D,
                threads_per_block_2D=self.TPB_2D,
                threads_per_block_3D=self.TPB_3D,
            )
            self._N_uv_d = self.cuda.to_device(self._N_uv)
            self._V_d = self.cuda.to_device(self._V)
            self._walks_edges_d = self.cuda.to_device(self._walks_edges)
            self._T_d = self.cuda.to_device(self._T)
            self._W_d = self.cuda.to_device(self._W)
            self._W_total_d = self.cuda.to_device(self._W_total)

        self.random_state = random_state(self.seed)


class BaseEvaluator(ABC):

    def __init__(
            self,
            gloader: GraphLoader,
            use_gpu: Optional[bool] = False,
            n_communities: Optional[int] = None,
            threads_per_block_1D: int = 484,
            threads_per_block_2D: Union[int,Tuple[int,int]] = 22,
            threads_per_block_3D: Union[int,Tuple[int,int,int]] = 8,
            stream: Optional[Any] = None
        ):

        self._gloader = gloader


        if n_communities is None:
            self._n_communities = self._gloader.n_attr
        else:
            self._n_communities = n_communities

        self._hhi = defaultdict(float)
        self._log_mpc = {}
        self._mpc = defaultdict(list)
        self._w_k = np.empty(0)
        self._w_ik = np.empty((self._gloader.n_nodes_spatial,0))
        self._w_i = np.empty(self._gloader.n_nodes_spatial)
        self._p_ik = np.empty((self._gloader.n_nodes_spatial,0))
        self._edges_communities = np.empty((self._gloader.n_unique_edges,0))
        self._distinct_communities = np.empty(0)
        self._communities_dict = {}

        self.use_gpu = use_gpu
        self._init_gpu = False

        self._post_init(
            threads_per_block_1D=threads_per_block_1D,
            threads_per_block_2D=threads_per_block_2D,
            threads_per_block_3D=threads_per_block_3D,
            stream=stream
        )

        self.membership = np.empty(self._gloader.n_attr)

    @cpu_gpu_dispatcher('_post_init_cpu', '_post_init_gpu')
    def _post_init(self, *args, **kwargs):
        pass

    @abstractmethod
    def _post_init_cpu(self, *args, **kwargs):
        # Extract thread parameters from kwargs
        self.TPB_1D = kwargs.get('threads_per_block_1D', 484)
        self.TPB_2D = kwargs.get('threads_per_block_2D', 22)
        self.TPB_3D = kwargs.get('threads_per_block_3D', 8)
        
        self._hhi_edges = np.zeros(self._gloader.n_unique_edges,dtype=np.float32)
        self._mpc_attrs = np.zeros(self._gloader.n_attr,dtype=np.float32)
        self._log_mpc_attrs = np.zeros(self._gloader.n_attr,dtype=np.float32)

        self._edges_communities = np.zeros((self._gloader.n_unique_edges,self._n_communities),dtype=np.float32)
        self._w_it = np.zeros((self._gloader.n_nodes_spatial,self._gloader.n_attr),dtype=np.float32)
        self._p_it = np.zeros((self._gloader.n_nodes_spatial,self._gloader.n_attr),dtype=np.float32)
        self._w_i = np.zeros(self._gloader.n_nodes_spatial,dtype=np.float32)
        self._w_ik = np.zeros((self._gloader.n_nodes_spatial,self._n_communities),dtype=np.float32)
        self._p_ik = np.zeros((self._gloader.n_nodes_spatial,self._n_communities),dtype=np.float32)
        self._w_k = np.zeros(self._n_communities)

        pass

    def _post_init_gpu(
            self,
            threads_per_block_1D: int,
            threads_per_block_2D: Union[int,Tuple[int,int]],
            threads_per_block_3D: Union[int,Tuple[int,int,int]],
            stream: Optional[Any] = None
        ):
        assert self.use_gpu, "error: use_gpu flag set to False"
        from .utilities_cuda import cuda
        self.cuda = cuda

        self.TPB_1D = threads_per_block_1D
        self.TPB_2D = threads_per_block_2D
        self.TPB_3D = threads_per_block_3D
        self._init_gpu = True

        if stream is None:
            self.stream = self.cuda.stream()
        else:
            self.stream = stream

        self._w_i_d = self.cuda.device_array((self._gloader.n_nodes_spatial),dtype=np.int32, stream=self.stream)
        self._w_it_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self._gloader.n_attr),dtype=np.float32, stream=self.stream)
        self._p_it_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self._gloader.n_attr),dtype=np.float32, stream=self.stream)

        self._edges_communities_d = self.cuda.device_array((self._gloader.n_unique_edges,self._n_communities), dtype=np.float32, stream=self.stream)
        self._hhi_edges_d = self.cuda.device_array((self._gloader.n_unique_edges), dtype=np.float32, stream=self.stream)

        self._w_ik_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self._n_communities), dtype=np.float32, stream=self.stream)
        self._p_ik_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self._n_communities), dtype=np.float32, stream=self.stream)
        self._w_k_d = self.cuda.device_array((self._n_communities), dtype=np.float32, stream=self.stream)
        self._mpc_attrs_d = self.cuda.device_array((self._gloader.n_attr), dtype=np.float32, stream=self.stream)

        from .utilities_cuda import define_cuda_kernels_sampling_utils

        kernels = define_cuda_kernels_sampling_utils()

        self.init_w_k_gpu       : Callable = kernels[0]
        self.calc_w_k_edges_gpu : Callable = kernels[1]
        self.calc_w_k_nodes_gpu : Callable = kernels[2]
        self.calc_w_k_log_gpu   : Callable = kernels[3]

    @property
    def membership(self)->np.ndarray:
        if self.use_gpu:
            return self._membership_d.copy_to_host()
        else:
            return self._membership.copy()

    @membership.setter
    def membership(self,value:np.ndarray):

        assert value.shape == (self._gloader.n_attr,), f"error: membership array shape {value.shape} does not match number of attributes {self._gloader.n_attr}"

        self._w_ik = np.zeros((self._gloader.n_nodes_spatial,self._n_communities),dtype=np.float32)
        self._w_k = np.zeros(self._n_communities)
        self._edges_communities = np.zeros((self._gloader.n_unique_edges,self._n_communities),dtype=np.float32)
        self._hhi_edges = np.zeros(self._gloader.n_unique_edges,dtype=np.float32)
        self._mpc_attrs = np.zeros(self._gloader.n_attr,dtype=np.float32)
        self._log_mpc_attrs = np.zeros(self._gloader.n_attr,dtype=np.float32)

        self._membership = value.copy()
        if self.use_gpu:
            self._membership_d = self.cuda.to_device(self._membership, stream=self.stream)

            self.stream.synchronize()

    @property
    def communities(self)->np.ndarray:
        return np.array(self._distinct_communities)

    @property
    def communities_dict(self)->Dict[str,int]:
        return {attr: self._communities_dict[attr] for attr in self._communities_dict.keys()}

    @property
    def n_communities(self):
        return self._n_communities

    @n_communities.setter
    def n_communities(self,value:int):
        self._n_communities = value

    @communities_dict.setter
    def communities_dict(self,value:Dict[str,int]):

        for i,attr in enumerate(self._gloader.attr_list):
            attr_value = value.get(attr,None)
            assert attr_value is not None, f"error: community not defined for {attr}"

        self._distinct_communities, self.n_distinct_communities = distinct_communities(value)

        assert self.n_distinct_communities <= self._n_communities, f"error: number of communities {self.n_distinct_communities} exceeds the number of attributes {self._n_communities}"

        self._communities_dict = value.copy()

        self.membership = np.array(list(self._communities_dict.values()))

    @property
    def hhi(self) -> Dict[str,float]:
        return {edge: self._hhi[edge] for edge in self._hhi.keys()}

    @property
    def mpc(self) -> Dict[str,float]:
        return {attr: self._mpc[attr] for attr in self._mpc.keys()}

    @property
    def log_mpc(self) -> Dict[str,float]:
        return {attr: self._log_mpc[attr] for attr in self._log_mpc.keys()}

    @log_mpc.setter
    def log_mpc(self, value: Dict[str,float]):
        self._log_mpc = value

    @property
    def mpc_attrs(self) -> np.ndarray:
        return self._mpc_attrs.copy()

    @property
    def log_mpc_attrs(self) -> np.ndarray:
        return self._log_mpc_attrs.copy()

    @property
    def hhi_edges(self):
        if self.use_gpu:
            return self._hhi_edges_d.copy_to_host()
        else:
            return self._hhi_edges.copy()

    @property
    def edges_communities(self):
        if self.use_gpu:
            return self._edges_communities_d.copy_to_host()
        else:
            return self._edges_communities.copy()

    @property
    def w_it(self):
        if self.use_gpu:
            return self._w_it_d.copy_to_host()
        else:
            return self._w_it.copy()

    @property
    def w_ik(self):
        if self.use_gpu:
            return self._w_ik_d.copy_to_host()
        else:
            return self._w_ik.copy()

    @property
    def w_i(self):
        if self.use_gpu:
            return self._w_i_d.copy_to_host()
        else:
            return self._w_i.copy()

    @property
    def w_k(self):
        if self.use_gpu:
            return self._w_k_d.copy_to_host()
        else:
            return self._w_k.copy()

    @property
    def p_it(self):
        if self.use_gpu:
            return self._p_it_d.copy_to_host()
        else:
            return self._p_it.copy()

    @property
    def p_ik(self):
        if self.use_gpu:
            return self._p_ik_d.copy_to_host()
        else:
            return self._p_ik.copy()

    @abstractmethod
    def update_scores(self, communities_dict: Dict[str,str]):
        self.communities_dict = communities_dict

    @abstractmethod
    def update_scores_membership(self, membership: np.ndarray):
        communities_dict = {attr: membership[i] for i,attr in enumerate(self._gloader.attr_list)}
        self.communities_dict = communities_dict

    @cpu_gpu_dispatcher('_update_community_weights_cpu', '_update_community_weights_gpu')
    def _update_community_weights(self):
        pass

    def _update_community_weights_cpu(self):
        self._w_k = safe_log(np.sum(self._w_ik,axis=0) / self._gloader.n_nodes_spatial) \
            + np.sum(self._edges_communities,axis=0) / self._gloader.n_unique_edges

    def _update_community_weights_gpu(self):
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._n_communities,), TPB = self.TPB_1D)

        self.init_w_k_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_k_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB = self.TPB_2D)
        self.calc_w_k_edges_gpu[blockspergrid,threadsperblock,self.stream](
            self._edges_communities_d,
            self._w_k_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB = self.TPB_2D)
        self.calc_w_k_nodes_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_ik_d,
            self._w_k_d
        )

        self.stream.synchronize()

        n_states, threadsperblock, blockspergrid = cuda_blocks(dims=(self._n_communities,), TPB = self.TPB_1D)
        self.calc_w_k_log_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_k_d
        )

        self.stream.synchronize()

    @_jsonmode
    def save(self, directory=None, **kwargs):
        data_to_save = {
            'communities_dict': self._communities_dict,
            'log_mpc': self._log_mpc,
            'mpc': self._mpc,
            'hhi': {str(k): v for k, v in self._hhi.items()},
            '_n_communities': self._n_communities,
        }

        for name, data in data_to_save.items():
            with open(f'{directory}/{name}.json', 'w') as f:
                json.dump(serialize(data), f, indent=4)

    @_jsonmode
    def load(self, directory=None, **kwargs):
        data_to_load = {
            'communities_dict':     {"name" : '_communities_dict', "dtype" : dict},
            'log_mpc':              {"name" : '_log_mpc', "dtype" : dict},
            'mpc':                  {"name" : '_mpc', "dtype" : dict},
            'hhi':                  {"name" : '_hhi', "dtype" : Dict[Tuple[str,str],int]},
            '_n_communities':        {"name" : '_n_communities', "dtype" : int},
        }

        for name, data_dict in data_to_load.items():
            with open(f'{directory}/{name}.json', 'r') as f:
                data = deserialize(json.load(f))
                dtype = data_dict["dtype"]
                data = dtype_load_func.get(dtype, load_dict)(data)
                setattr(self, data_dict["name"], data)

    def copy(self, memo: Optional[Dict[int, Any]] = None, stream: Optional[Any] = None) -> 'BaseEvaluator':
        if memo is None:
            memo = {}

        if self.use_gpu and stream is None:
            stream = self.cuda.stream()

        new_instance = self._create_copy_instance(stream)

        memo[id(self)] = new_instance

        # Copy shared attributes
        self.copy_attributes(new_instance)

        if self.use_gpu:
            self.copy_gpu_attributes(new_instance, stream)

        return new_instance

    @abstractmethod
    def _create_copy_instance(self, stream: Optional[Any] = None) -> 'BaseEvaluator':
        pass

    def copy_attributes(self, target: 'BaseEvaluator'):
        # Copy common attributes
        target._edges_communities = self.edges_communities
        target._hhi = self.hhi
        target._log_mpc = self.log_mpc
        target._mpc = self.mpc
        target.membership = self.membership
        target._communities_dict = self.communities_dict

        target._mpc_attrs = self.mpc_attrs
        target._log_mpc_attrs = self.log_mpc_attrs
        target._hhi_edges = self.hhi_edges
        target._w_it = self.w_it
        target._w_i = self.w_i
        target._p_it = self.p_it
        target._w_ik = self.w_ik
        target._w_k = self.w_k
        target._p_ik = self.p_ik

    def copy_gpu_attributes(self, target: 'BaseEvaluator', stream: Any):
        # Copy GPU-specific attributes
        target._w_it_d = self.cuda.to_device(self.w_it, stream=stream)
        target._w_i_d = self.cuda.to_device(self.w_i, stream=stream)
        target._p_it_d = self.cuda.to_device(self.p_it, stream=stream)
        target._w_ik_d = self.cuda.to_device(self.w_ik, stream=stream)
        target._w_k_d = self.cuda.to_device(self.w_k, stream=stream)
        target._p_ik_d = self.cuda.to_device(self.p_ik, stream=stream)

        target._mpc_attrs_d = self.cuda.to_device(self.mpc_attrs, stream=stream)
        target._edges_communities_d = self.cuda.to_device(self.edges_communities, stream=stream)
        target._hhi_edges_d = self.cuda.to_device(self.hhi_edges, stream=stream)
        target._membership_d = self.cuda.to_device(self.membership, stream=stream)

        stream.synchronize()

    def __getstate__(self):
        state = self.__dict__.copy()

        if self.use_gpu:
            del state['cuda']
            del state['stream']

            safe_delete(state, '_edges_communities_d')
            safe_delete(state, '_hhi_edges_d')
            safe_delete(state, '_w_it_d')
            safe_delete(state, '_w_i_d')
            safe_delete(state, '_p_it_d')
            safe_delete(state, '_w_ik_d')
            safe_delete(state, '_w_k_d')
            safe_delete(state, '_p_ik_d')
            safe_delete(state, '_mpc_attrs_d')
            safe_delete(state, '_membership_d')

            state.pop('init_w_k_gpu', None)
            state.pop('calc_w_k_edges_gpu', None)
            state.pop('calc_w_k_nodes_gpu', None)
            state.pop('calc_w_k_log_gpu', None)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.use_gpu:
            self._post_init_gpu(
                threads_per_block_1D=self.TPB_1D,
                threads_per_block_2D=self.TPB_2D,
                threads_per_block_3D=self.TPB_3D,
            )

    def __del__(self):
        if self.use_gpu and self._init_gpu:

            if hasattr(self, '_edges_communities_d'):
                del self._edges_communities_d
            if hasattr(self, '_hhi_edges_d'):
                del self._hhi_edges_d
            if hasattr(self, '_w_it_d'):
                del self._w_it_d
            if hasattr(self, '_w_i_d'):
                del self._w_i_d
            if hasattr(self, '_p_it_d'):
                del self._p_it_d
            if hasattr(self, '_w_ik_d'):
                del self._w_ik_d
            if hasattr(self, '_w_k_d'):
                del self._w_k_d
            if hasattr(self, '_p_ik_d'):
                del self._p_ik_d
            if hasattr(self, '_mpc_attrs_d'):
                del self._mpc_attrs_d
            if hasattr(self, '_membership_d'):
                del self._membership_d
                

class ExactEvaluator(BaseEvaluator):

    def __init__(
            self,
            gloader: GraphLoader,
            use_gpu: Optional[bool] = False,
            n_communities: Optional[int] = None,
            walk_length: Optional[int] = 3,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            stream: Optional[Any] = None
        ) -> None:
        self.walk_length = walk_length
        super().__init__(gloader, use_gpu, n_communities, threads_per_block_1D, threads_per_block_2D, threads_per_block_3D, stream)
        
    @property
    def p_et(self):
        if self.use_gpu:
            return self._gloader._cp_d.copy_to_host()
        else:
            return self._gloader._cp.copy()

    def _post_init_cpu(self, *args, **kwargs):
        super()._post_init_cpu(args, kwargs)

        self._p_iv = np.zeros((self._gloader.n_nodes_spatial,self._gloader.n_nodes_spatial),dtype=np.float32)
        self._p_iv_t = np.zeros((self._gloader.n_nodes_spatial,self._gloader.n_nodes_spatial, self._gloader.n_attr),dtype=np.float32)

        calc_w_it_cpu(
            adj=self._gloader._adj_sparse,
            adj3=self._gloader._adj3,
            w_it=self._w_it
        )

        self._w_i = np.sum(self._w_it,axis=1)

        calc_p_iv_cpu(
            adj_sparse = self._gloader._adj_sparse,
            adj_weight = self._gloader._adj,
            w_i = self._w_i,
            p_iv = self._p_iv
        )

        calc_p_iv_t_cpu(
            adj_multi_attr_sparse = self._gloader._adj_multi_attr_sparse,
            adj_multi_attr_type = self._gloader._adj_multi_attr_type,
            adj_multi_attr_weight = self._gloader._adj_multi_attr,
            adj_multi_attr_index = self._gloader._adj_multi_attr_index,
            adj_weight = self._gloader._adj,
            p_iv_t = self._p_iv_t
        )

        calc_p_it_L(
            p_iv = self._p_iv,
            p_iv_t = self._p_iv_t,
            L=self.walk_length,
            p_it=self._p_it
        )

        return None

    def _post_init_gpu(
            self,
            threads_per_block_1D: int,
            threads_per_block_2D: Union[int,Tuple[int,int]],
            threads_per_block_3D: Union[int,Tuple[int,int,int]],
            stream: Optional[Any] = None
        ):
        super()._post_init_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D,stream)

        from .utilities_cuda import define_cuda_kernels_deterministic

        # load kernels
        kernels = define_cuda_kernels_deterministic()
        self.init_edges_communities         : Callable = kernels[0]
        self.init_hhi                       : Callable = kernels[1]
        self.init_w_it                      : Callable = kernels[2]
        self.init_w_ik                      : Callable = kernels[3]
        self.init_w_i                       : Callable = kernels[4]
        self.init_mpc                       : Callable = kernels[5]
        self.edge_postprocess_gpu           : Callable = kernels[6]
        self.calc_hhi_gpu                   : Callable = kernels[7]
        self.calc_w_it_gpu                  : Callable = kernels[8]
        self.calc_w_i_gpu                   : Callable = kernels[9]
        self.calc_w_ik_gpu                  : Callable = kernels[10]
        self.normalize_w_ix_gpu             : Callable = kernels[11]
        self.calc_mpc_gpu                   : Callable = kernels[12]

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB = self.TPB_1D)

        self.init_w_i[blockspergrid,threadsperblock,self.stream](
            self._w_i_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)

        self.init_w_it[blockspergrid,threadsperblock,self.stream](
            self._w_it_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB = self.TPB_2D)

        self.calc_w_it_gpu[blockspergrid,threadsperblock,self.stream](
            self._gloader._adj_sparse_d,
            self._gloader._adj3_d,
            self._w_it_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB = self.TPB_1D)

        self.calc_w_i_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_it_d,
            self._w_i_d
        )

        self.stream.synchronize()

    def update_scores(self, communities_dict: Dict[str,str]):

        super().update_scores(communities_dict)

        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

    def update_scores_membership(self, membership: np.ndarray):
        super().update_scores_membership(membership)

        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

    @cpu_gpu_dispatcher('_update_hhi_scores_cpu', '_update_hhi_scores_gpu')
    def _update_hhi_scores(self, **kwargs):
        pass

    def _update_hhi_scores_cpu(self):

        edge_postprocess_cpu(
            edges_cp=self._gloader._cp,
            membership=self._membership,
            edges_communities=self._edges_communities
        )

        calc_hhi_cpu(
            edges_communities=self._edges_communities,
            hhi_edges=self._hhi_edges,
        )

        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = self._hhi_edges[i]

    def _update_hhi_scores_gpu(self):

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB = self.TPB_2D)

        self.init_edges_communities[blockspergrid,threadsperblock,self.stream](
            self._edges_communities_d,
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB = self.TPB_2D)

        self.edge_postprocess_gpu[blockspergrid,threadsperblock,self.stream](
            self._gloader._cp_d,
            self._membership_d,
            self._edges_communities_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges,), TPB = self.TPB_1D)

        self.init_hhi[blockspergrid,threadsperblock,self.stream](
            self._hhi_edges_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB = self.TPB_2D)

        self.calc_hhi_gpu[blockspergrid,threadsperblock,self.stream](
            self._edges_communities_d,
            self._hhi_edges_d
        )

        self.stream.synchronize()

        hhi = self._hhi_edges_d.copy_to_host()

        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = hhi[i]

    @cpu_gpu_dispatcher('_update_mpc_scores_cpu', '_update_mpc_scores_gpu')
    def _update_mpc_scores(self):
        pass

    def _update_mpc_scores_cpu(self):

        self._p_ik = np.zeros((self._gloader.n_nodes_spatial,self._n_communities),dtype=np.float32)

        calc_p_ik_L(
            p_iv=self._p_iv,
            p_iv_t=self._p_iv_t,
            L=self.walk_length,
            membership=self._membership,
            n_attr=self._gloader.n_attr,
            p_ik=self._p_ik
        )

        calc_mpc_cpu(
            membership=self._membership,
            p_it=self._p_it,
            p_ik=self._p_ik,
            mpc=self._mpc_attrs
        )

        self._mpc_attrs /= self._gloader.n_nodes_spatial
        self._log_mpc_attrs = safe_log(self._mpc_attrs)

        for attr in self._gloader.attr_list:
            self._mpc[attr] = self._mpc_attrs[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = self._log_mpc_attrs[self._gloader.attr_key2i[attr]]

    def _update_mpc_scores_gpu(self):

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB = self.TPB_2D)

        self.init_w_ik[blockspergrid,threadsperblock,self.stream](
            self._w_ik_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)
        self.calc_w_ik_gpu[blockspergrid,threadsperblock,self.stream](
            self._membership_d,
            self._w_it_d,
            self._w_ik_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB = self.TPB_1D)

        self.init_mpc[blockspergrid,threadsperblock,self.stream](
            self._mpc_attrs_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)
        self.calc_mpc_gpu[blockspergrid,threadsperblock,self.stream](
            self._membership_d,
            self._w_it_d,
            self._w_ik_d,
            self._w_i_d,
            self._mpc_attrs_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)
        self.normalize_w_ix_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_it_d,
            self._w_i_d,
            self._p_it_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB = self.TPB_2D)
        self.normalize_w_ix_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_ik_d,
            self._w_i_d,
            self._p_ik_d
        )

        self.stream.synchronize()

        self._mpc_attrs = self._mpc_attrs_d.copy_to_host() / self._gloader.n_nodes_spatial
        self._log_mpc_attrs = safe_log(self._mpc_attrs)

        for attr in self._gloader.attr_list:
            self._mpc[attr] = self._mpc_attrs[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = self._log_mpc_attrs[self._gloader.attr_key2i[attr]]

    def _create_copy_instance(self, stream: Optional[Any] = None) -> 'ExactEvaluator':
        if self.use_gpu:
            return self.__class__(
                gloader=self._gloader,
                use_gpu=self.use_gpu,
                walk_length=self.walk_length,
                n_communities=self._n_communities,
                threads_per_block_1D=self.TPB_1D,
                threads_per_block_2D=self.TPB_2D,
                threads_per_block_3D=self.TPB_3D,
                stream=stream
            )
        else:
            return self.__class__(
                gloader=self._gloader,
                use_gpu=self.use_gpu,
                walk_length=self.walk_length,
                n_communities=self._n_communities
            )

    def __getstate__(self):
        state = super().__getstate__()

        if self.use_gpu:

            state.pop('init_edges_communities', None)
            state.pop('init_hhi', None)
            state.pop('init_w_it', None)
            state.pop('init_w_ik', None)
            state.pop('init_w_i', None)
            state.pop('init_mpc', None)
            state.pop('edge_postprocess_gpu', None)
            state.pop('calc_hhi_gpu', None)
            state.pop('calc_w_it_gpu', None)
            state.pop('calc_w_i_gpu', None)
            state.pop('calc_w_ik_gpu', None)
            state.pop('calc_mpc_gpu', None)

        return state

    def __del__(self):
        super().__del__()


class ExactWeightedEvaluator(BaseEvaluator):
    """
    Exact weighted evaluator that computes p_it and p_ik using weighted recursive formulation.
    
    This evaluator implements the weighted capacity-based approach where:
    - P_i(v) = w_{i,v} / Z_i is the neighbor PMF
    - P_{i,v}(tau) = w_{i,v,tau} / w_{i,v} is the conditional attribute PMF
    - S_tau(i, L) and S_alpha_k(i, L) are computed recursively
    - Normalized MPC scores are computed from the weighted scores
    
    Parameters
    ----------
    gloader : GraphLoader
        Graph loader containing the multi-digraph data
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: False). Currently only CPU is supported.
    n_communities : int, optional
        Number of communities/alliances (default: number of attributes)
    walk_length : int, optional
        Depth L for recursive calculation (default: 3)
    lambda_decay : float, optional
        Decay factor lambda for recursive terms (default: 1.0)
    threads_per_block_1D : int, optional
        GPU threads per block for 1D kernels (default: 484)
    threads_per_block_2D : Union[int, Tuple[int,int]], optional
        GPU threads per block for 2D kernels (default: 22)
    threads_per_block_3D : Union[int, Tuple[int,int,int]], optional
        GPU threads per block for 3D kernels (default: 8)
    stream : optional
        CUDA stream for GPU operations (default: None)
    """
    
    def __init__(
            self,
            gloader: 'GraphLoader',
            use_gpu: Optional[bool] = False,
            n_communities: Optional[int] = None,
            walk_length: Optional[int] = 3,
            lambda_decay: Optional[float] = 1.0,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            stream: Optional[Any] = None
        ) -> None:
        self.walk_length = walk_length
        self.lambda_decay = lambda_decay
        
        super().__init__(gloader, use_gpu, n_communities, threads_per_block_1D, threads_per_block_2D, threads_per_block_3D, stream)

    @property
    def p_et(self):
        """Return edge conditional probabilities."""
        if self.use_gpu:
            return self._gloader._cp_d.copy_to_host()
        else:
            return self._gloader._cp.copy()

    def _post_init_cpu(self, *args, **kwargs):
        """Initialize CPU-specific arrays and compute base probabilities."""
        super()._post_init_cpu(args, kwargs)

        # Initialize probability arrays
        self._p_iv = np.zeros((self._gloader.n_nodes_spatial, self._gloader.n_nodes_spatial), dtype=np.float32)
        self._p_iv_t = np.zeros((self._gloader.n_nodes_spatial, self._gloader.n_nodes_spatial, self._gloader.n_attr), dtype=np.float32)
        
        # Weighted score arrays
        self._s_t = np.zeros((self._gloader.n_nodes_spatial, self._gloader.n_attr), dtype=np.float32)
        self._s_k = np.zeros((self._gloader.n_nodes_spatial, self._n_communities), dtype=np.float32)
        self._s_total = np.zeros(self._gloader.n_nodes_spatial, dtype=np.float32)
        
        # Compute w_it (attribute weights per node) - reuse existing method
        calc_w_it_cpu(
            adj=self._gloader._adj_sparse,
            adj3=self._gloader._adj3,
            w_it=self._w_it
        )

        # Compute w_i (total outgoing weights) - same as Z_i
        self._w_i = np.sum(self._w_it, axis=1)

        # Compute P_i(v) = w_{i,v} / w_i - reuse existing method
        calc_p_iv_cpu(
            adj_sparse=self._gloader._adj_sparse,
            adj_weight=self._gloader._adj,
            w_i=self._w_i,
            p_iv=self._p_iv
        )

        # Compute P_{i,v}(t) - conditional attribute PMF
        calc_p_iv_t_cpu(
            adj_multi_attr_sparse=self._gloader._adj_multi_attr_sparse,
            adj_multi_attr_type=self._gloader._adj_multi_attr_type,
            adj_multi_attr_weight=self._gloader._adj_multi_attr,
            adj_multi_attr_index=self._gloader._adj_multi_attr_index,
            adj_weight=self._gloader._adj,
            p_iv_t=self._p_iv_t
        )

        # Compute total capacity scores (attribute-agnostic)
        calc_s_total_weighted_cpu(
            p_iv=self._p_iv,
            w_i=self._w_i,
            L=self.walk_length,
            lambda_decay=self.lambda_decay,
            s_total=self._s_total
        )

        # Compute attribute scores
        calc_s_t_weighted_cpu(
            p_iv=self._p_iv,
            w_i=self._w_i,
            p_iv_t=self._p_iv_t,
            L=self.walk_length,
            lambda_decay=self.lambda_decay,
            s_t=self._s_t
        )

        # Normalize to get probabilities
        calc_normalized_p_t_weighted_cpu(
            s_t=self._s_t,
            s_total=self._s_total,
            p_t_normalized=self._p_it
        )

        return None

    def _post_init_gpu(
            self,
            threads_per_block_1D: int,
            threads_per_block_2D: Union[int,Tuple[int,int]],
            threads_per_block_3D: Union[int,Tuple[int,int,int]],
            stream: Optional[Any] = None
        ):
        """GPU initialization for ExactWeightedEvaluator."""
        super()._post_init_gpu(threads_per_block_1D, threads_per_block_2D, threads_per_block_3D, stream)

        from .utilities_cuda import cuda, define_cuda_kernels_deterministic
        from .utilities_weighted_cuda import define_cuda_kernels_weighted_exact

        # Load base deterministic kernels (needed for w_it computation)
        base_kernels = define_cuda_kernels_deterministic()
        self.init_edges_communities         : Callable = base_kernels[0]
        self.init_hhi                       : Callable = base_kernels[1]
        self.init_w_it                      : Callable = base_kernels[2]
        self.init_w_ik                      : Callable = base_kernels[3]
        self.init_w_i                       : Callable = base_kernels[4]
        self.init_mpc                       : Callable = base_kernels[5]
        self.edge_postprocess_gpu           : Callable = base_kernels[6]
        self.calc_hhi_gpu                   : Callable = base_kernels[7]
        self.calc_w_it_gpu                  : Callable = base_kernels[8]
        self.calc_w_i_gpu                   : Callable = base_kernels[9]
        self.calc_w_ik_gpu                  : Callable = base_kernels[10]
        self.normalize_w_ix_gpu             : Callable = base_kernels[11]
        self.calc_mpc_gpu                   : Callable = base_kernels[12]

        # Load weighted exact kernels
        weighted_kernels = define_cuda_kernels_weighted_exact(
            n_attr=self._gloader.n_attr,
            n_nodes=self._gloader.n_nodes_spatial,
            n_communities=self._n_communities
        )
        
        self.calc_s_t_weighted_gpu              : Callable = weighted_kernels[0]
        self.calc_s_k_weighted_gpu              : Callable = weighted_kernels[1]
        self.calc_s_total_weighted_gpu          : Callable = weighted_kernels[2]
        self.calc_normalized_p_t_weighted_gpu   : Callable = weighted_kernels[3]
        self.calc_normalized_p_k_weighted_gpu   : Callable = weighted_kernels[4]
        self.calc_mpc_weighted_gpu              : Callable = weighted_kernels[5]

        # Allocate GPU arrays for probability calculations
        self._p_iv_d = self.cuda.device_array((self._gloader.n_nodes_spatial, self._gloader.n_nodes_spatial), dtype=np.float32, stream=self.stream)
        self._p_iv_t_d = self.cuda.device_array((self._gloader.n_nodes_spatial, self._gloader.n_nodes_spatial, self._gloader.n_attr), dtype=np.float32, stream=self.stream)
        
        # Allocate weighted score arrays
        self._s_t_d = self.cuda.device_array((self._gloader.n_nodes_spatial, self._gloader.n_attr), dtype=np.float32, stream=self.stream)
        self._s_k_d = self.cuda.device_array((self._gloader.n_nodes_spatial, self._n_communities), dtype=np.float32, stream=self.stream)
        self._s_total_d = self.cuda.device_array(self._gloader.n_nodes_spatial, dtype=np.float32, stream=self.stream)
        
        # Compute w_it (attribute weights per node) using existing kernels
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB=self.TPB_2D)
        self.calc_w_it_gpu[blockspergrid, threadsperblock, self.stream](
            self._gloader._adj_sparse_d,
            self._gloader._adj3_d,
            self._w_it_d
        )

        # Compute w_i (total outgoing weights)
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB=self.TPB_1D)
        self.calc_w_i_gpu[blockspergrid, threadsperblock, self.stream](
            self._w_it_d,
            self._w_i_d
        )

        self.stream.synchronize()

        # For now, copy w_i to host to compute p_iv and p_iv_t on CPU, then copy back
        # TODO: Implement GPU versions of calc_p_iv_gpu and calc_p_iv_t_gpu
        w_it_host = self._w_it_d.copy_to_host()
        w_i_host = self._w_i_d.copy_to_host()
        
        p_iv_host = np.zeros((self._gloader.n_nodes_spatial, self._gloader.n_nodes_spatial), dtype=np.float32)
        p_iv_t_host = np.zeros((self._gloader.n_nodes_spatial, self._gloader.n_nodes_spatial, self._gloader.n_attr), dtype=np.float32)
        
        # Compute on CPU
        calc_p_iv_cpu(
            adj_sparse=self._gloader._adj_sparse,
            adj_weight=self._gloader._adj,
            w_i=w_i_host,
            p_iv=p_iv_host
        )

        calc_p_iv_t_cpu(
            adj_multi_attr_sparse=self._gloader._adj_multi_attr_sparse,
            adj_multi_attr_type=self._gloader._adj_multi_attr_type,
            adj_multi_attr_weight=self._gloader._adj_multi_attr,
            adj_multi_attr_index=self._gloader._adj_multi_attr_index,
            adj_weight=self._gloader._adj,
            p_iv_t=p_iv_t_host
        )
        
        # Copy to GPU
        self._p_iv_d.copy_to_device(p_iv_host)
        self._p_iv_t_d.copy_to_device(p_iv_t_host)

        # Compute total capacity scores (L=1 only)
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB=self.TPB_1D)
        self.calc_s_total_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._p_iv_d,
            self._w_i_d,
            self._s_total_d,
            1,  # L=1 always
            self.lambda_decay
        )

        # Compute attribute scores (L=1 only)
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB=self.TPB_2D)
        self.calc_s_t_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._p_iv_d,
            self._w_i_d,
            self._p_iv_t_d,
            self._s_t_d,
            1,  # L=1 always
            self.lambda_decay
        )

        # Normalize to get probabilities
        self.calc_normalized_p_t_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._s_t_d,
            self._s_total_d,
            self._p_it_d
        )

        self.stream.synchronize()

    def update_scores(self, communities_dict: Dict[str,str]):
        """Update HHI and MPC scores for a given community assignment."""
        super().update_scores(communities_dict)

        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

    def update_scores_membership(self, membership: np.ndarray):
        """Update HHI and MPC scores for a given membership array."""
        super().update_scores_membership(membership)

        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

    @cpu_gpu_dispatcher('_update_hhi_scores_cpu', '_update_hhi_scores_gpu')
    def _update_hhi_scores(self, **kwargs):
        pass

    def _update_hhi_scores_cpu(self):
        """Compute HHI scores based on edge community probabilities."""
        edge_postprocess_cpu(
            edges_cp=self._gloader._cp,
            membership=self._membership,
            edges_communities=self._edges_communities
        )

        calc_hhi_cpu(
            edges_communities=self._edges_communities,
            hhi_edges=self._hhi_edges,
        )

        # Update the hhi dictionary
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = self._hhi_edges[i]

    def _update_hhi_scores_gpu(self):
        """GPU version of HHI score computation."""
        # Use existing GPU HHI computation from base class
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB=self.TPB_2D)

        self.init_edges_communities[blockspergrid, threadsperblock, self.stream](
            self._edges_communities_d,
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB=self.TPB_2D)

        self.edge_postprocess_gpu[blockspergrid, threadsperblock, self.stream](
            self._gloader._cp_d,
            self._membership_d,
            self._edges_communities_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges,), TPB=self.TPB_1D)

        self.init_hhi[blockspergrid, threadsperblock, self.stream](
            self._hhi_edges_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB=self.TPB_2D)

        self.calc_hhi_gpu[blockspergrid, threadsperblock, self.stream](
            self._edges_communities_d,
            self._hhi_edges_d
        )

        self.stream.synchronize()

        hhi = self._hhi_edges_d.copy_to_host()

        # Update the hhi dictionary
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = hhi[i]

    @cpu_gpu_dispatcher('_update_mpc_scores_cpu', '_update_mpc_scores_gpu')
    def _update_mpc_scores(self):
        pass

    def _update_mpc_scores_cpu(self):
        """Compute MPC scores using weighted recursive formulation."""
        # Reset alliance scores
        self._s_k = np.zeros((self._gloader.n_nodes_spatial, self._n_communities), dtype=np.float32)
        self._p_ik = np.zeros((self._gloader.n_nodes_spatial, self._n_communities), dtype=np.float32)

        # Compute alliance scores S_k(i, L)
        calc_s_k_weighted_cpu(
            p_iv=self._p_iv,
            w_i=self._w_i,
            p_iv_t=self._p_iv_t,
            L=self.walk_length,
            lambda_decay=self.lambda_decay,
            membership=self._membership,
            s_k=self._s_k
        )

        # Normalize alliance scores
        calc_normalized_p_k_weighted_cpu(
            s_k=self._s_k,
            s_total=self._s_total,
            p_k_normalized=self._p_ik
        )

        # Compute MPC as product of attribute and alliance probabilities
        calc_mpc_weighted_cpu(
            membership=self._membership,
            p_t_normalized=self._p_it,
            p_k_normalized=self._p_ik,
            mpc=self._mpc_attrs
        )

        self._log_mpc_attrs = safe_log(self._mpc_attrs)

        # Update the mpc dictionaries
        for attr in self._gloader.attr_list:
            self._mpc[attr] = self._mpc_attrs[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = self._log_mpc_attrs[self._gloader.attr_key2i[attr]]

    def _update_mpc_scores_gpu(self):
        """GPU version of MPC score computation using weighted formulation."""
        # Compute alliance scores S_k(i, L) (L=1 only)
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB=self.TPB_2D)
        self.calc_s_k_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._p_iv_d,
            self._w_i_d,
            self._p_iv_t_d,
            self._membership_d,
            self._s_k_d,
            1,  # L=1 always
            self.lambda_decay
        )

        # Normalize alliance scores
        self.calc_normalized_p_k_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._s_k_d,
            self._s_total_d,
            self._p_ik_d
        )

        self.stream.synchronize()
        
        # Initialize MPC array
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_attr,), TPB=self.TPB_1D)
        self.init_mpc[blockspergrid, threadsperblock, self.stream](
            self._mpc_attrs_d
        )

        # Compute MPC as product of attribute and alliance probabilities
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_attr, self._gloader.n_nodes_spatial), TPB=self.TPB_2D)
        self.calc_mpc_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._membership_d,
            self._p_it_d,
            self._p_ik_d,
            self._mpc_attrs_d
        )

        self.stream.synchronize()

        self._mpc_attrs = self._mpc_attrs_d.copy_to_host()
        self._log_mpc_attrs = safe_log(self._mpc_attrs)

        # Update the mpc dictionaries
        for attr in self._gloader.attr_list:
            self._mpc[attr] = self._mpc_attrs[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = self._log_mpc_attrs[self._gloader.attr_key2i[attr]]

    def _create_copy_instance(self, stream: Optional[Any] = None) -> 'ExactWeightedEvaluator':
        """Create a copy of this evaluator instance."""
        return self.__class__(
            gloader=self._gloader,
            use_gpu=self.use_gpu,
            walk_length=self.walk_length,
            lambda_decay=self.lambda_decay,
            n_communities=self._n_communities
        )

    def __getstate__(self):
        """Prepare instance for pickling."""
        state = super().__getstate__()

        if self.use_gpu:
            safe_delete(state, '_p_iv_d')
            safe_delete(state, '_p_iv_t_d')
            safe_delete(state, '_s_t_d')
            safe_delete(state, '_s_k_d')
            safe_delete(state, '_s_total_d')

            state.pop('calc_s_t_weighted_gpu', None)
            state.pop('calc_s_k_weighted_gpu', None)
            state.pop('calc_s_total_weighted_gpu', None)
            state.pop('calc_normalized_mpc_weighted_gpu', None)
            state.pop('calc_normalized_mpc_k_weighted_gpu', None)
            state.pop('calc_mpc_weighted_gpu', None)

        return state

    def __del__(self):
        """Destructor."""
        if self.use_gpu and self._init_gpu:
            # Clean up additional GPU arrays allocated by ExactWeightedEvaluator
            if hasattr(self, '_p_iv_d'):
                del self._p_iv_d
            if hasattr(self, '_p_iv_t_d'):
                del self._p_iv_t_d
            if hasattr(self, '_s_t_d'):
                del self._s_t_d
            if hasattr(self, '_s_k_d'):
                del self._s_k_d
            if hasattr(self, '_s_total_d'):
                del self._s_total_d
        super().__del__()


class SampleEvaluator(BaseEvaluator):

    def __init__(
            self,
            gloader: GraphLoader,
            sampler: NeighbourSampler,
            use_gpu: bool | None = False,
            n_communities: Optional[int] = None,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            stream: Optional[Any] = None
        ) -> None:

        self._sampler = sampler
        super().__init__(
            gloader = gloader,
            use_gpu = use_gpu,
            n_communities = n_communities,
            threads_per_block_1D = threads_per_block_1D,
            threads_per_block_2D = threads_per_block_2D,
            threads_per_block_3D = threads_per_block_3D,
            stream = stream
        )

    @property
    def edges_attrs(self):
        if self.use_gpu:
            return self._edges_attrs_d.copy_to_host()
        else:
            return self._edges_attrs.copy()

    @property
    def p_it(self):
        if self.use_gpu:
            return self._w_it_d.copy_to_host() / (self._sampler.n_walks * self._sampler.walk_length)
        else:
            return self._w_it.copy() / (self._sampler.n_walks * self._sampler.walk_length)

    @property
    def p_et(self):
        if self.use_gpu:
            return self._edges_attrs_d.copy_to_host() / self._sampler.n_samples_edge
        else:
            return self._edges_attrs.copy() / self._sampler.n_samples_edge

    def _post_init_cpu(self, **kwargs):
        super()._post_init_cpu(**kwargs)

        self._edges_attrs = np.zeros((self._gloader.n_unique_edges,self._gloader.n_attr))

        edge_sample_p_et_cpu(
            samples = self._sampler.N_uv,
            counts = self._edges_attrs
        )

        calc_sample_w_i_cpu(
            T_ijk = self._sampler._T,
            w_i = self._w_i
        )

        calc_sample_w_it_cpu(
            T_ijk = self._sampler._T,
            w_it = self._w_it
        )

        # estimate p_it
        self._p_it = self._w_it / (self._sampler.n_walks * self._sampler.walk_length)

    def _post_init_gpu(
            self,
            threads_per_block_1D: int,
            threads_per_block_2D: Union[int,Tuple[int,int]],
            threads_per_block_3D: Union[int,Tuple[int,int,int]],
            stream: Optional[Any] = None
        ):
        super()._post_init_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D,stream)

        from .utilities_cuda import define_cuda_kernels_stochastic

        kernels = define_cuda_kernels_stochastic(n_walks=self._sampler.n_walks, walk_length=self._sampler.walk_length)


        self.init_sample_p_et_gpu           : Callable = kernels[0]
        self.init_sample_p_ek_gpu           : Callable = kernels[1]
        self.init_sample_hhi_gpu            : Callable = kernels[2]
        self.init_sample_count_i_gpu        : Callable = kernels[3]
        self.init_sample_count_it_gpu       : Callable = kernels[4]
        self.init_sample_count_ik_gpu       : Callable = kernels[5]
        self.init_sample_mpc_gpu            : Callable = kernels[6]
        self.edge_sample_p_et_gpu           : Callable = kernels[7]
        self.edge_sample_p_ek_gpu           : Callable = kernels[8]
        self.calc_sample_hhi_gpu            : Callable = kernels[9]
        self.calc_sample_count_i_gpu        : Callable = kernels[10]
        self.calc_sample_count_it_gpu       : Callable = kernels[11]
        self.calc_sample_count_ik_gpu       : Callable = kernels[12]
        self.calc_sample_mpc_gpu            : Callable = kernels[13]
        self.calc_sample_p_ix_gpu           : Callable = kernels[14]

        self._edges_attrs_d = self.cuda.device_array((self._gloader.n_unique_edges,self._gloader.n_attr),dtype=np.float32, stream=self.stream)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB = self.TPB_2D)

        self.init_sample_p_et_gpu[blockspergrid,threadsperblock,self.stream](
            self._edges_attrs_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB = self.TPB_2D)

        self.edge_sample_p_et_gpu[blockspergrid,threadsperblock,self.stream](
            self._sampler.N_uv,
            self._edges_attrs_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB = self.TPB_1D)

        self.init_sample_count_i_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_i_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._sampler.n_walks), TPB = self.TPB_2D)

        self.calc_sample_count_i_gpu[blockspergrid,threadsperblock,self.stream](
            self._sampler._T_d,
            self._w_i_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,self._gloader.n_attr), TPB = self.TPB_2D)

        self.init_sample_count_it_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_it_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._sampler.n_walks), TPB = self.TPB_2D)

        self.calc_sample_count_it_gpu[blockspergrid,threadsperblock,self.stream](
            self._sampler._T_d,
            self._w_it_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)

        self.calc_sample_p_ix_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_it_d,
            self._p_it_d
        )

    @cpu_gpu_dispatcher('_update_hhi_scores_cpu', '_update_hhi_scores_gpu')
    def _update_hhi_scores(self, **kwargs):
        pass

    def _update_hhi_scores_cpu(self):
        #################################
        # merge values

        edge_sample_p_ek_cpu(
            samples = self._sampler.N_uv,
            membership = self._membership,
            counts = self._edges_communities
        )

        edge_counts = self._edges_communities / self._sampler.n_samples_edge

        hhi_estimate = edge_counts**2
        self._hhi_edges = np.nansum(hhi_estimate,axis=1)

        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = self._hhi_edges[i]

    def _update_hhi_scores_gpu(self):

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB = self.TPB_2D)

        self.init_sample_p_ek_gpu[blockspergrid,threadsperblock,self.stream](
            self._edges_communities_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB = self.TPB_2D)

        self.edge_sample_p_ek_gpu[blockspergrid,threadsperblock,self.stream](
            self._sampler._N_uv_d,
            self._membership_d,
            self._edges_communities_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges,), TPB = self.TPB_1D)

        self.init_sample_hhi_gpu[blockspergrid,threadsperblock,self.stream](
            self._hhi_edges_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB = self.TPB_2D)

        self.calc_sample_hhi_gpu[blockspergrid,threadsperblock,self.stream](
            self._edges_communities_d,
            self._hhi_edges_d,
            self._sampler.n_samples_edge
        )

        self.stream.synchronize()

        hhi_estimate = self._hhi_edges_d.copy_to_host()

        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = hhi_estimate[i]

    @cpu_gpu_dispatcher('_update_mpc_scores_cpu', '_update_mpc_scores_gpu')
    def _update_mpc_scores(self):
        pass

    def _update_mpc_scores_cpu(self):

        self._w_ik = np.zeros((self._gloader.n_nodes_spatial,self._n_communities),dtype=np.float32)
        calc_sample_w_ik_cpu(
            T_ijk = self._sampler._T,
            membership = self._membership,
            w_ik = self._w_ik
        )

        # estimate p_ik
        self._p_ik = self._w_ik / (self._sampler.n_walks * self._sampler.walk_length)

        mpc_array = np.zeros((self._gloader.n_nodes_spatial,self._gloader.n_attr),dtype=np.float32)
        calc_sample_mpc_cpu(
            w_it = self._w_it,
            w_ik = self._w_ik,
            membership = self._membership,
            mpc = mpc_array
        )

        mpc_array /= (self._sampler.walk_length * self._sampler.n_walks)**2

        self._mpc_attrs = np.nansum(mpc_array,axis=0) / mpc_array.shape[0]
        self._log_mpc_attrs = safe_log(self._mpc_attrs)

        # Update the mpc
        for attr in self._gloader.attr_list:
            self._mpc[attr] = self._mpc_attrs[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = self._log_mpc_attrs[self._gloader.attr_key2i[attr]]

    def _update_mpc_scores_gpu(self):

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB = self.TPB_2D)

        self.init_sample_count_ik_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_ik_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB = self.TPB_2D)

        self.calc_sample_count_ik_gpu[blockspergrid,threadsperblock,self.stream](
            self._sampler._T_d,
            self._membership_d,
            self._w_ik_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB = self.TPB_2D)

        self.calc_sample_p_ix_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_ik_d,
            self._p_ik_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_attr,), TPB = self.TPB_1D)

        self.init_sample_mpc_gpu[blockspergrid,threadsperblock,self.stream](
            self._mpc_attrs_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)

        self.calc_sample_mpc_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_it_d,
            self._w_ik_d,
            self._membership_d,
            self._mpc_attrs_d
        )

        self.stream.synchronize()

        mpc_estimate = self._mpc_attrs_d.copy_to_host() / self._gloader.n_nodes_spatial
        mpc_estimate /= (self._sampler.walk_length * self._sampler.n_walks)**2
        self._mpc_attrs = mpc_estimate
        self._log_mpc_attrs = safe_log(self._mpc_attrs)

        # Update the mpc
        for attr in self._gloader.attr_list:
            self._mpc[attr] = self._mpc_attrs[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = self._log_mpc_attrs[self._gloader.attr_key2i[attr]]

    def update_scores(self, communities_dict: Dict[str,str]):

        super().update_scores(communities_dict)

        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

        if self.use_gpu:
            self.stream.synchronize()

    def update_scores_membership(self, membership: np.ndarray):
        super().update_scores_membership(membership)

        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

        if self.use_gpu:
            self.stream.synchronize()

    def save(self, directory=None, **kwargs):
        super().save(directory, **kwargs)
        self._sampler.save(directory, **kwargs)

    def load(self, directory=None, **kwargs):
        super().load(directory, **kwargs)
        self._sampler.load(directory, **kwargs)

    def _create_copy_instance(self, stream: Optional[Any] = None) -> 'SampleEvaluator':
        if self.use_gpu:
            return self.__class__(
                gloader=self._gloader,
                sampler=self._sampler,
                use_gpu=self.use_gpu,
                n_communities=self._n_communities,
                threads_per_block_1D=self.TPB_1D,
                threads_per_block_2D=self.TPB_2D,
                threads_per_block_3D=self.TPB_3D,
                stream=stream
            )
        else:
            return self.__class__(
                gloader=self._gloader,
                sampler=self._sampler,
                use_gpu=self.use_gpu,
                n_communities=self._n_communities
            )

    def copy_attributes(self, target: 'SampleEvaluator'):
        super().copy_attributes(target)

        target._edges_attrs = self.edges_attrs

    def copy_gpu_attributes(self, target: 'SampleEvaluator', stream: Any):
        super().copy_gpu_attributes(target, stream)

        target._edges_attrs_d = self.cuda.to_device(self.edges_attrs, stream=stream)

    def __getstate__(self):
        state = super().__getstate__()

        if self.use_gpu:

            safe_delete(state, '_edges_attrs_d')

            state.pop('init_sample_p_et_gpu', None)
            state.pop('init_sample_p_ek_gpu', None)
            state.pop('init_sample_hhi_gpu', None)
            state.pop('init_sample_count_i_gpu', None)
            state.pop('init_sample_count_it_gpu', None)
            state.pop('init_sample_count_ik_gpu', None)
            state.pop('init_sample_mpc_gpu', None)
            state.pop('edge_sample_p_et_gpu', None)
            state.pop('edge_sample_p_ek_gpu', None)
            state.pop('calc_sample_hhi_gpu', None)
            state.pop('calc_sample_count_i_gpu', None)
            state.pop('calc_sample_count_it_gpu', None)
            state.pop('calc_sample_count_ik_gpu', None)
            state.pop('calc_sample_mpc_gpu', None)

        return state

    def __del__(self):
        super().__del__()

        if self.use_gpu and self._init_gpu:
            if hasattr(self, '_edges_attrs_d'):
                del self._edges_attrs_d


class EvaluatorManager:
    def __init__(self, num_streams: int, evaluator: BaseEvaluator) -> None:
        self.evaluators = [evaluator.copy() for i in range(num_streams)]
        self.locks = [threading.Lock() for _ in range(num_streams)]

    def get_evaluator(self) -> Tuple[BaseEvaluator, int, threading.Lock]:
        # Example strategy: find the first available evaluator
        for i, lock in enumerate(self.locks):
            if lock.acquire(True):  # blocking
                return self.evaluators[i], i, lock

        raise RuntimeError("No available evaluators")


class SampleWeightedEvaluator(BaseEvaluator):
    """
    Sample-based weighted evaluator that estimates p_it and p_ik using random walks with weighted sampling.
    
    This evaluator implements the sampling-based weighted approach where:
    - Random walks generate samples V_{i,j,k}, T_{i,j,k}, W_{i,j,k}
    - S_hat_t(i,L) and S_hat_k(i,L) are estimated from weighted samples
    - Normalized probabilities p_hat(t|i,L) and p_hat(k|i,L) are computed
    - MPC scores are estimated from the weighted sampling distributions
    
    Parameters
    ----------
    gloader : GraphLoader
        Graph loader containing the multi-digraph data
    sampler : NeighbourSampler
        Random walk sampler providing T, W tensors
    use_gpu : bool, optional
        Whether to use GPU acceleration (default: False). Currently only CPU is supported.
    n_communities : int, optional
        Number of communities/alliances (default: number of attributes)
    threads_per_block_1D : int, optional
        GPU threads per block for 1D kernels (default: 484)
    threads_per_block_2D : Union[int, Tuple[int,int]], optional
        GPU threads per block for 2D kernels (default: 22)
    threads_per_block_3D : Union[int, Tuple[int,int,int]], optional
        GPU threads per block for 3D kernels (default: 8)
    stream : optional
        CUDA stream for GPU operations (default: None)
    """
    
    def __init__(
            self,
            gloader: 'GraphLoader',
            sampler: 'NeighbourSampler',
            use_gpu: Optional[bool] = False,
            n_communities: Optional[int] = None,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            stream: Optional[Any] = None
        ) -> None:
        
        self._sampler = sampler
        super().__init__(
            gloader = gloader,
            use_gpu = use_gpu,
            n_communities = n_communities,
            threads_per_block_1D = threads_per_block_1D,
            threads_per_block_2D = threads_per_block_2D,
            threads_per_block_3D = threads_per_block_3D,
            stream = stream
        )

    @property
    def edges_attrs(self):
        if self.use_gpu:
            return self._edges_attrs_d.copy_to_host()
        else:
            return self._edges_attrs.copy()

    @property
    def s_t(self):
        """Weighted attribute scores S_hat_t(i, L)."""
        if self.use_gpu:
            return self._s_t_d.copy_to_host()
        else:
            return self._s_t.copy()

    @property
    def s_k(self):
        """Weighted alliance scores S_hat_k(i, L).""" 
        if self.use_gpu:
            return self._s_k_d.copy_to_host()
        else:
            return self._s_k.copy()

    @property
    def s_total(self):
        """Total capacity scores S_hat_total(i, L)."""
        if self.use_gpu:
            return self._s_total_d.copy_to_host()
        else:
            return self._s_total.copy()

    def _post_init_cpu(self, **kwargs):
        super()._post_init_cpu(**kwargs)

        self._edges_attrs = np.zeros((self._gloader.n_unique_edges,self._gloader.n_attr))

        edge_sample_p_et_cpu(
            samples = self._sampler.N_uv,
            counts = self._edges_attrs
        )

        # Initialize weighted score arrays
        self._s_t = np.zeros((self._gloader.n_nodes_spatial, self._gloader.n_attr), dtype=np.float32)
        self._s_k = np.zeros((self._gloader.n_nodes_spatial, self.n_communities), dtype=np.float32)
        self._s_total = np.zeros((self._gloader.n_nodes_spatial,), dtype=np.float32)

        # Calculate membership-independent scores from sampling using W_total for consistency
        calc_sample_s_t_weighted_cpu(
            T_ijk = self._sampler.T,
            W_total_ijk = self._sampler.W_total,
            s_t = self._s_t
        )

        calc_sample_s_total_weighted_cpu(
            W_total_ijk = self._sampler.W_total,
            s_total = self._s_total
        )

        # Calculate normalized attribute probabilities (doesn't depend on membership)
        calc_sample_normalized_p_t_weighted_cpu(
            s_t = self._s_t,
            s_total = self._s_total,
            p_t_normalized = self._p_it
        )

        # Note: s_k and p_k_normalized will be calculated in _update_scores methods
        # when membership information becomes available

    def _post_init_gpu(
            self,
            threads_per_block_1D: int,
            threads_per_block_2D: Union[int,Tuple[int,int]],
            threads_per_block_3D: Union[int,Tuple[int,int,int]],
            stream: Optional[Any] = None
        ):
        """GPU initialization for SampleWeightedEvaluator."""
        super()._post_init_gpu(threads_per_block_1D, threads_per_block_2D, threads_per_block_3D, stream)

        from .utilities_cuda import define_cuda_kernels_stochastic
        from .utilities_weighted_cuda import define_cuda_kernels_weighted_sampling

        # Load base stochastic kernels (needed for edge sampling and HHI computation)
        base_kernels = define_cuda_kernels_stochastic(n_walks=self._sampler.n_walks, walk_length=self._sampler.walk_length)
        self.init_sample_p_et           : Callable = base_kernels[0]
        self.init_sample_p_ek           : Callable = base_kernels[1]
        self.init_sample_hhi            : Callable = base_kernels[2]
        self.init_sample_count_i        : Callable = base_kernels[3]
        self.init_sample_count_it       : Callable = base_kernels[4]
        self.init_sample_count_ik       : Callable = base_kernels[5]
        self.init_sample_mpc            : Callable = base_kernels[6]
        self.edge_sample_p_et           : Callable = base_kernels[7]
        self.edge_sample_p_ek           : Callable = base_kernels[8]
        self.calc_sample_hhi            : Callable = base_kernels[9]
        self.calc_sample_count_i        : Callable = base_kernels[10]
        self.calc_sample_count_it       : Callable = base_kernels[11]
        self.calc_sample_count_ik       : Callable = base_kernels[12]
        self.calc_sample_mpc            : Callable = base_kernels[13]

        # Load weighted sampling kernels
        weighted_kernels = define_cuda_kernels_weighted_sampling(
            n_attr=self._gloader.n_attr,
            n_nodes=self._gloader.n_nodes_spatial,
            n_communities=self._n_communities
        )
        
        self.calc_sample_s_t_weighted_gpu               : Callable = weighted_kernels[0]
        self.calc_sample_s_k_weighted_gpu               : Callable = weighted_kernels[1]
        self.calc_sample_s_total_weighted_gpu           : Callable = weighted_kernels[2]
        self.calc_sample_normalized_p_t_weighted_gpu    : Callable = weighted_kernels[3]
        self.calc_sample_normalized_p_k_weighted_gpu    : Callable = weighted_kernels[4]
        self.calc_sample_mpc_weighted_gpu               : Callable = weighted_kernels[5]

        self._edges_attrs_d = self.cuda.device_array((self._gloader.n_unique_edges,self._gloader.n_attr),dtype=np.float32, stream=self.stream)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB = self.TPB_2D)
        self.init_sample_p_et[blockspergrid,threadsperblock,self.stream](
            self._edges_attrs_d
        )

        self.stream.synchronize()

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB = self.TPB_2D)
        self.edge_sample_p_et[blockspergrid,threadsperblock,self.stream](
            self._sampler.N_uv,
            self._edges_attrs_d
        )

        # Allocate weighted score arrays
        self._s_t_d = self.cuda.device_array((self._gloader.n_nodes_spatial, self._gloader.n_attr), dtype=np.float32, stream=self.stream)
        self._s_k_d = self.cuda.device_array((self._gloader.n_nodes_spatial, self._n_communities), dtype=np.float32, stream=self.stream)
        self._s_total_d = self.cuda.device_array((self._gloader.n_nodes_spatial,), dtype=np.float32, stream=self.stream)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB = self.TPB_1D)
        self.init_sample_count_i[blockspergrid,threadsperblock,self.stream](
            self._s_total_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,self._gloader.n_attr), TPB = self.TPB_2D)
        self.init_sample_count_it[blockspergrid,threadsperblock,self.stream](
            self._s_t_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,self._gloader.n_attr), TPB = self.TPB_2D)
        self.init_sample_count_it[blockspergrid,threadsperblock,self.stream](
            self._p_it_d
        )

        self.stream.synchronize()

        # Calculate membership-independent scores from sampling using W_total for consistency
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB=self.TPB_2D)
        self.calc_sample_s_t_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._sampler._T_d,
            self._sampler._W_total_d,
            self._s_t_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB=self.TPB_1D)
        self.calc_sample_s_total_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._sampler._W_total_d,
            self._s_total_d
        )

        self.stream.synchronize()

        # Calculate normalized attribute probabilities (doesn't depend on membership)
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB=self.TPB_2D)
        self.calc_sample_normalized_p_t_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._s_t_d,
            self._s_total_d,
            self._p_it_d
        )

        self.stream.synchronize()

        # Note: s_k and p_k_normalized will be calculated in _update_scores methods
        # when membership information becomes available

    def update_scores(self, communities_dict: Dict[str,str]):
        """Update scores using new community assignments."""
        super().update_scores(communities_dict)
        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

    def update_scores_membership(self, membership: np.ndarray):
        """Update scores using new membership array."""
        super().update_scores_membership(membership)
        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

    @cpu_gpu_dispatcher('_update_hhi_scores_cpu', '_update_hhi_scores_gpu')
    def _update_hhi_scores(self):
        pass

    def _update_hhi_scores_cpu(self):
        """Calculate HHI scores - same as existing SampleEvaluator."""
        # HHI calculation remains identical to SampleEvaluator
        # We can reuse the edge sampling approach
        edge_sample_p_ek_cpu(
            samples = self._sampler.N_uv,
            membership = self._membership,
            counts = self._edges_communities
        )

        edge_counts = self._edges_communities / self._sampler.n_samples_edge

        hhi_estimate = edge_counts**2
        self._hhi_edges = np.nansum(hhi_estimate,axis=1)

        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = self._hhi_edges[i]

    def _update_hhi_scores_gpu(self):
        """GPU version of HHI score computation using existing sampling kernels."""
        # Initialize edge communities
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB=self.TPB_2D)
        self.init_sample_p_ek[blockspergrid, threadsperblock, self.stream](
            self._edges_communities_d
        )

        # Sample edge communities
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._sampler.n_samples_edge), TPB=self.TPB_2D)
        self.edge_sample_p_ek[blockspergrid, threadsperblock, self.stream](
            self._sampler._N_uv_d,
            self._membership_d,
            self._edges_communities_d
        )

        # Calculate HHI from edge communities
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges,), TPB=self.TPB_1D)
        self.init_sample_hhi[blockspergrid, threadsperblock, self.stream](
            self._hhi_edges_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._n_communities), TPB=self.TPB_2D)
        self.calc_sample_hhi[blockspergrid, threadsperblock, self.stream](
            self._edges_communities_d,
            self._hhi_edges_d,
            self._sampler.n_samples_edge
        )

        self.stream.synchronize()

        hhi_edges = self._hhi_edges_d.copy_to_host()

        # Update the hhi dictionary
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = hhi_edges[i]

    @cpu_gpu_dispatcher('_update_mpc_scores_cpu', '_update_mpc_scores_gpu')
    def _update_mpc_scores(self):
        pass

    def _update_mpc_scores_cpu(self):
        """Calculate MPC scores using weighted sampling approach."""
        
        # Recalculate weighted alliance scores with new membership using W_total for consistency
        calc_sample_s_k_weighted_cpu(
            T_ijk = self._sampler.T,
            W_total_ijk = self._sampler.W_total,
            membership = self._membership,
            s_k = self._s_k
        )

        # Recalculate normalized alliance probabilities
        calc_sample_normalized_p_k_weighted_cpu(
            s_k = self._s_k,
            s_total = self._s_total,
            p_k_normalized = self._p_ik
        )

        # Calculate MPC using weighted sampling approach
        calc_sample_mpc_weighted_cpu(
            membership = self._membership,
            p_t_normalized = self._p_it,
            p_k_normalized = self._p_ik,
            mpc = self._mpc_attrs
        )

        self._log_mpc_attrs = safe_log(self._mpc_attrs)

        # Update the mpc dictionaries
        for attr in self._gloader.attr_list:
            attr_index = self._gloader.attr_key2i[attr]
            self._mpc[attr] = float(self._mpc_attrs[attr_index])
            self._log_mpc[attr] = float(self._log_mpc_attrs[attr_index])

    def _update_mpc_scores_gpu(self):
        """GPU version of MPC score computation using weighted sampling approach."""
        
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB=self.TPB_2D)
        self.init_sample_count_ik[blockspergrid, threadsperblock, self.stream](
            self._s_k_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB = self.TPB_2D)
        self.init_sample_count_ik[blockspergrid,threadsperblock,self.stream](
            self._p_ik_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_attr,), TPB = self.TPB_1D)
        self.init_sample_mpc[blockspergrid,threadsperblock,self.stream](
            self._mpc_attrs_d
        )

        self.stream.synchronize()

        # Recalculate weighted alliance scores with new membership using W_total for consistency
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB=self.TPB_2D)
        self.calc_sample_s_k_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._sampler._T_d,
            self._sampler._W_total_d,
            self._membership_d,
            self._s_k_d
        )

        self.stream.synchronize()

        # Recalculate normalized alliance probabilities
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._n_communities), TPB = self.TPB_2D)
        self.calc_sample_normalized_p_k_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._s_k_d,
            self._s_total_d,
            self._p_ik_d
        )

        self.stream.synchronize()

        # Calculate MPC using weighted sampling approach
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_attr, self._gloader.n_nodes_spatial), TPB=self.TPB_2D)
        self.calc_sample_mpc_weighted_gpu[blockspergrid, threadsperblock, self.stream](
            self._membership_d,
            self._p_it_d,
            self._p_ik_d,
            self._mpc_attrs_d
        )

        self.stream.synchronize()

        self._mpc_attrs = self._mpc_attrs_d.copy_to_host()
        self._log_mpc_attrs = safe_log(self._mpc_attrs)

        # Update the mpc dictionaries
        for attr in self._gloader.attr_list:
            attr_index = self._gloader.attr_key2i[attr]
            self._mpc[attr] = float(self._mpc_attrs[attr_index])
            self._log_mpc[attr] = float(self._log_mpc_attrs[attr_index])

    def _create_copy_instance(self, stream: Optional[Any] = None) -> 'SampleWeightedEvaluator':
        if self.use_gpu:
            return self.__class__(
                gloader=self._gloader,
                sampler=self._sampler,
                use_gpu=self.use_gpu,
                n_communities=self._n_communities,
                threads_per_block_1D=self.TPB_1D,
                threads_per_block_2D=self.TPB_2D,
                threads_per_block_3D=self.TPB_3D,
                stream=stream
            )
        else:
            return self.__class__(
                gloader=self._gloader,
                sampler=self._sampler,
                use_gpu=self.use_gpu,
                n_communities=self._n_communities
            )

    def copy_attributes(self, target: 'SampleWeightedEvaluator'):
        super().copy_attributes(target)

        target._edges_attrs = self.edges_attrs
        target._s_t = self.s_t.copy()
        target._s_k = self.s_k.copy()
        target._s_total = self.s_total.copy()

    def copy_gpu_attributes(self, target: 'SampleWeightedEvaluator', stream: Any):
        super().copy_gpu_attributes(target, stream)

        target._edges_attrs_d = self.cuda.to_device(self.edges_attrs, stream=stream)
        target._s_t_d = self.cuda.to_device(self.s_t, stream=stream)
        target._s_k_d = self.cuda.to_device(self.s_k, stream=stream)
        target._s_total_d = self.cuda.to_device(self.s_total, stream=stream)

    def __getstate__(self):
        state = super().__getstate__()
        # Remove sampler from state if using GPU to avoid CUDA context issues
        if self.use_gpu:
            safe_delete(state, '_edges_attrs_d')
            safe_delete(state, '_s_t_d')
            safe_delete(state, '_s_k_d')
            safe_delete(state, '_s_total_d')

            state.pop('init_sample_p_et', None)
            state.pop('init_sample_p_ek', None)
            state.pop('init_sample_hhi', None)
            state.pop('init_sample_count_i', None)
            state.pop('init_sample_count_it', None)
            state.pop('init_sample_count_ik', None)
            state.pop('init_sample_mpc', None)
            state.pop('edge_sample_p_et', None)
            state.pop('edge_sample_p_ek', None)
            state.pop('calc_sample_hhi', None)
            state.pop('calc_sample_count_i', None)
            state.pop('calc_sample_count_it', None)
            state.pop('calc_sample_count_ik', None)
            state.pop('calc_sample_mpc', None)

        return state

    def __del__(self):
        """Destructor."""
        if self.use_gpu and self._init_gpu:
            # Clean up additional GPU arrays allocated by SampleWeightedEvaluator
            if hasattr(self, '_edges_attrs_d'):
                del self._edges_attrs_d
            if hasattr(self, '_s_t_d'):
                del self._s_t_d
            if hasattr(self, '_s_k_d'):
                del self._s_k_d
            if hasattr(self, '_s_total_d'):
                del self._s_total_d
        super().__del__()


class BaseOptimizer(ABC):

    def __init__(self, evaluator: BaseEvaluator, seed: int) -> None:
        self._evaluator = evaluator
        self.seed = seed

        self._communities_history = []
        self._f_history = []
        self._multi_history = []

        self.random_state: Tuple[random.Random,np.random.RandomState] = random_state(self.seed)

        self._f1: float = None
        self._f2: float = None
        self._f = -float('inf')  # Start with the worst possible increase

        self._communities_dict = {}

    def _post_init_cpu(self, **kwargs):
        pass

    def _post_init_gpu(
            self,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
        ):

        assert self.use_gpu, "error: use_gpu flag set to False"
        from .utilities_cuda import cuda
        self.cuda = cuda

        self.TPB_1D = threads_per_block_1D
        self.TPB_2D = threads_per_block_2D
        self.TPB_3D = threads_per_block_3D
        self._init_gpu = True

    @property
    def f(self) -> float:
        return self._f

    @property
    def f1(self) -> float:
        return self._f1

    @property
    def f2(self) -> float:
        return self._f2

    @property
    def communities_dict(self) -> Dict[str,int]:
        return self._communities_dict.copy()

    @property
    def communities_history(self) -> List[Dict[str,int]]:
        return self._communities_history.copy()

    @property
    def f_history(self) -> List[float]:
        return self._f_history.copy()

    @property
    def multi_history(self) -> List[Tuple[float,float]]:
        return self._multi_history.copy()

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    @cpu_gpu_dispatcher('_post_init_cpu', '_post_init_gpu')
    def _post_init(self, **kwargs):
        pass

    @_jsonmode
    def save(self, directory=None, **kwargs):
        data_to_save = {
            'communities_history': self.communities_history,
            'f_history': self.f_history,
            'multi_history': self.multi_history
        }

        for name, data in data_to_save.items():
            with open(f'{directory}/{name}.json', 'w') as f:
                json.dump(serialize(data), f, indent=4)

        self._evaluator.save(directory, **kwargs)

    @_jsonmode
    def load(self, directory=None, **kwargs):
        data_to_load = {
            'communities_history':  {"name" : '_communities_history', "dtype" : List[Dict[str,int]]},
            'f_history':            {"name" : '_f_history', "dtype" : list},
            'multi_history':        {"name" : '_multi_history', "dtype" : List[Tuple[float,float]]},
        }

        for name, data_dict in data_to_load.items():
            with open(f'{directory}/{name}.json', 'r') as f:
                data = deserialize(json.load(f))
                dtype = data_dict["dtype"]
                data = dtype_load_func.get(dtype, load_dict)(data)
                setattr(self, data_dict["name"], data)

        self._evaluator.load(directory, **kwargs)


class BruteForce(BaseOptimizer):

    def __init__(
            self,
            evaluator: ExactEvaluator | SampleEvaluator,
            seed: int,
            weights: Optional[Tuple[float,float]] = (0.5,0.5),
            num_streams: Optional[int] = 1,
            use_gpu: Optional[bool] = False
        ) -> None:
        super().__init__(evaluator, seed)

        self.weights = weights
        self.num_streams = num_streams
        self.use_gpu = use_gpu

        self._memberships = list(generate_possible_memberships(
            self._evaluator._gloader.n_attr,
            self._evaluator.n_communities
        ))

        self._results_library = {}

        self._post_init()

    def _post_init_cpu(self, **kwargs):
        super()._post_init_cpu(**kwargs)
        self._eval_manager = EvaluatorManager(self.num_streams, self._evaluator)

    def _post_init_gpu(
            self,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
        ):
        super()._post_init_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D)
        self._eval_manager = EvaluatorManager(self.num_streams, self._evaluator)

    @property
    def results_library(self) -> Dict[int, Any]:
        return deepcopy(self._results_library)

    def optimize(
            self,
            max_iterations: Optional[int] = None,
            rtol: Optional[float] = 1e-12,
            multiprocess: Optional[bool] = True,
        ):

        max_iterations = max_iterations or len(self._memberships)


        if multiprocess:
            vargs_iterator = [(membership,) for i,membership in enumerate(self._memberships)]
            vkwargs_iterator = [{},] * len(vargs_iterator)
            fargs = [
                self.weights,
                self._evaluator,
            ]
            fkwargs = {}

            results = parallel_sampling(
                self._process_batch,
                vargs_iterator,
                vkwargs_iterator,
                fargs,
                fkwargs,
                n_jobs=self.num_streams
            )

        else:
            with tqdm(total=len(self._memberships), desc="memberships evaluated", position=0, leave=False) as pbar_outer:

                # Use ThreadPoolExecutor to parallelize the loop
                results = []
                with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
                    futures = []
                    for batch_index, (membership) in enumerate(self._memberships):
                        future = executor.submit(
                            self._process_batch_thread,
                            membership,
                            self.weights,
                            self._eval_manager,
                        )
                        futures.append(future)

                    # Collect results and update max_increase and best_x
                    for i,future in enumerate(futures):
                        results.append(future.result())
                        pbar_outer.update(1)

        for i, (x, f, hhi, mpc) in enumerate(results):

            # Calculate the initial mean hhi over all routes
            f1 = -np.mean(hhi)
            f2 = np.mean(mpc)

            self._results_library[i] = {
                'membership': x,
                'f': f,
                'f1': f1,
                'f2': f2
            }

        self.postprocess_results(rtol)

    def postprocess_results(self, rtol: Optional[float] = 1e-12):

        self._communities_history = []
        self._f_history = []
        self._multi_history = []

        with tqdm(total=len(self._memberships), desc="postprocessing results", position=0, leave=False) as pbar_post:
            for i, values in self._results_library.items():
                x = values['membership']
                f = values['f']

                # log history
                if f - self._f > rtol:

                    self._evaluator.update_scores_membership(x)

                    # Calculate the initial mean hhi over all routes
                    self._f1 = -np.mean(list(self._evaluator.hhi.values()))
                    self._f2 = np.mean(list(self._evaluator.log_mpc.values()))
                    self._f = multiobjective(self._f1,self._f2,self.weights)

                    self._communities_dict = self._evaluator.communities_dict
                    self._communities_history.append(self._communities_dict)
                    self._f_history.append(self._f)
                    self._multi_history.append((self._f1,self._f2))

                pbar_post.update(1)

    @staticmethod
    def _process_batch(
            membership: np.ndarray,
            weights: Tuple[float,float],
            evaluator: ExactEvaluator | SampleEvaluator,
        ):
        # Perform stochastic evaluation using the selected stream

        evaluator.update_scores_membership(membership)

        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f = multiobjective(f1, f2, weights=weights)

        return membership, f, list(evaluator.hhi.values()), list(evaluator.log_mpc.values())

    @staticmethod
    def _process_batch_thread(
            membership: np.ndarray,
            weights: Tuple[float,float],
            eval_manager: EvaluatorManager,
        ):

        evaluator, stream_id, lock = eval_manager.get_evaluator()
        try:
            evaluator.update_scores_membership(membership)
        finally:
            lock.release()

        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f = multiobjective(f1, f2, weights=weights)

        return membership, f, list(evaluator.hhi.values()), list(evaluator.log_mpc.values())

    @_jsonmode
    def save(self, directory=None, **kwargs):
        super().save(directory, **kwargs)


        data_to_save = {
            'results_library': self._results_library,
        }

        for name, data in data_to_save.items():
            with open(f'{directory}/{name}.json', 'w') as f:
                json.dump(serialize(data), f, indent=4)

    @_jsonmode
    def load(self, directory=None, **kwargs):
        super().load(directory, **kwargs)

        data_to_load = {
            'results_library':  {"name" : '_results_library', "dtype" : Dict[int, Any]},
        }

        for name, data_dict in data_to_load.items():
            with open(f'{directory}/{name}.json', 'r') as f:
                data = deserialize(json.load(f))
                dtype = data_dict["dtype"]
                data = dtype_load_func.get(dtype, load_dict)(data)
                setattr(self, data_dict["name"], data)


class GreedyOptimizer(BaseOptimizer):

    def __init__(
            self,
            evaluator: ExactEvaluator | SampleEvaluator,
            seed: int,
            communities_dict: Optional[Dict[str,int]] = None,
            weights: Optional[Tuple[float,float]] = (0.5,0.5),
            num_streams: Optional[int] = 1,
            use_gpu: Optional[bool] = False
        ) -> None:
        super().__init__(evaluator, seed)

        if communities_dict is None:
            self._communities_dict = {attr: i for i, attr in enumerate(self._evaluator._gloader.attr_list)}
        else:
            self._communities_dict = communities_dict

        self._n_evals = []
        self._n_iterations = []

        self.weights = weights
        self.num_streams = num_streams
        self.use_gpu = use_gpu

        self._evaluator.update_scores(self._communities_dict)
        self.n_communities = self._evaluator.n_distinct_communities

        # Calculate the initial mean hhi over all routes
        self._f1 = -np.mean(list(self._evaluator.hhi.values()))
        self._f2 = np.mean(list(self._evaluator.log_mpc.values()))
        self._f = multiobjective(self._f1,self._f2,self.weights)

        # log history
        self._communities_history.append(self._communities_dict)
        self._f_history.append(self._f)
        self._multi_history.append((self._f1,self._f2))
        self._n_evals.append(0)
        self._n_iterations.append(0)

        self._membership_pairs = {}

        self._post_init()

    def _post_init_cpu(self, **kwargs):
        super()._post_init_cpu(**kwargs)
        self._eval_manager = EvaluatorManager(self.num_streams, self._evaluator)

    def _post_init_gpu(
            self,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
        ):
        super()._post_init_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D)
        self._eval_manager = EvaluatorManager(self.num_streams, self._evaluator)

    @property
    def membership_pairs(self) -> np.ndarray:
        return self._membership_pairs.copy()

    @property
    def n_evals(self) -> List[int]:
        return self._n_evals.copy()

    @property
    def n_iterations(self) -> List[int]:
        return self._n_iterations.copy()

    def optimize(
            self,
            max_iterations: Optional[int] = None,
            epsilon: Optional[float] = 0.1,
            rtol: Optional[float] = 1e-12,
            n_sample_pairs: Optional[int] = None,
            multiprocess: Optional[bool] = True
        ):

        # Initialize flag for improvement
        improvement = True

        # Loop while improvement is possible
        n_iterations = 0
        n_evals = 0
        max_iterations = max_iterations if max_iterations is not None else self._evaluator.n_distinct_communities

        with tqdm(total=max_iterations, desc="Outer Loop", position=0, leave=False) as pbar_outer:
            while improvement and n_iterations < max_iterations:

                improvement = False
                max_increase = 0
                best_x = None

                self._membership_pairs = generate_membership_combinations(
                    self._evaluator.membership,
                    self._evaluator.n_communities,
                )

                # Create streams
                self._sampled_communities_pairs = set()
                improvement_sample = False
                n_feasible_pairs = len(self._membership_pairs) - len(self._sampled_communities_pairs)
                n_sample_pairs = _default_if_none(n_sample_pairs,n_feasible_pairs,'n_sample_pairs')

                with tqdm(total=n_feasible_pairs, desc="Inner Loop", position=1, leave=False) as pbar_inner:
                    while True:
                        sample_pairs = self._sample_community_pairs(n_sample_pairs, epsilon)

                        if not sample_pairs:
                            break

                        if multiprocess:
                            vargs_iterator = [[pairs[0],pairs[1]] for i,pairs in enumerate(sample_pairs)]
                            vkwargs_iterator = [{},] * len(vargs_iterator)
                            fargs = [
                                self.weights,
                                self._membership_pairs,
                                self._evaluator,
                                self._f
                            ]
                            fkwargs = {}

                            results = parallel_sampling(
                                self._process_batch,
                                vargs_iterator,
                                vkwargs_iterator,
                                fargs,
                                fkwargs,
                                n_jobs=self.num_streams,
                                show_pbar=False
                            )

                            pbar_inner.update(len(sample_pairs))
                            n_evals += len(sample_pairs)
                        else:
                            # Use ThreadPoolExecutor to parallelize the loop
                            results = []
                            with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
                                futures = []
                                for batch_index, (community_i, community_j) in enumerate(sample_pairs):
                                    future = executor.submit(
                                        self._process_batch_thread,
                                        community_i,
                                        community_j,
                                        self.weights,
                                        self._membership_pairs[(community_i, community_j)].copy(),
                                        self._eval_manager,
                                        self._f
                                    )
                                    futures.append(future)

                                # Collect results and update max_increase and best_x
                                for future in futures:
                                    increase, community_i, community_j = future.result()
                                    results.append((increase, community_i, community_j))
                                    pbar_inner.update(1)
                                    n_evals += 1

                        for increase, community_i, community_j in results:
                            if increase - max_increase > rtol:
                                max_increase = increase
                                best_x = (community_i, community_j)
                                improvement_sample = True

                        if self.use_gpu:
                            for evaluator in self._eval_manager.evaluators:
                                evaluator.stream.synchronize()

                        if improvement_sample:
                            break

                n_iterations += 1

                # If the maximum increase is greater than 0, merge these two communities and update the mean hhi
                if max_increase > 0:

                    self._evaluator.update_scores_membership(self.membership_pairs[best_x])
                    self._communities_dict = self._evaluator.communities_dict

                    # Calculate the initial mean hhi over all routes
                    self._f1 = -np.mean(list(self._evaluator.hhi.values()))
                    self._f2 = np.mean(list(self._evaluator.log_mpc.values()))
                    self._f = multiobjective(self._f1,self._f2,self.weights)
                    # self._f += max_increase
                    improvement = True

                    # log history
                    self._communities_history.append(self._communities_dict)
                    self._f_history.append(self._f)
                    self._multi_history.append((self._f1,self._f2))
                    self._n_evals.append(n_evals)
                    self._n_iterations.append(n_iterations)

                    self.n_communities -= 1

                pbar_outer.update(1)

    def _sample_community_pairs(self, n_samples: int = 5, epsilon: float = 0.1, replace: Optional[bool] = False) -> Union[list,bool]:

        rng_np = self.random_state[1]
        all_samples = list(self._membership_pairs.keys())

        sample_set = set(all_samples) - self._sampled_communities_pairs
        sample_list = list(sample_set)  # convert set to list

        if len(sample_list) == 0:
            return False

        probabilities = self._evaluator.w_k[sample_list] / np.sum(self._evaluator.w_k)
        probabilities = np.nan_to_num(probabilities, nan=0.0)
        probabilities[probabilities < 0] = 0.0

        probabilities_pairs = probabilities[:,0] * probabilities[:,1]

        probabilities_pairs = np.nan_to_num(probabilities_pairs, nan=0.0)
        probabilities_pairs[probabilities_pairs < 0] = 0.0

        if np.sum(probabilities_pairs) > 0:
            probabilities_pairs = probabilities_pairs / np.sum(probabilities_pairs)
        else:
            probabilities_pairs = np.ones_like(probabilities_pairs) / len(probabilities_pairs)

        if not replace:
            n_samples = min(len(sample_list),n_samples)

        if len(sample_list) == 0:
            return False

        max_samples = np.count_nonzero(probabilities_pairs) # to protect against non-zero probabilities

        n_samples_is = min(max_samples,math.ceil(n_samples * (1-epsilon)))
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

    @_jsonmode
    def save(self, directory=None, **kwargs):
        super().save(directory, **kwargs)

        data_to_save = {
            'n_evals': self.n_evals,
            'n_iterations': self.n_iterations
        }

        for name, data in data_to_save.items():
            with open(f'{directory}/{name}.json', 'w') as f:
                json.dump(serialize(data), f, indent=4)

    @_jsonmode
    def load(self, directory=None, **kwargs):
        super().load(directory, **kwargs)

        data_to_load = {
            'n_evals':  {"name" : '_n_evals', "dtype" : list},
            'n_iterations':  {"name" : '_n_iterations', "dtype" : list},
        }

        for name, data_dict in data_to_load.items():
            with open(f'{directory}/{name}.json', 'r') as f:
                data = deserialize(json.load(f))
                dtype = data_dict["dtype"]
                data = dtype_load_func.get(dtype, load_dict)(data)
                setattr(self, data_dict["name"], data)

    @staticmethod
    def _process_batch(
            community_i: int,
            community_j: int,
            weights: Tuple[float,float],
            membership_pairs: Dict[Tuple[int,int],np.ndarray],
            evaluator: ExactEvaluator | SampleEvaluator,
            f: float
        ) -> Tuple[float,int,int]:
        # Perform stochastic evaluation using the selected stream

        evaluator.update_scores_membership(membership_pairs[(community_i, community_j)])

        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f_new = multiobjective(f1, f2, weights=weights)

        increase = f_new - f

        return increase, community_i, community_j

    @staticmethod
    def _process_batch_thread(
            community_i: int,
            community_j: int,
            weights: Tuple[float,float],
            membership: np.ndarray,
            eval_manager: EvaluatorManager,
            f: float
        ) -> Tuple[float,int,int]:

        evaluator, stream_id, lock = eval_manager.get_evaluator()
        try:
            evaluator.update_scores_membership(membership)
        finally:
            lock.release()

        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f_new = multiobjective(f1, f2, weights=weights)

        increase = f_new - f

        return increase, community_i, community_j


class ExactOptimizer(BaseOptimizer):

    def __init__(
            self,
            evaluator: SampleEvaluator | ExactEvaluator,
            seed: int,
            weights: Optional[Tuple[float,float]] = (0.5,0.5),
            log_dir: Optional[str] = 'models',
            log_interval: Optional[int] = 100,
            model_name: Optional[str] = 'exact_optimizer',
            model_settings: Optional[Dict[str,Any]] = {}
        ) -> None:

        super().__init__(evaluator, seed)

        self._evaluator = evaluator
        self.weights = weights
        self._model_settings = model_settings
        self._model_settings['seed'] = seed
        self._log_dir = log_dir
        self._log_interval = log_interval
        self._model_name = model_name

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

        # model parameters
        self._K = list(range(self._evaluator.n_communities))
        self._E = self._evaluator._gloader.edge_unique_list
        self._V = list(range(self._evaluator._gloader.n_nodes_spatial))
        self._T = list(range(self._evaluator._gloader.n_attr))

        self._mip_logs = []
        self._simplex_logs = []
        self._barrier_logs = []
        self._mip_log_counter = 0

        self._x: Dict[Tuple[int,int], Var] = {} # Decision variables
        self.model_built = False
        self.optimal = False

    def process_communities_dict(self, x: Dict[Tuple[int, int], Var]) -> Dict[str, int]:

        assert self.model_built, "Model not built yet. Run build_model() first."
        assert len(x) > 0, "No decision variables found. Run build_model() first."

        communities_dict = {self._evaluator._gloader.attr_i2key[t]: None for t in self._T}

        for t in self._T:
            for k in self._K:
                if hasattr(x[t, k], 'X'):
                    if x[t, k].getAttr('X') > 0.5:
                        assert communities_dict[self._evaluator._gloader.attr_i2key[t]] is None, "error: multiple communities assigned to the same attribute"
                        communities_dict[self._evaluator._gloader.attr_i2key[t]] = k
        
        # number of attributes with a community assigned
        n_unassigned = sum(1 for v in communities_dict.values() if v is None)
        # Handle case where no community is assigned to an attribute by using the unused k values
        if n_unassigned > 0:
            raise ValueError(
                f"Error: {n_unassigned} attributes have no community assigned. "
                f"Please ensure that all attributes are assigned to a community. "
            )
        
        return communities_dict

    def process_communities_values(self, x: Dict[Tuple[int, int], int]) -> Dict[str, int]:

        assert self.model_built, "Model not built yet. Run build_model() first."
        assert len(x) > 0, "No decision variables found. Run build_model() first."

        communities_dict = {self._evaluator._gloader.attr_i2key[t]: None for t in self._T}

        for t in self._T:
            for k in self._K:
                if x[t, k] > 0.5:
                    assert communities_dict[self._evaluator._gloader.attr_i2key[t]] is None, "error: multiple communities assigned to the same attribute"
                    communities_dict[self._evaluator._gloader.attr_i2key[t]] = k

        return communities_dict

    @property
    def simplex_logs(self) -> List[Tuple[float,float,float,float,float,float]]:
        return self._simplex_logs.copy()

    @property
    def mip_logs(self) -> List[Tuple[float,float,float,float,float,float]]:
        return self._mip_logs.copy()

    @property
    def barrier_logs(self) -> List[Tuple[float,float,float,float,float,float]]:
        return self._barrier_logs.copy()

    @property
    def x(self) -> Dict[Tuple[int,int],bool]:
        return {key: value.X > 0.5 for key, value in self._x.items()}

    @property
    def x_opt(self) -> Dict[Tuple[int,int],bool]:
        if self.optimal:
            return self.x
        else:
            logger.warning("Model not solved to optimality. Returning None.")
            return None

    def _load_vars(self):
        variables = self._model.getVars()
        # reconstruct self._x[t, k]
        self._x = {}
        for var in variables:
            if var.varName.startswith('x_'):
                _, t, k = var.varName.split('_')
                t = int(t)
                k = int(k)
                self._x[(t, k)] = var

    def build_model(
            self, 
            log_approx: Optional[str] = 'EXPR', 
            slack: Optional[float] = None, 
            num_points: Optional[int] = -1, 
            PWL_TOL: Optional[float] = 1e-6,
            epsilon: Optional[float] = 1e-10
        ):
        """
        Build the optimization model.

        Parameters
        ----------
        log_approx : str, optional
            The method to approximate the log function, by default 'EXPR'
        slack : float, optional
            The slack value for the log constraint, by default None
        num_points : int, optional
            The number of points to use for the PWL approximation, by default -1
        PWL_TOL : float, optional
            The tolerance for the PWL approximation, by default 1e-6
        epsilon : float, optional
            The epsilon value to use for numerical stability, by default 1e-10
        """

        p_it = self._evaluator.p_it

        uv_dict = self._evaluator._gloader.edge_pair2ui
        p_et = self._evaluator.p_et

        n_edges = self._evaluator._gloader.n_unique_edges
        n_nodes = self._evaluator._gloader.n_nodes_spatial
        n_attr = self._evaluator._gloader.n_attr

        self._model = Model("Graph Optimization: {}".format(self._model_name)) # Initialize the model

        # Decision variables
        for t in self._T:
            for k in self._K:
                self._x[t, k] = self._model.addVar(vtype=GRB.BINARY, name=f"x_{t}_{k}")

        # Pre-compute the products if they are reused often
        p_ek_precomputed = {}
        for u,v in tqdm(self._E, desc="computing p_ek_precomputed"):
            for k in self._K:
                p_ek_precomputed[u, v, k] = quicksum(self._x[t, k] * p_et[uv_dict[(u, v)], t] for t in self._T if p_et[uv_dict[(u, v)], t] > 0)

        f1_expr = quicksum(p_ek_precomputed[u, v, k] ** 2 / n_edges for u, v in self._E for k in self._K)

        # Objective function: Linear weighted interaction part
        p_ik_exp = {}
        for i in tqdm(self._V, desc="computing p_ik_exp"):
            for k in self._K:
                p_ik_exp[i,k] = quicksum(self._x[t, k] * p_it[i, t] for t in self._T if p_it[i, t] > 0)

        # Create a PWL approximation of log(f2)
        min_val = epsilon
        max_val = 1 # Maximum expected value of f2_exp

        f2_exp = 0
        for t in tqdm(self._T, desc="Building f2 expression"):
            f2_t_exp = 0
            all_zeros = True
            for i in self._V:
                p_it_exp = p_it[i, t]
                for k in self._K:
                    if p_it_exp > 0:
                        f2_t_exp += ((self._x[t, k] * p_ik_exp[i,k] * p_it_exp) / (n_nodes))
                        all_zeros = False

            # take the log of f2_t_exp
            if not all_zeros:
                f2_t_var = self._model.addVar(name=f"f2_t_{t}", lb=min_val, ub=max_val)
                log_f2_t = self._model.addVar(name=f"log_f2_t_{t}", lb=np.log(min_val), ub=0)

                if slack is None:
                    self._model.addConstr(f2_t_var == f2_t_exp, f"Link_f2_t_{t}")
                else:
                    slack_var = self._model.addVar(lb=-slack, ub=slack, name=f"slack_{t}")
                    self._model.addConstr(f2_t_var == f2_t_exp + slack_var, f"Link_f2_t_{t}")

                if log_approx == 'PWL':
                    num_points = num_points if num_points > 0 else 100
                    breakpoints_x = np.logspace(np.log10(min_val), np.log10(max_val), num=num_points)
                    breakpoints_y = np.log(breakpoints_x)

                    # Add the PWL constraint that approximates log(f2)
                    self._model.addGenConstrPWL(f2_t_var, log_f2_t, breakpoints_x, breakpoints_y, name=f"pwl_log_f2_t_{t}")
                elif log_approx == 'EXPR':
                    # Adding the logarithmic constraint
                    self._model.addGenConstrLog(f2_t_var, log_f2_t, name=f"log_constraint_t_{t}", options="FuncPieces=%i FuncPieceError=%f"%(num_points, PWL_TOL))

                f2_exp += log_f2_t / n_attr

        # Set the objective
        self._model.setObjective(-self.weights[0]*f1_expr + self.weights[1]*f2_exp, GRB.MAXIMIZE)

        # Constraints: Ensure each attribute is in exactly one subset
        for t in self._T:
            self._model.addConstr(quicksum(self._x[t, k] for k in self._K) == 1, name=f"assign_{t}")

        # algorithmic parameters
        for key, value in self._model_settings.items():
            self._model.setParam(key, value)

        for t in self._T:
            k = t % len(self._K) # round-robin, or any greedy rule
            self._x[t, k].start = 1

        self.model_built = True

    def save(self, directory=None, **kwargs):
        super().save(directory, **kwargs)
        # Save self._model in LP format
        try:
            self._model.write(os.path.join(directory,"{}.lp".format(self._model_name)))
        except Exception as e:
            logger.error(f"[ExactOptimizer] Error saving model: {e}")

    def load(self, directory=None, **kwargs):
        super().load(directory, **kwargs)

        if os.path.exists(os.path.join(directory,"{}.lp".format(self._model_name))) is False:
            logger.error(f"[ExactOptimizer] Model file {self._model_name}.lp not found in {directory}. Please build the model first.")
        else:
            logger.info(f"[ExactOptimizer] Loading model from {directory}/{self._model_name}.lp")

            self._model = read(os.path.join(directory,"{}.lp".format(self._model_name)))
            self._load_vars()
            self.model_built = True

        if os.path.exists(os.path.join(self._log_dir, 'mip_solution.csv')) is False:
            logger.error(f"[ExactOptimizer] Log file mip_solution.csv not found in {self._log_dir}. Please run optimize() first.")
        else:
            logger.info(f"[ExactOptimizer] Loading MIP logs from {self._log_dir}/mip_solution.csv")
            with open(os.path.join(self._log_dir, 'mip_solution.csv'), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self._mip_logs.append(tuple(row))

        if os.path.exists(os.path.join(self._log_dir, 'simplex_solution.csv')) is False:
            logger.error(f"[ExactOptimizer] Log file simplex_solution.csv not found in {self._log_dir}. Please run optimize() first.")
        else:
            logger.info(f"[ExactOptimizer] Loading Simplex logs from {self._log_dir}/simplex_solution.csv")
            with open(os.path.join(self._log_dir, 'simplex_solution.csv'), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self._simplex_logs.append(tuple(row))

        if os.path.exists(os.path.join(self._log_dir, 'barrier_solution.csv')) is False:
            logger.error(f"[ExactOptimizer] Log file barrier_solution.csv not found in {self._log_dir}. Please run optimize() first.")
        else:
            logger.info(f"[ExactOptimizer] Loading Barrier logs from {self._log_dir}/barrier_solution.csv")
            with open(os.path.join(self._log_dir, 'barrier_solution.csv'), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self._barrier_logs.append(tuple(row))

    def optimize(self, wall_limit: Optional[float] = None):

        # prepare logs
        f = open(os.path.join(self._log_dir, 'mip_solution.csv'),'w'); f.write("runtime, objbst, objbnd, mip_gap, node_count, simplex_iterations\n")
        f.close()

        f = open(os.path.join(self._log_dir, 'simplex_solution.csv'),'w'); f.write("runtime, obj_val, primal_inf, dual_inf, simplex_iterations\n")
        f.close()

        f = open(os.path.join(self._log_dir, 'barrier_solution.csv'),'w'); f.write("itcnt, primobj, dualobj, priminf, dualinf, cmpl\n")
        f.close()

        t0 = time.time()

        def _mycallback(model: Model, where: int):

            if where == GRB.Callback.MIPSOL:
                # Check if termination was requested
                if hasattr(self, '_terminate_requested') and self._terminate_requested:
                    logger.info("[ExactOptimizer] Termination requested, capturing final solution in callback.")
                    model.terminate()
                
                x = {}  # Dictionary to store variable values
                for t in self._T:
                    for k in self._K:
                        var_name = f"x_{t}_{k}"
                        var = model.getVarByName(var_name)
                        if var is not None:  # Ensure the variable exists
                            x[(t, k)] = model.cbGetSolution(var)
                        else:
                            logger.warning(f"[ExactOptimizer] Variable {var_name} not found.")

                communities_dict = self.process_communities_values(x)
                self._evaluator.update_scores(communities_dict)
                self._f1 = -np.mean(list(self._evaluator.hhi.values()))
                self._f2 = np.mean(list(self._evaluator.log_mpc.values()))
                self._f = multiobjective(self._f1,self._f2,self.weights)

                self._f_history.append(self._f)
                self._communities_history.append(communities_dict)
                self._multi_history.append((self._f1,self._f2))

            if where == GRB.Callback.MIP:
                # Check if termination was requested
                if hasattr(self, '_terminate_requested') and self._terminate_requested:
                    logger.info("[ExactOptimizer] Termination requested in MIP callback.")
                    model.terminate()
                
                runtime = model.cbGet(GRB.Callback.RUNTIME)    # Runtime
                self._mip_log_counter += 1

                # Basic MIP info
                objbst = model.cbGet(GRB.Callback.MIP_OBJBST)  # Best objective
                objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)  # Best bound

                # Compute the MIP gap manually
                if objbst < GRB.INFINITY and objbnd > -GRB.INFINITY:
                    mip_gap = abs(objbst - objbnd) / (1.0 + abs(objbst))
                else:
                    mip_gap = None

                # Additional metrics
                node_count = model.cbGet(GRB.Callback.MIP_NODCNT)  # Node count
                simplex_iterations = model.cbGet(GRB.Callback.MIP_ITRCNT)  # Simplex iterations

                if objbst < GRB.INFINITY and objbnd > -GRB.INFINITY:
                    mip_gap = abs(objbst - objbnd) / (1.0 + abs(objbst))
                else:
                    mip_gap = None

                # log every 100 iterations
                if self._mip_log_counter % self._log_interval == 0:
                    # Format each variable appropriately before writing
                    mip_gap_str = 'None' if mip_gap is None else "{:.8f}".format(mip_gap)
                    node_count_str = 'None' if node_count is None else str(node_count)
                    simplex_iterations_str = 'None' if simplex_iterations is None else str(simplex_iterations)
                    log_entry = "{:.8e},{:.8e},{:.8e},{},{},{}\n".format(
                        runtime, objbst, objbnd, mip_gap_str, node_count_str, simplex_iterations_str
                    )
                    with open(os.path.join(self._log_dir, 'mip_solution.csv'), 'a') as f:
                        f.write(log_entry)

                    self._mip_logs.append((runtime, objbst, objbnd, mip_gap, node_count, simplex_iterations))

            elif where == GRB.Callback.SIMPLEX:
                # Logging during simplex iterations
                simplex_iter = model.cbGet(GRB.Callback.SPX_ITRCNT)
                obj_val = model.cbGet(GRB.Callback.SPX_OBJVAL)
                primal_inf = model.cbGet(GRB.Callback.SPX_PRIMINF)
                dual_inf = model.cbGet(GRB.Callback.SPX_DUALINF)
                
                log_entry = "{:.8e},{:.8e},{:.8e},{:.8e},{}\n".format(
                    runtime, obj_val, primal_inf, dual_inf, simplex_iter
                )

                with open(os.path.join(self._log_dir, 'simplex_solution.csv'), 'a') as f:
                    f.write(log_entry)

            elif where == GRB.Callback.BARRIER:
                # Barrier callback
                itcnt = model.cbGet(GRB.Callback.BARRIER_ITRCNT)
                primobj = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
                dualobj = model.cbGet(GRB.Callback.BARRIER_DUALOBJ)
                priminf = model.cbGet(GRB.Callback.BARRIER_PRIMINF)
                dualinf = model.cbGet(GRB.Callback.BARRIER_DUALINF)
                cmpl = model.cbGet(GRB.Callback.BARRIER_COMPL)

                log_entry = "{},{},{},{},{},{}\n".format(
                    itcnt, primobj, dualobj, priminf, dualinf, cmpl
                )
                
                with open(os.path.join(self._log_dir, 'barrier_solution.csv'), 'a') as f:
                    f.write(log_entry)

            elif where == GRB.Callback.PRESOLVE:
                # Presolve phase callbacks
                removed_constrs = model.cbGet(GRB.Callback.PRE_ROWDEL)
                removed_vars = model.cbGet(GRB.Callback.PRE_COLDEL)

                # Log or handle the removed constraints and variables
                logger.info(f"[ExactOptimizer] Presolve removed {removed_constrs} constraints and {removed_vars} variables.")

        self._terminate_requested = False
        self._worker_done = threading.Event()
        
        def optimization_runner():
            try:
                self._model.optimize(_mycallback)
            except GurobiError as e:
                logger.error(f"[ExactOptimizer] Gurobi Error: {e}")
            except AttributeError as e:
                logger.error(f"[ExactOptimizer] Attribute Error: {e}")
            finally:
                self._worker_done.set()
                
        worker = threading.Thread(target=optimization_runner, daemon=True)
        worker.start()
        worker_finished = self._worker_done.wait(timeout=wall_limit)

        # If wall_limit is set, wait for that amount of time
        if wall_limit is not None:
            if not worker_finished:
                logger.warning(f"[ExactOptimizer] Wall time limit of {wall_limit}s reached. Attempting to terminate optimization.")
                self._terminate_requested = True
                try:
                    self._model.terminate()
                    # Give a short grace period for the callback to process termination
                    worker_finished = self._worker_done.wait(timeout=10.0)
                    # kill the worker thread if it hasn't finished
                    if worker.is_alive():
                        logger.warning("[ExactOptimizer] Worker thread did not finish after termination request. Forcefully terminating.")
                        worker.join(timeout=10.0)
                    if not worker_finished:
                        logger.warning("[ExactOptimizer] Optimization did not terminate gracefully after timeout.")
                except Exception as e:
                    logger.error(f"[ExactOptimizer] Error during termination: {e}")
        else:
            # Wait for the worker to finish without a timeout
            self._model.terminate()
            worker_finished = self._worker_done.wait(timeout=10.0)
            worker.join()

        # Check model status regardless of how it terminated
        if getattr(self._model, 'status', None) is None or self._model.status == GRB.INFEASIBLE:
            status = "INFEASIBLE"
            logger.warning(f"[ExactOptimizer] Optimization completed with status {status}. The model may be infeasible or not solved to optimality.")
            self.optimal = False
        else:
            status = self._model.status
            if status == GRB.OPTIMAL:
                logger.info(f"[ExactOptimizer] Optimization completed successfully with status {status}.")
                self.optimal = True
            else:
                logger.warning(f"[ExactOptimizer] Optimization completed with status {status}. The model may not be solved to optimality.")
                self.optimal = False
                
        # Even if we were terminated early, try to get the best solution found so far
        try: 
            self._f = self._model.objVal
        except Exception as e:
            logger.warning("[ExactOptimizer] Model does not have an objective value. Obtaining it from history.")
            if len(self._f_history) > 0:
                self._f = self._f_history[-1]
            else:
                logger.error("[ExactOptimizer] No objective value found in history. Setting to None.")
                self._f = None

        try:
            self._communities_dict = self.process_communities_dict(self._x)
        except Exception as e:
            logger.error(f"[ExactOptimizer] Error processing communities from decision variables: {e}")
            if len(self._communities_history) > 0:
                logger.info("[ExactOptimizer] Using best communities from history.")
                best_idx = np.argmax(self._f_history) if len(self._f_history) > 0 else -1
                if best_idx >= 0:
                    self._communities_dict = self._communities_history[best_idx]
                else:
                    self._communities_dict = None

        # Use this solution to update scores
        if self._communities_dict is None:
            logger.error("[ExactOptimizer] Communities dictionary is None. Cannot update scores.")
            raise ValueError("Communities dictionary is None. Ensure the model was built and solved correctly.")
        
        self._evaluator.update_scores(self._communities_dict)
        self._f1 = -np.mean(list(self._evaluator.hhi.values()))
        self._f2 = np.mean(list(self._evaluator.log_mpc.values()))

        return
    
    def tune(self, time_limit: Optional[int] = 3600) -> Dict[str,float|int]:
        
        self._model.setParam('TuneTimeLimit', time_limit)
        
        self._model.tune()

        if self._model.tuneResultCount > 0:
            # Apply the first (best) tuned parameter set
            self._model.getTuneResult(0)
    
        # Check how many tuning results are available
        tune_results = self._model.tuneResultCount
        logger.info(f"[ExactOptimizer] Number of tuning results: {tune_results}")

        tuned_params = {
            'Heuristics': self._model.getParamInfo('Heuristics')[2],
            'MIPFocus': self._model.getParamInfo('MIPFocus')[2],
            'VarBranch': self._model.getParamInfo('VarBranch')[2],
            'Cuts': self._model.getParamInfo('Cuts')[2],
            'Presolve': self._model.getParamInfo('Presolve')[2],
            'Method': self._model.getParamInfo('Method')[2],
        }
        logger.info(json.dumps(tuned_params, indent=4))

        return tuned_params

    def __str__(self):
        print(f"Model: {self._model_name}")
        print(f"Objective: {self._f}")
        print(f"Communities: {self._communities_dict}")
        if self.model_built:
            print("Total number of variables:", self._model.NumVars)
            print("Total number of integer variables:", self._model.NumIntVars)
            print("Total number of binary variables:", self._model.NumBinVars)
            print("Total number of continuous variables:", self._model.NumPWLObjVars)
            print("Total number of constraints:", self._model.NumConstrs)
            print("Total number of general constraints:", self._model.NumGenConstrs)
            print("Total number of quadratic non-zero coefficients:", self._model.NumQCNZs)
            print("Total number of non-zero coefficients:", self._model.NumNZs)

            # If the model has an objective function, print the number of terms
            if self._model.getObjective() is not None:
                objective = self._model.getObjective()
                # Check if the objective function is linear or quadratic
                if objective.size() > 0:
                    print("Number of terms in the objective function:", objective.size())
                else:
                    # This might be a linear objective with a simple sum of variables
                    print("Objective function appears to be a simple linear function.")

    def __del__(self):
        self._model.dispose()