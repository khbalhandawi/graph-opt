from collections import defaultdict
from re import T
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

from .utilities import distinct_communities
from .utilities_cpu import edge_sampler_cpu, spatial_random_walk, attr_sampler_cpu
from .utilities_cpu import calc_sample_w_i_cpu, calc_sample_w_it_cpu, edge_sample_postprocess_cpu, calc_sample_w_ik_cpu, calc_sample_mpc_cpu
from .utilities_cpu import edge_postprocess_cpu, calc_hhi_cpu, calc_w_it_cpu, calc_w_ik_cpu, calc_mpc_cpu

from ..utilities import _default_if_none, dict2array, cpu_gpu_dispatcher, cuda_blocks, safe_log, serialize, deserialize
from ..random_walk.utilities import naive_weighted_choices

from typing import Dict, List, Union, Tuple, Optional, NewType
from ..random_settings import random_state

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


class GraphLoader:

    def __init__(
            self,
            MG: MultiDiGraph,
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
        self.edge_pairkey2data = {}
        self.edge_pair2pairkey_list = defaultdict(list)
        self.edge_pairkey2i = {}

        self._cond_probs_attr = defaultdict(lambda: defaultdict(float))
        self._adj_dense = nx.to_numpy_array(self.G, weight="weight")
        self._adj_dense_index = np.full((self.G.number_of_nodes(), self.G.number_of_nodes()), np.nan)
        self._cp = np.empty(0)
        self._adj_sparse = np.empty(0)
        self._adj3 = np.empty(0)
        self._adj_dense_index = np.empty(0)
        self._adj = np.empty(0)

        self.seed = seed
        self.random_state: Tuple[random.Random,np.random.RandomState] = random_state(self.seed)

        self.use_gpu = use_gpu
        self._init_gpu = False

        self._parse_graph()

        if self.use_gpu:
            self._post_init_gpu()

    @cpu_gpu_dispatcher('_post_init_cpu', '_post_init_gpu')
    def _post_init(self, **kwargs):
        pass

    def _post_init_cpu(self, **kwargs):
        pass

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

        self._cp = cp
        self._adj3 = adj3
        self._adj = adj
        self._adj_dense_index = adj_dense_index    
        self._node_spatial_degree = np.array(self.node_spatial_degree)

    def __del__(self):
        if self.use_gpu and self._init_gpu:
            # print("Destructor: Instance destroyed")
            del self._adj_d
            del self._adj3_d
            del self._adj_dense_d
            del self._adj_dense_index_d


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
            stream: Any
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
            stream: Optional[Any] = None
        ) -> None:
        super().__init__(gloader, seed, use_gpu, threads_per_block_1D, threads_per_block_2D, threads_per_block_3D)

        self.n_samples_edge = n_samples_edge
        self.n_walks = n_walks
        self.walk_length = walk_length

        self._V = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length+1), dtype=np.int32)
        self._walks_edges = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.int32)
        self._T = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.int32)
        self._W = np.zeros((self._gloader.n_nodes_spatial,self.n_walks,self.walk_length), dtype=np.float32)

        self._init_sampling()

    @property
    def N_uv(self):
        if self.use_gpu:
            return self._N_uv_d.copy_to_host()
        else:
            return self._N_uv

    @property
    def V(self):
        if self.use_gpu:
            return self._V_d.copy_to_host()
        else:
            return self._V
        
    @property
    def walks_edges(self):
        if self.use_gpu:
            return self._walks_edges_d.copy_to_host()
        else:
            return self._walks_edges

    @property
    def T(self):
        if self.use_gpu:
            return self._T_d.copy_to_host()
        else:
            return self._T
        
    @property
    def W(self):
        if self.use_gpu:
            return self._W_d.copy_to_host()
        else:
            return self._W
    
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
            self._walks_edges_d,
            cuda_random_state,
            cumsum_d,
            weights_d
        )

        self.cuda.synchronize()

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

        self.cuda.synchronize()

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


class BaseEvaluator(ABC):

    def __init__(
            self,
            gloader: GraphLoader,
            seed: int = None,
            use_gpu: Optional[bool] = False,
            n_communities: Optional[int] = None,
            threads_per_block_1D: int = 484,
            threads_per_block_2D: Union[int,Tuple[int,int]] = 22,
            threads_per_block_3D: Union[int,Tuple[int,int,int]] = 8,
            stream: Optional[Any] = None
        ):

        self._hhi = defaultdict(float)
        self._cond_probs_community = defaultdict(lambda: defaultdict(float))
        self._cond_probs_community_sparse = []

        self._log_mpc = {}
        self._mpc = defaultdict(list)
        self._w_k = np.empty(0)
        self._sampled_communities_pairs = set()

        self.seed = seed
        self.random_state: Tuple[random.Random,np.random.RandomState] = random_state(self.seed)

        self.use_gpu = use_gpu
        self._init_gpu = False

        self._gloader = gloader

        if n_communities is None:
            self.n_communities = self._gloader.n_attr
        else:
            self.n_communities = n_communities

        self._post_init(
            threads_per_block_1D=threads_per_block_1D,
            threads_per_block_2D=threads_per_block_2D,
            threads_per_block_3D=threads_per_block_3D,
            stream=stream
        )

    @cpu_gpu_dispatcher('_post_init_cpu', '_post_init_gpu')
    def _post_init(self, *args, **kwargs):
        pass

    @abstractmethod
    def _post_init_cpu(self, *args, **kwargs):
        self._w_it = np.zeros((self._gloader.n_nodes_spatial,self._gloader.n_attr),dtype=np.float32)
        self._w_i = np.zeros(self._gloader.n_nodes_spatial,dtype=np.float32)
        self._w_ik = np.zeros((self._gloader.n_nodes_spatial,self.n_communities),dtype=np.float32)
        self._w_k = np.zeros(self.n_communities)

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

        self._edges_communities_d = self.cuda.device_array((self._gloader.n_unique_edges,self.n_communities), dtype=np.float32, stream=self.stream)
        self._edges_hhi_d = self.cuda.device_array((self._gloader.n_unique_edges), dtype=np.float32, stream=self.stream)
        
        self._w_ik_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self.n_communities), dtype=np.float32, stream=self.stream)
        self._w_k_d = self.cuda.device_array((self.n_communities), dtype=np.float32, stream=self.stream)
        self._mpc_d = self.cuda.device_array((self._gloader.n_attr), dtype=np.float32, stream=self.stream)

    @property
    def membership(self)->np.ndarray:
        return np.array(list(self._communities_dict.values()))

    @property
    def communities(self)->np.ndarray:
        return np.array(self._distinct_communities)

    @property
    def communities_dict(self)->Dict[str,int]:
        return {attr: self._communities_dict[attr] for attr in self._communities_dict.keys()}

    @communities_dict.setter
    def communities_dict(self,value:Dict[str,int]):

        for i,attr in enumerate(self._gloader.attr_list):
            attr_value = value.get(attr,None)
            assert attr_value is not None, f"error: community not defined for {attr}"

        self._distinct_communities, self.n_distinct_communities = distinct_communities(value)

        assert self.n_distinct_communities <= self.n_communities, f"error: number of communities {self.n_distinct_communities} exceeds the number of attributes {self.n_communities}"

        self._communities_dict = value

        self._w_ik = np.zeros((self._gloader.n_nodes_spatial,self.n_communities),dtype=np.float32)
        self._w_k = np.zeros(self.n_communities)

        if self.use_gpu:
            self._membership_d = self.cuda.to_device(self.membership, stream=self.stream)
        
    @property
    def cond_probs_community(self) -> type_cps:
        return {edge: self._cond_probs_community[edge].copy() for edge in self._cond_probs_community.keys()}

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
    def w_it(self):
        if self.use_gpu:
            return self._w_it_d.copy_to_host()
        else:
            return self._w_it
        
    @property
    def w_ik(self):
        if self.use_gpu:
            return self._w_ik_d.copy_to_host()
        else:
            return self._w_ik
    
    @property
    def w_i(self):
        if self.use_gpu:
            return self._w_i_d.copy_to_host()
        else:
            return self._w_i

    @property
    def w_k(self):
        if self.use_gpu:
            return self._w_k_d.copy_to_host()
        else:
            return self._w_k.copy()

    @abstractmethod
    def update_scores(self, communities_dict: Optional[Dict[str,str]]):
        self.communities_dict = communities_dict
        pass

    @cpu_gpu_dispatcher('_update_community_weights_cpu', '_update_community_weights_gpu')
    def _update_community_weights(self):
        pass

    def _update_community_weights_cpu(self):
        self._w_k = np.sum(self._w_ik,axis=1)

    def _update_community_weights_gpu(self):
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self.n_communities), TPB = self.TPB_2D)
        self.calc_w_k_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_ik_d,
            self._w_k_d
        )

    @_jsonmode
    def save(self, directory=None, **kwargs):
        data_to_save = {
            'communities_dict': self._communities_dict,
            'log_mpc': self._log_mpc,
            'mpc': self._mpc,
            'hhi': {str(k): v for k, v in self._hhi.items()},
            'n_communities': self.n_communities,
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
            'n_communities':        {"name" : 'n_communities', "dtype" : int},
        }

        for name, data_dict in data_to_load.items():
            with open(f'{directory}/{name}.json', 'r') as f:
                data = deserialize(json.load(f))
                dtype = data_dict["dtype"]
                data = dtype_load_func.get(dtype, load_dict)(data)
                setattr(self, data_dict["name"], data)


class ExactEvaluator(BaseEvaluator):

    def __init__(
            self,
            gloader: GraphLoader,
            seed: int = None,
            use_gpu: Optional[bool] = False,
            n_communities: Optional[int] = None,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8,
            stream: Optional[Any] = None
        ) -> None:
        super().__init__(gloader, seed, use_gpu, n_communities, threads_per_block_1D, threads_per_block_2D, threads_per_block_3D, stream) 

    def _post_init_cpu(self, *args, **kwargs):
        super()._post_init_cpu(args, kwargs)

        calc_w_it_cpu(
            adj=self._gloader._adj_sparse,
            adj3=self._gloader._adj3,
            w_it=self._w_it
        )

        self._w_i = np.sum(self._w_it,axis=1)

    def _post_init_gpu(
            self,
            threads_per_block_1D: int,
            threads_per_block_2D: Union[int,Tuple[int,int]],
            threads_per_block_3D: Union[int,Tuple[int,int,int]],
            stream: Optional[Any] = None
        ):
        super()._post_init_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D,stream)

        self._post_init_cpu()

        from .utilities_cuda import define_cuda_kernels_deterministic

        self._w_it_d = self.cuda.to_device(self._w_it, stream=self.stream)
        self._w_i_d = self.cuda.to_device(self._w_i, stream=self.stream)

        # load kernels
        edge_postprocess_gpu, calc_hhi_gpu, calc_w_ik_gpu, calc_mpc_gpu, calc_w_k_gpu = define_cuda_kernels_deterministic()
        self.edge_postprocess_gpu           : Callable = edge_postprocess_gpu
        self.calc_hhi_gpu                   : Callable = calc_hhi_gpu
        self.calc_w_ik_gpu                  : Callable = calc_w_ik_gpu
        self.calc_mpc_gpu                   : Callable = calc_mpc_gpu
        self.calc_w_k_gpu                   : Callable = calc_w_k_gpu

    def update_scores(self, communities_dict: Optional[Dict[str,str]]):

        super().update_scores(communities_dict)

        self._update_hhi_scores()
        self._update_mpc_scores()
        self._update_community_weights()

        if self.use_gpu:
            self.stream.synchronize()

    @cpu_gpu_dispatcher('_update_hhi_scores_cpu', '_update_hhi_scores_gpu')
    def _update_hhi_scores(self, **kwargs):
        pass
    
    def _update_hhi_scores_cpu(self):

        edges_communities = np.zeros((self._gloader.n_unique_edges, self.n_communities),dtype=np.float32)

        edge_postprocess_cpu(
            edges_cp=self._gloader._cp,
            membership=self.membership,
            edges_communities=edges_communities
        )

        edges_hhi = np.zeros(self._gloader.n_unique_edges,dtype=np.float32)

        calc_hhi_cpu(
            edges_communities=edges_communities,
            edges_hhi=edges_hhi,
        )

        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = edges_hhi[i]

    def _update_hhi_scores_gpu(self):

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self._gloader.n_attr), TPB = self.TPB_2D)

        self.edge_postprocess_gpu[blockspergrid,threadsperblock,self.stream](
            self._gloader._cp_d,
            self._membership_d,
            self._edges_communities_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self.n_communities), TPB = self.TPB_2D)

        self.calc_hhi_gpu[blockspergrid,threadsperblock,self.stream](
            self._edges_communities_d,
            self._edges_hhi_d
        )

        hhi = self._edges_hhi_d.copy_to_host()

        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = hhi[i]

    @cpu_gpu_dispatcher('_update_mpc_scores_cpu', '_update_mpc_scores_gpu')
    def _update_mpc_scores(self):
        pass

    def _update_mpc_scores_cpu(self):

        self._w_ik = np.zeros((self._gloader.n_nodes_spatial,self.n_communities),dtype=np.float32)
        calc_w_ik_cpu(
            membership=self.membership,
            w_it=self._w_it,
            w_ik=self._w_ik
        )

        mpc = np.zeros(self._gloader.n_attr,dtype=np.float32)
        calc_mpc_cpu(
            membership=self.membership,
            w_it=self._w_it,
            w_ik=self._w_ik,
            w_i=self._w_i,
            mpc=mpc
        )

        log_mpc = safe_log(mpc)

        for attr in self._gloader.attr_list:
            self._mpc[attr] = mpc[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = log_mpc[self._gloader.attr_key2i[attr]]

    def _update_mpc_scores_gpu(self):
        
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)
        self.calc_w_ik_gpu[blockspergrid,threadsperblock,self.stream](
            self._membership_d,
            self._w_it_d,
            self._w_ik_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)
        self.calc_mpc_gpu[blockspergrid,threadsperblock,self.stream](
            self._membership_d,
            self._w_it_d,
            self._w_ik_d,
            self._w_i_d,
            self._mpc_d
        )

        mpc = self._mpc_d.copy_to_host()
        log_mpc = safe_log(mpc)

        for attr in self._gloader.attr_list:
            self._mpc[attr] = mpc[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = log_mpc[self._gloader.attr_key2i[attr]]


class SampleEvaluator(BaseEvaluator):
    
    def __init__(
            self,
            gloader: GraphLoader,
            sampler: NeighbourSampler,
            seed: int = None,
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
            seed = seed,
            use_gpu = use_gpu,
            n_communities = n_communities,
            threads_per_block_1D = threads_per_block_1D,
            threads_per_block_2D = threads_per_block_2D,
            threads_per_block_3D = threads_per_block_3D,
            stream = stream
        )

    def _post_init_cpu(self, **kwargs):
        super()._post_init_cpu(**kwargs)

        calc_sample_w_i_cpu(
            T_ijk = self._sampler._T,
            w_i = self._w_i
        )

        calc_sample_w_it_cpu(
            T_ijk = self._sampler._T,
            w_it = self._w_it
        )

        self._w_it = self._w_it
        self._w_i = self._w_i

    def _post_init_gpu(
            self,
            threads_per_block_1D: int,
            threads_per_block_2D: Union[int,Tuple[int,int]],
            threads_per_block_3D: Union[int,Tuple[int,int,int]],
            stream: Optional[Any] = None
        ):
        super()._post_init_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D,stream)

        from .utilities_cuda import define_cuda_kernels_stochastic, define_cuda_kernels_deterministic

        kernels = define_cuda_kernels_stochastic()
        
        self.edge_sample_postprocess_gpu    : Callable = kernels[0]
        self.calc_sample_hhi_gpu            : Callable = kernels[1]
        self.init_sample_w_i_gpu            : Callable = kernels[2]
        self.calc_sample_w_i_gpu            : Callable = kernels[3]
        self.calc_sample_w_it_gpu           : Callable = kernels[4]
        self.calc_sample_w_ik_gpu           : Callable = kernels[5]
        self.calc_sample_mpc_gpu            : Callable = kernels[6]

        kernels = define_cuda_kernels_deterministic()

        self.calc_w_k_gpu                   : Callable = kernels[4]

        self._w_i_d = self.cuda.device_array((self._gloader.n_nodes_spatial),dtype=np.int32, stream=self.stream)
        self._w_it_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self._gloader.n_attr),dtype=np.float32, stream=self.stream)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial,), TPB = self.TPB_1D)

        self.init_sample_w_i_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_i_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._sampler.n_walks), TPB = self.TPB_2D)

        self.calc_sample_w_i_gpu[blockspergrid,threadsperblock,self.stream](
            self._sampler._T_d,
            self._w_i_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._sampler.n_walks), TPB = self.TPB_2D)

        self.calc_sample_w_it_gpu[blockspergrid,threadsperblock,self.stream](
            self._sampler._T_d,
            self._w_it_d
        )

        self.stream.synchronize()

    @cpu_gpu_dispatcher('_update_hhi_scores_cpu', '_update_hhi_scores_gpu')
    def _update_hhi_scores(self, **kwargs):
        pass

    def _update_hhi_scores_cpu(self):
        
        reduced_counts = np.zeros((self._gloader.n_unique_edges,self.n_communities), dtype=np.float32)
        #################################
        # merge values

        edge_sample_postprocess_cpu(
            samples = self._sampler.N_uv,
            membership = self.membership,
            n_communities = self.n_communities,
            counts = reduced_counts
        )

        edge_counts = reduced_counts / self._sampler.n_samples_edge

        hhi_estimate = edge_counts**2
        hhi_estimate = np.nansum(hhi_estimate,axis=1)
        
        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = hhi_estimate[i]

    def _update_hhi_scores_gpu(self):
        
        reduced_counts_d = self.cuda.device_array((self._gloader.n_unique_edges,self.n_communities), dtype=np.float32, stream=self.stream)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self.n_communities), TPB = self.TPB_2D)

        self.edge_sample_postprocess_gpu[blockspergrid,threadsperblock,self.stream](
            self._sampler._N_uv_d,
            self._membership_d,
            reduced_counts_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_unique_edges, self.n_communities), TPB = self.TPB_2D)

        self.calc_sample_hhi_gpu[blockspergrid,threadsperblock,self.stream](
            reduced_counts_d,
            self._edges_hhi_d,
            self._sampler.n_samples_edge
        )

        hhi_estimate = self._edges_hhi_d.copy_to_host()

        # Update the hhi
        for i, edge in enumerate(self._gloader.edge_unique_list):
            self._hhi[edge] = hhi_estimate[i]

    @cpu_gpu_dispatcher('_update_mpc_scores_cpu', '_update_mpc_scores_gpu')
    def _update_mpc_scores(self):
        pass

    def _update_mpc_scores_cpu(self):

        self._w_ik = np.zeros((self._gloader.n_nodes_spatial,self.n_communities),dtype=np.float32)
        calc_sample_w_ik_cpu(
            T_ijk = self._sampler._T,
            membership = self.membership,
            w_ik = self._w_ik
        )

        mpc_array = np.zeros((self._gloader.n_nodes_spatial,self._gloader.n_attr),dtype=np.float32)
        calc_sample_mpc_cpu(
            w_it = self._w_it,
            w_ik = self._w_ik,
            w_i = self._w_i,
            membership = self.membership,
            mpc = mpc_array
        )

        mpc_estimate = np.nansum(mpc_array,axis=0) / mpc_array.shape[0]
        log_mpc = safe_log(mpc_estimate)

        # Update the mpc
        for attr in self._gloader.attr_list:
            self._mpc[attr] = mpc_estimate[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = log_mpc[self._gloader.attr_key2i[attr]]

    def _update_mpc_scores_gpu(self):
        
        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self.n_communities), TPB = self.TPB_2D)

        self.calc_sample_w_ik_gpu[blockspergrid,threadsperblock,self.stream](
             self._sampler._T_d,
            self._membership_d,
            self._w_ik_d
        )

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self._gloader.n_nodes_spatial, self._gloader.n_attr), TPB = self.TPB_2D)

        self.calc_sample_mpc_gpu[blockspergrid,threadsperblock,self.stream](
            self._w_it_d,
            self._w_ik_d,
            self._w_i_d,
            self._membership_d,
            self._mpc_d
        )

        mpc_estimate = self._mpc_d.copy_to_host()
        mpc_estimate = mpc_estimate / mpc_estimate.shape[0]
        log_mpc = safe_log(mpc_estimate)

        # Update the mpc
        for attr in self._gloader.attr_list:
            self._mpc[attr] = mpc_estimate[self._gloader.attr_key2i[attr]]
            self._log_mpc[attr] = log_mpc[self._gloader.attr_key2i[attr]]

    def update_scores(self, communities_dict: Optional[Dict[str,str]]):

        super().update_scores(communities_dict)

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


class BaseOptimizer(ABC):
    
    def __init__(self, evaluator: BaseEvaluator) -> None:
        self.evaluator = evaluator

        self.communities_history = []
        self.f_history = []
        self.multi_history = []

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
        mpc score is maximized

        Parameters
        ----------
        weights : Optional[Tuple[float,float]], optional
            The weighting used for the importance of
            hhi vs mpc, by default (0.5,0.5)
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

    @_jsonmode
    def load(self, directory=None, **kwargs):
        data_to_load = {
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

class GreedyOptimizer(BaseOptimizer):
    
    def _init_random_cpu(self, **kwargs):
        super()._init_random_cpu(**kwargs)
        self._distributions_changed = np.zeros(self._gloader.n_unique_edges, dtype=np.int32)
        self._cum_weight = np.zeros((self._gloader.n_nodes_spatial,self._gloader.n_attr), dtype=np.float32)

    def _init_random_gpu(
            self,
            threads_per_block_1D: Optional[int] = 484,
            threads_per_block_2D: Optional[Union[int,Tuple[int,int]]] = 22,
            threads_per_block_3D: Optional[Union[int,Tuple[int,int,int]]] = 8
        ):
        super()._init_random_gpu(threads_per_block_1D,threads_per_block_2D,threads_per_block_3D)

        self._distributions_changed_d = self.cuda.device_array(self._gloader.n_unique_edges, dtype=np.int32)
        self._cum_weight_d = self.cuda.device_array((self._gloader.n_nodes_spatial,self._gloader.n_attr), dtype=np.float32)
        self._reduced_counts_d = self.cuda.device_array((self._gloader.n_unique_edges,self.n_communities), dtype=np.float32)


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

        hhi_estimate, mpc_estimate, log_mpc = self.update_scores(0,0,reduced_counts_base,cum_weight_base)

        f1 = -np.mean(list(self._hhi.values()))
        f2 = np.mean(list(self._log_mpc.values()))
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
                    mpc_pairs = np.zeros((self.n_communities, self.n_communities, self.n_attr), dtype=np.float32)

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
                                    self._N_uv,
                                    self._W,
                                    self._T,
                                    membership_pairs,
                                    reduced_counts_base,
                                    weights,
                                    f
                                )
                                futures.append(future)

                            # Collect results and update max_increase and best_pair
                            for future in futures:
                                increase, community_i, community_j, hhi_estimate, mpc_estimate = future.result()
                                pbar_inner.update(1)
                                if community_i < community_j:
                                    if increase - max_increase > rtol:
                                        max_increase = increase
                                        best_pair = (community_i, community_j)
                                        improvement_sample = True

                                    if testing_opt:
                                        hhi_pairs[community_i,community_j,:] = hhi_estimate
                                        mpc_pairs[community_i,community_j,:] = mpc_estimate

                        if improvement_sample:
                            break

                # If the maximum increase is greater than 0, merge these two communities and update the mean hhi
                if max_increase > 0:

                    self.n_communities -= 1
                    reduced_counts_base = np.zeros((self.n_unique_edges,self.n_communities), dtype=np.int32)
                    cum_weight_base = np.zeros((self.n_nodes_spatial,self.n_attr), dtype=np.float32)
                    hhi_estimate, mpc_estimate, log_mpc = self.update_scores(*best_pair, reduced_counts_base, cum_weight_base)
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -np.mean(list(self._hhi.values()))
                    f2 = np.mean(list(self._log_mpc.values()))

                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        if testing_opt:
            return membership_pairs, hhi_pairs, mpc_pairs

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

        hhi_estimate, mpc_estimate, log_mpc = self.update_scores(0,0,reduced_counts_base_d,cum_weight_base_d)

        f1 = -np.mean(list(self._hhi.values()))
        f2 = np.mean(list(self._log_mpc.values()))
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
                    mpc_pairs_h = np.zeros((self.n_communities, self.n_communities, self.n_attr), dtype=np.float32)

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
                                    self._N_uv_d,
                                    self._W_d,
                                    self._T_d,
                                    membership_pairs_d,
                                    reduced_counts_base_d,
                                    weights,
                                    f,
                                    parallel_reduce
                                )
                                futures.append(future)

                            # Collect results and update max_increase and best_pair
                            for future in futures:
                                increase, community_i, community_j, hhi_estimate, mpc_estimate = future.result()
                                pbar_inner.update(1)
                                if community_i < community_j:
                                    if increase - max_increase > rtol:
                                        max_increase = increase
                                        best_pair = (community_i, community_j)
                                        improvement_sample = True

                                    if testing_opt:
                                        hhi_pairs_h[community_i,community_j,:] = hhi_estimate
                                        mpc_pairs_h[community_i,community_j,:] = mpc_estimate

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
                    hhi_estimate, mpc_estimate, log_mpc = self.update_scores(*best_pair, reduced_counts_base_d, cum_weight_base_d)
                    f += max_increase
                    improvement = True

                    # log history
                    f1 = -np.mean(list(self._hhi.values()))
                    f2 = np.mean(list(self._log_mpc.values()))

                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        if testing_opt:
            return membership_pairs_d.copy_to_host(), hhi_pairs_h, mpc_pairs_h

    def _update_community_weights(self):
        self.w_k = np.zeros(self.n_communities)
        for i, v in self.mpc.items():
            self.w_k[self._communities_dict[i]] += v

        self._sampled_communities_pairs = set()

        # populate excluded pairs
        for i in range(self.n_communities):
            for j in range(self.n_communities):
                if i >= j:
                    self._sampled_communities_pairs.add((i,j))

                weight_i = self.w_k[i]
                weight_j = self.w_k[j]

                # pop empty attributes
                if weight_i == 0.0 or weight_j == 0.0:
                    self._sampled_communities_pairs.add((i,j))

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
            N_uv,
            walks_nodes_weights,
            walks_nodes,
            membership_pairs,
            reduced_counts_base,
            weights,
            f,
        ):

        if community_i < community_j:
            stream_id = batch_index % self.num_streams

            # Perform stochastic evaluation using the selected stream
            hhi_estimate, mpc_estimate, log_mpc = self._stochastic_eval_cpu(
                community_i,
                community_j,
                N_uv,
                walks_nodes_weights,
                walks_nodes,
                membership_pairs,
                stream_id = stream_id
            )
            f1 = -np.mean(hhi_estimate)
            f2 = np.mean(log_mpc)
            f_new = multiobjective(f1, f2, weights=weights)

            reduced_counts = self.reduced_counts_streams[stream_id]
            # Assuming your matrices are named matrix1 and matrix2
            hhi_new = np.sort(reduced_counts, axis=1)
            hhi_base = np.sort(reduced_counts_base, axis=1)
            distributions_changed = np.any(hhi_new != hhi_base, axis=1)

            increase = f_new - f if np.any(distributions_changed) else 0.0

            return increase, community_i, community_j, hhi_estimate, mpc_estimate

        return 0, community_i, community_j, None, None

    def _calculate_members_gpu(self):

        # iteration constants
        communities_d = self.cuda.to_device(self._distinct_communities)
        membership_d = self.cuda.to_device(list((self._communities_dict.values())))

        # iteration values
        membership_pairs_d = self.cuda.device_array((self.n_communities, self.n_communities, self.n_attr), dtype=np.int16)

        nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_communities, self.n_communities, self.n_attr), TPB = self.TPB_3D)
        self.set_membership[blockspergrid,threadsperblock,self.stream](
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
            N_uv_d,
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
            hhi_estimate, mpc_estimate, log_mpc = self._stochastic_eval_gpu(
                community_i,
                community_j,
                N_uv_d,
                walks_nodes_weights_d,
                walks_nodes_d,
                membership_pairs_d,
                stream=stream,
                stream_id=stream_id,
                parallel_reduce = parallel_reduce
            )
            f1 = -np.mean(hhi_estimate)
            f2 = np.mean(log_mpc)
            f_new = multiobjective(f1, f2, weights=weights)

            nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(self.n_unique_edges, ), TPB = self.TPB_1D)

            distributions_changed_d = self.distributions_changed_streams[stream_id]
            reduced_counts_d = self.reduced_counts_streams[stream_id]

            self.check_edge_distribution_gpu[threadsperblock,blockspergrid,stream](
                reduced_counts_base_d,
                reduced_counts_d,
                distributions_changed_d
            )

            stream.synchronize()

            distributions_changed = distributions_changed_d.copy_to_host()
            distributions_changed = distributions_changed > 0

            increase = f_new - f if np.any(distributions_changed) else 0.0

            return increase, community_i, community_j, hhi_estimate, mpc_estimate

        return 0, community_i, community_j, None, None

    def sample_community_pairs(self, n_samples: int = 5, epsilon: float = 0.1, replace: Optional[bool] = False) -> Union[list,bool]:

        rng_np = self.random_state[1]
        all_samples = list(itertools.product(range(self.n_communities), range(self.n_communities)))

        sample_set = set(all_samples) - self._sampled_communities_pairs
        sample_list = list(sample_set)  # convert set to list

        if len(sample_list) == 0:
            return False

        probabilities = self._w_k[sample_list] / np.sum(self._w_k)
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

