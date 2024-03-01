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
from typing import Callable, Dict, Any, Union

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from .utilities import resort_communities
from ..utilities import _default_if_none, dict2array, cpu_gpu_dispatcher

from typing import Dict, List, Union, Tuple, Optional, NewType
from ..random_settings import random_state

type_share = NewType('type_share', Dict[int,float])
type_shares = NewType('type_shares', Dict[Tuple[int, int], type_share])
type_scores = NewType('type_scores', Dict[Tuple[int, int], Dict[int, float]])


class AllianceBuilder():

    def __init__(
            self,
            MG,
            communities_dict: Dict[str,List[str]] = None,
            seed: int = None,
            use_gpu: Optional[bool] = False
        ) -> None:

        self._communities_dict = {}
        self.G: MultiDiGraph = deepcopy(MG)
        self.node_list = []
        self.node_spatial_list = []
        self.node_dict = {}
        self.node_spatial_dict = {}

        self._HHI = defaultdict(float)
        self._market_shares = defaultdict(lambda: defaultdict(float))

        self._airline_score = {}
        self._weights_dict = defaultdict(list)
        self._weights_individual_dict = defaultdict(list)
        
        self.edges_by_type = defaultdict(list)
        self.edges_by_source = defaultdict(list)
        self.edges_by_pair = {}

        self.communities_history = []
        self.f_history = []
        self.multi_history = []

        self.seed = seed
        self.random_state: Tuple[random.Random,np.random.RandomState] = None

        self.use_gpu = use_gpu

        self._parse_graph()
        self._initialize_communities(communities_dict)
        self._initialize_HHI()
        self._initialize_MAI()

        # Populate the adjacency matrix
        edge_nodes = []
        edge_weights = []
        adj = np.zeros((self.G.number_of_edges(), 2), dtype=np.int32)
        for i, (u, v, key, data) in enumerate(self.G.edges(data=True, keys=True)):
            adj[i,:] = np.array([self.node_spatial_dict[u],self.node_spatial_dict[v]])
            edge_nodes.append(self.node_dict[data["edge_type"]])
            edge_weights.append(data["weight"])

        self._adj = adj
        self._edge_nodes = np.array(edge_nodes,dtype=np.int32)
        self._edge_weights = np.array(edge_weights,dtype=np.float32)

        if self.use_gpu:
            from .utilities_cuda import cuda, merge_score_kernel
            self.cuda = cuda
            self.merge_score_kernel: Callable = merge_score_kernel

            # move constants to memory
            self._adj_d = self.cuda.to_device(self._adj)
            self._edge_nodes_d = self.cuda.to_device(self._edge_nodes)
            self._edge_weights_d = self.cuda.to_device(self._edge_weights)
            self._node_weight_individual_d = self.cuda.to_device(np.array(list(self._weights_individual_dict.values()),dtype=np.float32))

        else:
            self.cuda = None
            self.merge_score_kernel: Callable = None

            self._adj_d = None
            self._edge_nodes_d = None
            self._edge_weights_d = None

    @property
    def communities_dict(self)->Dict[str,int]:
        return {node: self._communities_dict[node] for node in self._communities_dict.keys()}

    @communities_dict.setter
    def communities_dict(self,value:Dict[str,int]):
        key, self._communities_dict, self._distinct_communities, self.n_communities = resort_communities(value)

    @property
    def market_shares(self):
        return {edge: self._market_shares[edge].copy() for edge in self._market_shares.keys()}

    @property
    def market_shares_dense(self):
        
        market_share = {}
        for u, v in self._market_shares.keys():
            market_share[(u, v)] = {}
            for node, value in self._communities_dict.items():
                market_share[(u, v)][value] = self._market_shares[(u, v)].get(value,0.0)

        return market_share

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

        for u, v, key, attr in self.G.edges(keys=True, data=True):
            self.edges_by_type[attr['edge_type']].append((u, v, key))
            self.edges_by_source[u].append((u, v, key))
            self.edges_by_pair[(u, v, key)] = attr
        
        for i, u in enumerate(self.G.nodes()):
            self.node_spatial_dict[u] = i
            self.node_spatial_list.append(u)

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
                self.node_dict[node] = i

            self.communities_dict = communities_dict
        else:
            self.communities_dict = communities_dict
            node_list = sorted(list(self._communities_dict.keys()))
            for i,node in enumerate(node_list):
                self.node_list.append(node)
                self.node_dict[node] = i   

        self.n_nodes = len(self.node_list)
        
    def _initialize_HHI(self):
        """
        Initializes the HHI attribute of each edge of the graph
        """
        for u, v, data in self.G.edges(data=True):
            # Accumulate the squares for each route
            self._market_shares[(u, v)][self._communities_dict[data['edge_type']]] += data['market_share_percent']

        self.market_shares_nodal = self.market_shares_dense
        
        # Calculate the HHI for each edge
        for edge, communities in self._market_shares.items():
            self._HHI[edge] = sum(market_share ** 2 for market_share in communities.values())

        self.unique_edges = list(self._market_shares.keys())
        self.n_edges = len(self.unique_edges)

    def _initialize_MAI(self):
        """
        Initializes the HHI attribute of each edge of the graph
        """
        for node in self.node_list:
            target_community = self._communities_dict[node]

            # Get edges by 'edge_type' from the pre-grouped dictionary
            edges_node = self.edges_by_type[node]
            G_sub = self.G.edge_subgraph(edges_node)

            # Get 'to' nodes
            to_nodes = {v for _, v, _ in edges_node}

            # Calculate weights
            if len(to_nodes) > 0:
                edges_neighbors = set(sum((self.edges_by_source[node] for node in to_nodes), []))
                weights = [self.edges_by_pair[(u,v,key)]['weight'] for u,v,key in edges_neighbors if self._communities_dict[self.edges_by_pair[(u,v,key)]['edge_type']] == target_community]
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
            n_samples: Optional[int] = None,
            debug: bool = False
        ) -> Tuple[Dict[int,int], float]:
        

        def multiobjective(f1: Union[float,np.ndarray], f2: Union[float,np.ndarray], weights: Optional[Tuple[float,float]]=(0.5,0.5)):
            return weights[0]*f1 + weights[1]*f2

        # Calculate the initial mean HHI over all routes
        f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
        f2 = sum(self._airline_score.values()) / len(self._airline_score)
        f = multiobjective(f1,f2,weights)

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
                market_shares_nodal_d = self.cuda.to_device(dict2array(self.market_shares_nodal))

                edge_HHI_d = self.cuda.to_device(list(self._HHI.values()))
                node_weight_d = self.cuda.to_device(list(self._weights_dict.values()))
                node_score_d = self.cuda.to_device(list(self._airline_score.values()))

                # initialize memory
                # market_shares_d = cuda.device_array((len(self._HHI),len(distinct_communities)), dtype=np.float32)
                # n_communities = 500
                # n_nodes = 500
                # n_edges = 20000
                # Note: no need to reduce n_communities by 1 during iteration
                membership_pairs_d = self.cuda.device_array((self.n_communities,self.n_communities,self.n_nodes), dtype=np.int32)
                market_shares_pairs_d = self.cuda.device_array((self.n_communities,self.n_communities,self.n_edges,self.n_communities), dtype=np.float32)
                hhi_pairs_d = self.cuda.device_array((self.n_communities,self.n_communities,self.n_edges), dtype=np.float32)
                weights_pairs_d = self.cuda.device_array((self.n_communities,self.n_communities,self.n_nodes), dtype=np.float32)
                score_pairs_d = self.cuda.device_array((self.n_communities,self.n_communities,self.n_nodes), dtype=np.float32)
                
                market_shares_pairs_d.fill(0)
                hhi_pairs_d.fill(0)
                weights_pairs_d.fill(0)
                score_pairs_d.fill(0)

                # Define the thread and block dimensions
                TPB = 4
                threadsperblock = (TPB, TPB)
                blockspergrid_x = math.ceil(self.n_communities / threadsperblock[0])
                blockspergrid_y = math.ceil(self.n_communities / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)

                self.merge_score_kernel[blockspergrid,threadsperblock](
                    self.n_communities,
                    self.n_edges,
                    self.n_nodes,
                    self.m_edges,
                    communities_d,
                    membership_d,
                    market_shares_nodal_d,
                    self._adj_d,
                    self._edge_nodes_d,
                    self._edge_weights_d,
                    edge_HHI_d,
                    node_weight_d,
                    self._node_weight_individual_d,
                    node_score_d,
                    membership_pairs_d,
                    market_shares_pairs_d,
                    hhi_pairs_d,
                    weights_pairs_d,
                    score_pairs_d
                )
                self.cuda.synchronize()

                market_shares_pairs_h = market_shares_pairs_d.copy_to_host()
                hhi_pairs_h = hhi_pairs_d.copy_to_host()
                weights_pairs_h = weights_pairs_d.copy_to_host()
                score_pairs_h = score_pairs_d.copy_to_host()
                membership_pairs_h = membership_pairs_d.copy_to_host()

                f1_pairs = -np.mean(hhi_pairs_h,axis=2) / 100
                f2_pairs = np.mean(score_pairs_h,axis=2)

                f_pairs = multiobjective(f1_pairs,f2_pairs,weights=weights)

                # Find the index of the maximum element
                max_index_flat = np.argmax(f_pairs)
                best_pair = np.unravel_index(max_index_flat, f_pairs.shape)

                # Calculate the increase in objective
                f_new = f_pairs[best_pair[0], best_pair[1]]
                f1 = f1_pairs[best_pair[0], best_pair[1]]
                f2 = f2_pairs[best_pair[0], best_pair[1]]
                max_increase = f_new - f

                # free memory
                communities_d = None
                membership_d = None
                market_shares_nodal_d = None
                edge_HHI_d = None
                node_weight_d = None
                node_score_d = None
                membership_pairs_d = None
                market_shares_pairs_d = None
                hhi_pairs_d = None
                weights_pairs_d = None
                score_pairs_d = None

                # If the maximum increase is greater than 0, merge these two communities and update the mean HHI
                if max_increase > 0:
                    communities_dict = dict(zip(self.node_list,membership_pairs_h[best_pair[0], best_pair[1]]))
                    HHI = dict(zip(self.unique_edges,hhi_pairs_h[best_pair[0], best_pair[1]]))
                    market_shares = market_shares_pairs_h[best_pair[0], best_pair[1]]
                    airline_score = dict(zip(self.node_list,score_pairs_h[best_pair[0], best_pair[1]]))
                    
                    to_original_key, self._communities_dict, self._distinct_communities, self.n_communities = resort_communities(communities_dict)

                    # update current structure
                    for u, v in self.unique_edges:
                        self._market_shares[(u, v)] = defaultdict(float)

                    for i, (u, v) in enumerate(self.unique_edges):
                        for k in self._distinct_communities:
                            if market_shares[i,to_original_key[k]] > 0.0:
                                self._market_shares[(u, v)][k] = market_shares[i,to_original_key[k]]

                    self._HHI = HHI
                    self._airline_score = airline_score

                    f += max_increase
                    improvement = True

                    # log history
                    self.communities_history.append(self.communities_dict)
                    self.f_history.append(f)
                    self.multi_history.append((f1,f2))

                n_iterations += 1
                pbar_outer.update(1)

        if debug:
            return membership_pairs_h, hhi_pairs_h, market_shares_pairs_h, weights_pairs_h, score_pairs_h

    def __del__(self):
        print("Destructor: Instance destroyed")
        self._adj_d = None
        self._edge_nodes_d = None
        self._edge_weights_d = None
        self._node_weight_individual_d = None

    def _greedy_optimize_cpu(
            self, 
            weights: Optional[Tuple[float,float]]=(0.5,0.5), 
            max_iterations: Optional[int]=1000,
            n_samples: Optional[int] = None
        ) -> Tuple[Dict[int,int], float]:

        self.random_state = _default_if_none(self.random_state, random_state(self.seed), "rng_state")

        def multiobjective(f1, f2, weights: Optional[Tuple[float,float]]=(0.5,0.5)):
            return weights[0]*f1 + weights[1]*f2

        rng = self.random_state[0]
        rng_np = self.random_state[1]

        # Calculate the initial mean HHI over all routes
        f1 = -(sum(self._HHI.values())*100) / (len(self._HHI)*10000)
        f2 = sum(self._airline_score.values()) / len(self._airline_score)
        f = multiobjective(f1,f2,weights)

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
                market_shares, HHI = self._get_cache_HHI()
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
                                self._set_cache_HHI(market_shares, HHI)
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

    def _update_communities(self, community_to_merge, new_community, update_communities=True):
        # Find the nodes in the community to merge
        nodes_to_update_HHI = [k for k, v in self._communities_dict.items() if v in [community_to_merge,]]
        nodes_to_update_MAI = [k for k, v in self._communities_dict.items() if v in [community_to_merge,new_community]]

        # Update the community in communities_dict ands reset the market shares for this route
        edges_to_update = set()
        for node in nodes_to_update_HHI:
            if update_communities:
                self._communities_dict[node] = new_community

            edges_to_update.update(set(((u,v) for u,v,key in self.edges_by_type[node])))

        return edges_to_update, nodes_to_update_MAI
    
    def _update_HHI_scores(self, edges_to_update):

        for u, v in edges_to_update:
            self._market_shares[(u, v)] = defaultdict(float)

        for u, v in edges_to_update:
            data = self.G.get_edge_data(u,v)
            for key,data_node in data.items():
                # Recalculate the market shares and HHI score for this route
                self._market_shares[(u, v)][self._communities_dict[data_node['edge_type']]] += data_node['market_share_percent']

        # Recalculate the community HHI for each affected route
        for edge, communities in self._market_shares.items():
            self._HHI[edge] = sum(market_share ** 2 for market_share in communities.values())

    def _update_scores(self, nodes_to_update):

        # Recalculate scores for the affected airlines
        for node in nodes_to_update:
            target_community = self._communities_dict[node]

            # Get edges by 'edge_type' from the pre-grouped dictionary
            edges = self.edges_by_type[node]

            # Get 'to' nodes
            to_nodes = [v for _, v, _ in edges]

            # Calculate weights
            if len(to_nodes) > 0:
                edges = sum((self.edges_by_source[node] for node in to_nodes), [])
                weights = [self.edges_by_pair[(u,v,key)]['weight'] for u,v,key in edges if self._communities_dict[self.edges_by_pair[(u,v,key)]['edge_type']] == target_community]

                self._weights_dict[node] = sum(weights)

                if self._weights_individual_dict[node] > 0:
                    self._airline_score[node] = np.log(self._weights_dict[node] + self._weights_individual_dict[node])

    def _get_cache_HHI(self, edges:Optional[List[Tuple[int,int]]]=None) -> Tuple[type_shares, type_scores]:

        if edges is None:
            edges = self._market_shares.keys()

        market_shares = {}
        HHI = {}
        for edge in edges:
            market_shares[edge] = self._market_shares[edge].copy()
            HHI[edge] = self._HHI[edge]
        
        return market_shares, HHI
    
    def _get_cache_MAI(self, nodes: List[int]=None) -> Tuple[type_scores, type_scores]:

        if nodes is None:
            nodes = self._weights_dict.keys()

        weights_dict = {}
        airline_score = {}
        for node in nodes:
            weights_dict[node] = self._weights_dict[node]
            airline_score[node] = self._airline_score[node]
        
        return weights_dict, airline_score
    
    def _set_cache_HHI(self, market_shares: type_shares, HHI: type_scores):
        for edge in market_shares.keys():
            self._market_shares[edge] = market_shares[edge].copy()
            self._HHI[edge] = HHI[edge]

    def _set_cache_MAI(self, weights_dict: type_scores, airline_score: type_scores):
        for node in weights_dict.keys():
            self._weights_dict[node] = weights_dict[node]
            self._airline_score[node] = airline_score[node]

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
    data_dir = os.path.join('..','test_data')
    MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph
    with open(os.path.join(script_dir,data_dir,'test_airline_nodes.json'),'r') as file:
        nodes_airlines_dict = json.load(file)

    param = {
        "weights": (0.1,0.9),
        "test_communities_dict": {
            'AT': 0,
            'BT': 1,
            'CT': 2,
            'HT': 3,
            'VT': 4
        },
        "n_samples": None
    }

    weights = param["weights"]
    n_samples = param["n_samples"]
    test_communities_dict = param["test_communities_dict"]

    test_class = AllianceBuilder(MG,seed=1234, use_gpu=True)

    test_class.greedy_optimize(weights=weights,n_samples=n_samples)