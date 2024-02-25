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

# Check if the code is running in a Jupyter notebook
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from ..utilities import _default_if_none

from typing import Dict, List, Union, Tuple, Optional, NewType
from ..random import random_state

type_share = NewType('type_share', Dict[int,float])
type_shares = NewType('type_shares', Dict[Tuple[int, int], type_share])
type_scores = NewType('type_scores', Dict[Tuple[int, int], Dict[int, float]])

class AllianceBuilder():

    def __init__(
            self,
            MG,
            node_list: List[str] = None,
            communities_dict: Dict[str,List[str]] = None,
            seed: int = None
        ) -> None:

        self._communities_dict = {}
        self.G: MultiDiGraph = deepcopy(MG)
        self.node_list = []

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

        self._parse_graph()
        self._initialize_communities(node_list, communities_dict)
        self._initialize_HHI()
        self._initialize_MAI()

    @property
    def communities_dict(self):
        return {node: self._communities_dict[node] for node in self._communities_dict.keys()}

    @property
    def market_shares(self):
        return {edge: self._market_shares[edge].copy() for edge in self._market_shares.keys()}

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
        for u, v, key, attr in self.G.edges(keys=True, data=True):
            self.edges_by_type[attr['edge_type']].append((u, v, key))
            self.edges_by_source[u].append((u, v, key))
            self.edges_by_pair[(u, v, key)] = attr

    def _initialize_communities(self, node_list: List[str], communities_dict: Dict[str,str]):
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

            if node_list is None:
                edge_types = set()
                for _, _, data in self.G.edges(data=True):
                    edge_type = data.get('edge_type', None)
                    if edge_type:
                        edge_types.add(edge_type)

                node_list = sorted(edge_types)

            for i,node in enumerate(node_list):
                self._communities_dict[node] = i
                self.node_list.append(node)
        else:
            self._communities_dict = deepcopy(communities_dict)

    def _initialize_HHI(self):
        """
        Initializes the HHI attribute of each edge of the graph
        """
        for u, v, data in self.G.edges(data=True):
            # Accumulate the squares for each route
            self._market_shares[(u, v)][self._communities_dict[data['edge_type']]] += data['market_share_percent']

        # Calculate the HHI for each edge
        for edge, communities in self._market_shares.items():
            self._HHI[edge] = sum(market_share ** 2 for market_share in communities.values())

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

    def greedy_optimize(
            self, 
            weights: Optional[Tuple[float,float]]=(0.5,0.5), 
            max_iterations: Optional[int]=1000,
            n_samples: Optional[int] = None
        ):
        ########################################
        ## 4. Algorithm
        ########################################
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
            while improvement or n_iterations < max_iterations:
                improvement = False
                max_increase = 0
                best_pair = None

                # cache current partition
                distinct_communities = set(self._communities_dict.values())
                communities_dict = self._communities_dict.copy()
                market_shares, HHI = self._get_cache_HHI()
                weights_dict, airline_score = self._get_cache_MAI()
                
                if n_samples is None:
                    search_communities = distinct_communities
                else:
                    assert n_samples >= 2, "number of samples must be at least 2"
                    search_communities = rng.sample(distinct_communities,k=n_samples)

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
                                self._communities_dict = communities_dict.copy()
                                self._set_cache_HHI(market_shares, HHI)
                                self._set_cache_MAI(weights_dict, airline_score)

                            pbar_inner.update(1)

                # If the maximum increase is greater than 0, merge these two communities and update the mean HHI
                if max_increase > 0:
                    self.update_scores(*best_pair)
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
            to_nodes = {v for _, v, _ in edges}

            # Calculate weights
            if len(to_nodes) > 0:
                edges = set(sum((self.edges_by_source[node] for node in to_nodes), []))
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
