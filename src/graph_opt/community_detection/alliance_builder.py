from collections import defaultdict
import networkx as nx
from networkx import MultiDiGraph
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from copy import deepcopy

from ..utilities import _default_if_none

from typing import Dict, List, Union

class AllianceBuilder():

    def __init__(self,
        MG,
        node_list: List[str] = None,
        communities_dict: Dict[str,List[str]] = None) -> None:

        self.communities_dict = {}
        self.G: MultiDiGraph = deepcopy(MG)
        self.node_list = []

        self.HHI = defaultdict(float)
        self.market_shares = defaultdict(lambda: defaultdict(float))

        self.airline_score = {}
        self.weights_dict = defaultdict(list)
        self.weights_individual_dict = defaultdict(list)

        self.edges_by_type = defaultdict(list)
        self.edges_by_source = defaultdict(list)
        self.edges_by_pair = {}

        self._parse_graph()
        self._initialize_communities(node_list, communities_dict)
        self._initialize_HHI()
        self._initialize_MAI()

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
                self.communities_dict[node] = i
                self.node_list.append(node)
        else:
            self.communities_dict = deepcopy(communities_dict)

    def _initialize_HHI(self):
        """
        Initializes the HHI attribute of each edge of the graph
        """
        for u, v, data in self.G.edges(data=True):
            # Accumulate the squares for each route
            self.market_shares[(u, v)][self.communities_dict[data['edge_type']]] += data['market_share_percent']

        # Calculate the HHI for each edge
        for edge, communities in self.market_shares.items():
            self.HHI[edge] = sum(market_share ** 2 for market_share in communities.values())

    def _initialize_MAI(self):
        """
        Initializes the HHI attribute of each edge of the graph
        """
        for node in self.node_list:
            target_community = self.communities_dict[node]

            # Get edges by 'edge_type' from the pre-grouped dictionary
            edges_node = self.edges_by_type[node]
            G_sub = self.G.edge_subgraph(edges_node)

            # Get 'to' nodes
            to_nodes = {v for _, v, _ in edges_node}

            # Calculate weights
            if len(to_nodes) > 0:
                edges_neighbors = set(sum((self.edges_by_source[node] for node in to_nodes), []))
                weights = [self.edges_by_pair[(u,v,key)]['weight'] for u,v,key in edges_neighbors if self.communities_dict[self.edges_by_pair[(u,v,key)]['edge_type']] == target_community]
                weights_individual = [attr['weight'] for _, _, attr in G_sub.edges(data=True)]

                self.weights_dict[node] = sum(weights)
                self.weights_individual_dict[node] = sum(weights_individual)

                if np.sum(weights_individual) > 0:
                    self.airline_score[node] = np.log(self.weights_dict[node] + self.weights_individual_dict[node])

    def update_scores(self, community_to_merge, new_community):
        # Find the nodes in the community to merge
        nodes_to_update_HHI = [k for k, v in self.communities_dict.items() if v in [community_to_merge,]]
        nodes_to_update_MAI = [k for k, v in self.communities_dict.items() if v in [community_to_merge,new_community]]

        # Update the community in communities_dict ands reset the market shares for this route
        edges_to_update = set()
        for node in nodes_to_update_HHI:
            self.communities_dict[node] = new_community
            edges_to_update.update(set(((u,v) for u,v,key in self.edges_by_type[node])))

        self._update_HHI_scores(edges_to_update)
        self._update_scores(nodes_to_update_MAI)

    def _update_HHI_scores(self, edges_to_update):

        for u, v in edges_to_update:
            self.market_shares[(u, v)] = defaultdict(float)

        for u, v in edges_to_update:
            data = self.G.get_edge_data(u,v)
            for key,data_node in data.items():
                # Recalculate the market shares and HHI score for this route
                self.market_shares[(u, v)][self.communities_dict[data_node['edge_type']]] += data_node['market_share_percent']

        # Recalculate the community HHI for each affected route
        for edge, communities in self.market_shares.items():
            self.HHI[edge] = sum(market_share ** 2 for market_share in communities.values())

    def _update_scores(self, nodes_to_update):

        # Recalculate scores for the affected airlines
        for node in nodes_to_update:
            target_community = self.communities_dict[node]

            # Get edges by 'edge_type' from the pre-grouped dictionary
            edges = self.edges_by_type[node]

            # Get 'to' nodes
            to_nodes = {v for _, v, _ in edges}

            # Calculate weights
            if len(to_nodes) > 0:
                edges = set(sum((self.edges_by_source[node] for node in to_nodes), []))
                weights = [self.edges_by_pair[(u,v,key)]['weight'] for u,v,key in edges if self.communities_dict[self.edges_by_pair[(u,v,key)]['edge_type']] == target_community]

                self.weights_dict[node] = sum(weights)

                if self.weights_individual_dict[node] > 0:
                    self.airline_score[node] = np.log(self.weights_dict[node] + self.weights_individual_dict[node])