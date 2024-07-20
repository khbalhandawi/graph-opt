from cgi import test
from collections import defaultdict
from doctest import debug
from ipaddress import v4_int_to_packed
from optparse import Option
import pytest
import os

from sklearn.covariance import log_likelihood
from ..community_detection.alliance_builder import DeterministicCommunity, StochasticCommunity, BaseCommunity, multiobjective
from ..community_detection.utilities import resort_communities, same_community
from ..community_detection.utilities_cuda import define_cuda_kernels_stochastic
from ..community_detection.utilities_cpu import edge_postprocess_cpu, attr_postprocess_cpu, community_weight_cpu
from ..utilities import cuda_blocks

import numpy as np
import networkx as nx
import json
import os
import math
import itertools
from scipy.stats import chisquare
from scipy.special import kl_div
from numba import cuda

from typing import Optional, Tuple, Dict, List

seed = 1234
tolerance = 1e-6

def get_conditional_prob(test_class: StochasticCommunity):

    cond_prob_attr_edge = np.zeros((test_class.n_unique_edges,test_class.n_attr))
    for i, (u,v) in enumerate(test_class.edge_unique_list):

        for j,attr in enumerate(test_class.attr_list):
            cond_prob_attr_edge[i,j] = test_class._cond_probs_attr[(u,v)].get(attr,0.0)/100

    return cond_prob_attr_edge

def get_likelihood_prob(test_class: StochasticCommunity):

    likelihood_edge_attr = np.zeros((test_class.n_attr,test_class.n_unique_edges))
    for i,attr in enumerate(test_class.attr_list):
        attr_weight = test_class.attr_key2w[attr]

        for u,v,key in test_class.edge_attr2keypair_list[attr]:
            j = test_class.edge_pairkey2ui[(u,v,key)]
            edge_weight = test_class.edge_pairkey2data[(u,v,key)]["weight"]
            likelihood_edge_attr[i,j] += edge_weight / attr_weight

    return likelihood_edge_attr

def get_prior(test_class: StochasticCommunity):

    prior_attr = np.zeros(test_class.n_attr)
    attr_weights = list(test_class.attr_key2w.values())
    for i,attr in enumerate(test_class.attr_list):
        attr_weight = test_class.attr_key2w[attr]
        prior_attr[i] = attr_weight / sum(attr_weights)

    return prior_attr

def get_marginal(test_class: StochasticCommunity):

    marginal_edge = np.zeros(test_class.n_unique_edges)
    edge_weights = list(test_class.edge_pair2w.values())
    for i,edge in enumerate(test_class.edge_unique_list):
        edge_weight = test_class.edge_pair2w[edge]
        marginal_edge[i] = edge_weight / sum(edge_weights)

    return marginal_edge

def get_source_prob(test_class:StochasticCommunity):

    source_p = np.array(list(test_class.node_spatial_key2w.values()),dtype=np.float64)
    total = source_p.sum()
    source_p /= total

    return source_p

def get_edge_prob(test_class: StochasticCommunity):

    # get sampled edge probabilities (marginal)
    cond_edge_p = np.zeros((test_class.n_nodes_spatial,test_class.n_unique_edges))
    edge_p = np.zeros(test_class.n_unique_edges)

    total_weight = sum([test_class.edge_pair2w[(u,v)] for u,v in test_class.edge_unique_list])

    for i, source in enumerate(test_class.node_spatial_list):

        source_edges = test_class.edge_u2pair_list[source]
        source_edge_weights = [test_class.edge_pair2w[(u,v)] for u,v in source_edges]
        total_source_edge_weights = sum(source_edge_weights)

        for j, (u,v) in enumerate(test_class.edge_unique_list):
            if (u,v) in source_edges:
                edge_weight = test_class.edge_pair2w[(u,v)]
                cond_edge_p[i,j] = edge_weight/total_source_edge_weights

    for j, (u,v) in enumerate(test_class.edge_unique_list):
        edge_weight = test_class.edge_pair2w[(u,v)]
        edge_p[j] = edge_weight/total_weight

    return edge_p, cond_edge_p

def get_attr_bfs_probabilities(
        test_class: StochasticCommunity,
        source_u: str,
        target_attr: str,
        max_depth:Optional[int] = 3,
        current_depth:Optional[int]=1,
        community_weight:Optional[Dict[str,float]]=None,
        attr_weight:Optional[Dict[str,float]]=None,
        all_weight:Optional[Dict[str,float]]=None,
        previous_edge:Optional[Tuple[str,str, int]]=None
    ):

    if community_weight is None or all_weight is None or attr_weight is None:
        community_weight = [0,] * max_depth
        attr_weight = [0,] * max_depth
        all_weight = [0,] * max_depth

    if current_depth > max_depth:
        return community_weight, attr_weight, all_weight

    edges = test_class.edge_u2pairkey_list[source_u]
    total_weight = sum(test_class.edge_pairkey2data[edge]['weight'] for edge in edges)
    target_community = test_class.communities_dict[target_attr]

    for u, v, key in edges:
        curr_edge = (u, v, key)
        edge_prob = test_class.edge_pair2w[(u,v)] / total_weight
        edge_type_prob = test_class.edge_pairkey2data[curr_edge]['market_share_percent'] / 100.0
        edge_type_weight = test_class.edge_pairkey2data[curr_edge]['weight']
        edge_type = test_class.edge_pairkey2data[curr_edge]['edge_type']
        current_community = test_class.communities_dict[edge_type]

        previous_edge_type = test_class.edge_pairkey2data[previous_edge]['edge_type'] if previous_edge is not None else None
        previous_community = test_class.communities_dict.get(previous_edge_type,None)

        if current_community == target_community:
            community_weight[current_depth - 1] += edge_type_weight
        if edge_type == target_attr:
            attr_weight[current_depth - 1] += edge_type_weight

        all_weight[current_depth - 1] += edge_type_weight

        get_attr_bfs_probabilities(
            test_class,
            v,
            target_attr,
            max_depth,
            current_depth + 1,
            community_weight,
            attr_weight,
            all_weight,
            curr_edge
        )

    return community_weight, attr_weight, all_weight

def get_bfs_probabilities(
        test_class: StochasticCommunity,
        max_depth:Optional[int] = 3,
    ):

    attr_bfs_p = np.zeros((test_class.n_nodes_spatial,test_class.n_attr,max_depth))

    for i, source_node in enumerate(test_class.node_spatial_list):
        for j, target_attr in enumerate(test_class.attr_list):

            w_community,w_attr,w_total = get_attr_bfs_probabilities(
                test_class,
                source_node,
                target_attr,
                max_depth
            )

            wk_i = sum(w_community)
            wt_i = sum(w_attr)
            w_i = sum(w_total)

            attr_bfs_p[i,j,:] = (wk_i*wt_i) / (w_i*w_i) if wt_i > 0 else 0.0

    return attr_bfs_p

def get_estimate_edge_prob(test_class: StochasticCommunity, walks_edges_spatial: np.ndarray):

    # get sampled edge probabilities (marginal)
    cond_edge_p = np.zeros((test_class.n_nodes_spatial,test_class.n_unique_edges))
    edge_p = np.zeros(test_class.n_unique_edges)

    counts_cond = np.zeros((test_class.n_nodes_spatial,test_class.n_unique_edges))
    counts = np.zeros(test_class.n_unique_edges)

    for i, source in enumerate(test_class.node_spatial_list):

        for j, (u,v) in enumerate(test_class.edge_unique_list):
            counts_cond[i,j] = np.sum(walks_edges_spatial[test_class.node_spatial_key2i[source], :, 0] == test_class.edge_pair2ui[(u,v)])

        if counts_cond[i,:].sum() != 0:
            cond_edge_p[i,:] = counts_cond[i,:]/counts_cond[i,:].sum()

    edge_p = cond_edge_p.sum(axis=0) / test_class.n_nodes_spatial

    return edge_p, cond_edge_p

def get_estimate_edge_weighted_prob(
        test_class: StochasticCommunity,
        n_samples_edge: int,
        n_samples_source_node: int,
        walks_nodes_spatial: np.ndarray,
        walks_edges_spatial: np.ndarray,
        testing: Optional[bool] = False
    ):

    # get sampled edge probabilities (marginal)
    cond_edge_p = np.zeros((test_class.n_nodes_spatial,test_class.n_unique_edges))

    counts_cond = np.zeros((test_class.n_nodes_spatial,test_class.n_unique_edges))
    counts_source = np.zeros(test_class.n_nodes_spatial)

    for i_sample in range(n_samples_source_node):
        for j_sample in range(n_samples_edge):
            sample_walk_edges = walks_edges_spatial[i_sample, j_sample, :]
            sample_walk_spatial_nodes = walks_nodes_spatial[i_sample, j_sample, :]

            i = sample_walk_spatial_nodes[0]
            j = sample_walk_edges[0]

            counts_source[i] += 1
            if j in test_class.edge_ui2pair.keys():
                counts_cond[i,j] += 1


    p_source = counts_source / counts_source.sum()

    for i, source in enumerate(test_class.node_spatial_list):
        if counts_cond[i,:].sum() != 0:
            cond_edge_p[i,:] = counts_cond[i,:]/counts_cond[i,:].sum()

    edge_p = p_source @ cond_edge_p

    if testing:
        return p_source, edge_p, cond_edge_p
    else:
        return edge_p, cond_edge_p

def get_estimate_node_bfs_probabilities(
        test_class: StochasticCommunity,
        n_samples_edge: int,
        walk_length: int,
        walks_attr: np.ndarray,
        walks_nodes_spatial: np.ndarray,
    ):

    # get sampled edge probabilities (marginal)
    bfs = np.zeros((test_class.n_nodes_spatial,test_class.n_attr,walk_length-1))
    bfs_compound = np.zeros((test_class.n_nodes_spatial,test_class.n_attr,walk_length-1))
    counts_bfs = np.zeros((test_class.n_nodes_spatial,test_class.n_attr,walk_length-1))
    counts_spatial = np.zeros((test_class.n_nodes_spatial,walk_length-1))

    for i in range(test_class.n_nodes_spatial):
        for j in range(n_samples_edge):
            sample_walk_spatial_nodes = walks_nodes_spatial[i, j, :]
            sample_walk_attr = walks_attr[i, j, :]

            for t in test_class.attr_i2key.keys():
                t_target_community = test_class.communities_dict[test_class.attr_i2key[t]]
                for l in range(walk_length-1):
                    t_sample = sample_walk_attr[l]
                    if t_sample in test_class.attr_i2key.keys():
                        t_sample_community = test_class.communities_dict[test_class.attr_i2key[t_sample]]
                        counts_spatial[i,l] += 1
                        if t_sample_community==t_target_community:
                            counts_bfs[i,t,l] += 1
                        else:
                            break

    for i in range(test_class.n_nodes_spatial):
        for l in range(walk_length-1):

            bfs[i,:,l] = counts_bfs[i,:,l] / counts_spatial[i,l].sum() if counts_spatial[i,l].sum() > 0 else 0.0

    bfs_compound[:,:,0] = bfs[:,:,0]
    for l in range(1,walk_length-1):
        bfs_compound[:,:,l] = bfs[:,:,l] * bfs[:,:,l-1]

    return bfs_compound


@pytest.fixture(scope="class")
def example_conditionals():

    likelihood = np.array(
        [
            [0.        , 1.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 1.        , 0.        , 0.        ],
            [0.        , 0.        , 0.29411765, 0.        , 0.70588235, 0.        ],
            [0.72727273, 0.        , 0.        , 0.        , 0.27272727, 0.        ],
            [0.40677966, 0.        , 0.        , 0.        , 0.        , 0.59322034]
        ]
    )

    posterior = np.array(
        [
            [0. , 0. , 0. , 0.4, 0.6],
            [1. , 0. , 0. , 0. , 0. ],
            [0. , 0. , 1. , 0. , 0. ],
            [0. , 1. , 0. , 0. , 0. ],
            [0. , 0. , 0.8, 0.2, 0. ],
            [0. , 0. , 0. , 0. , 1. ]
        ]
    )

    total = 37_789
    prior = np.array(
        [
            8789.0,
            6000.0,
            6800.0,
            4400.0,
            11800.0
        ]
    )/total

    marginal = np.array(
        [
            8000.0,
            8789.0,
            2000.0,
            6000.0,
            6000.0,
            7000.0
        ]
    )/total

    source = np.array(
        [
            18789.0,
            6000.0,
            0.0,
            6000.0,
            7000.0,
        ]
    )/total

    edge_source_cond = np.array(
        [
            [0.42578104, 0.4677737 , 0.10644526, 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 1.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 1.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        , 1.        ]
        ]
    )

    return posterior, likelihood, prior, marginal, source, edge_source_cond

@pytest.fixture(scope="class",
    params=[
        {
            "merge_communities_test": {
                'AT': 0,
                'BT': 1,
                'CT': 2,
                'HT': 3,
                'VT': 4
            },
            "test_HHI": {
                ('0', '1'): 5200.0 / 10000,
                ('0', '2'): 10000.0 / 10000,
                ('0', '4'): 10000.0 / 10000,
                ('1', '2'): 10000.0 / 10000,
                ('3', '4'): 6800.0 / 10000,
                ('4', '0'): 10000.0 / 10000
            },
            "test_weights_dict": {
                'AT': 0,
                'BT': 0,
                'CT': 0,
                'HT': 0,
                'VT': 4800
            }
        },
        {
            "merge_communities_test": {
                'AT': 1,
                'BT': 1,
                'CT': 2,
                'HT': 3,
                'VT': 4
            },
            "test_HHI": {
                ('0', '1'): 5200.0 / 10000,
                ('0', '2'): 10000.0 / 10000,
                ('0', '4'): 10000.0 / 10000,
                ('1', '2'): 10000.0 / 10000,
                ('3', '4'): 6800.0 / 10000,
                ('4', '0'): 10000.0 / 10000
            },
            "test_weights_dict": {
                'AT': 0,
                'BT': 0,
                'CT': 0,
                'HT': 0,
                'VT': 4800
            }
        },
        {
            "merge_communities_test": {
                'AT': 0,
                'BT': 1,
                'CT': 4,
                'HT': 3,
                'VT': 4
            },
            "test_HHI": {
                ('0', '1'): 5200.0 / 10000,
                ('0', '2'): 10000.0 / 10000,
                ('0', '4'): 10000.0 / 10000,
                ('1', '2'): 10000.0 / 10000,
                ('3', '4'): 6800.0 / 10000,
                ('4', '0'): 10000.0 / 10000
            },
            "test_weights_dict": {
                'AT': 0,
                'BT': 0,
                'CT': 7000,
                'HT': 0,
                'VT': 6800 # 4800 + 2000
            }
        },
        {
            "merge_communities_test": {
                'AT': 0,
                'BT': 1,
                'CT': 2,
                'HT': 4,
                'VT': 4
            },
            "test_HHI": {
                ('0', '1'): 10000.0 / 10000,
                ('0', '2'): 10000.0 / 10000,
                ('0', '4'): 10000.0 / 10000,
                ('1', '2'): 10000.0 / 10000,
                ('3', '4'): 6800.0 / 10000,
                ('4', '0'): 10000.0 / 10000
            },
            "test_weights_dict": {
                'AT': 0,
                'BT': 0,
                'CT': 0,
                'HT': 7000,
                'VT': 8000
            }
        },
    ],
    ids=[
        'default init',
        'nondefault init 1',
        'nondefault init 2',
        'nondefault init 3'
    ]
)
def example_case_param(request):

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
    data_dir = os.path.join('..','test_data')
    MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph
    with open(os.path.join(script_dir,data_dir,'test_airline_nodes.json'),'r') as file:
        nodes_airlines_dict = json.load(file)

    test_weights_individual_dict = {
        'AT': 8789,
        'BT': 6000,
        'CT': 6800,
        'HT': 4400,
        'VT': 11800
    }

    param = request.param
    communities_test = param["merge_communities_test"]
    test_HHI = param["test_HHI"]
    test_weights_dict = param["test_weights_dict"]

    return MG, nodes_airlines_dict, \
        test_HHI, \
        test_weights_dict, test_weights_individual_dict, \
        communities_test

@pytest.fixture(scope="class")
def example_case():

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
    data_dir = os.path.join('..','test_data')
    MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph
    with open(os.path.join(script_dir,data_dir,'test_airline_nodes.json'),'r') as file:
        nodes_airlines_dict = json.load(file)

    communities_test = {
        'AT': 0,
        'BT': 1,
        'CT': 2,
        'HT': 3,
        'VT': 4
    }

    test_HHI = {
        ('0', '1'): 5200.0,
        ('0', '2'): 10000.0,
        ('0', '4'): 10000.0,
        ('1', '2'): 10000.0,
        ('3', '4'): 6800.0,
        ('4', '0'): 10000.0
    }

    test_weights_dict = {
        'AT': 0,
        'BT': 0,
        'CT': 0,
        'HT': 0,
        'VT': 4800
    }

    test_weights_individual_dict = {
        'AT': 8789,
        'BT': 6000,
        'CT': 6800,
        'HT': 4400,
        'VT': 11800
    }

    return MG, nodes_airlines_dict, \
        test_HHI, \
        test_weights_dict, test_weights_individual_dict, \
        communities_test

@pytest.fixture(scope="class",
    params=[
        {
            "community_to_merge": 0,
            "new_community": 1,
            "merge_communities_test": {
                'AT': 1,
                'BT': 1,
                'CT': 2,
                'HT': 3,
                'VT': 4
            },
            "test_HHI": {
                ('0', '1'): 5200.0 / 10000,
                ('0', '2'): 10000.0 / 10000,
                ('0', '4'): 10000.0 / 10000,
                ('1', '2'): 10000.0 / 10000,
                ('3', '4'): 6800.0 / 10000,
                ('4', '0'): 10000.0 / 10000
            },
            "test_weights_dict": {
                'AT': 0,
                'BT': 0,
                'CT': 0,
                'HT': 0,
                'VT': 4800
            }
        },
        {
            "community_to_merge": 2,
            "new_community": 4,
            "merge_communities_test": {
                'AT': 0,
                'BT': 1,
                'CT': 4,
                'HT': 3,
                'VT': 4
            },
            "test_HHI": {
                ('0', '1'): 5200.0 / 10000,
                ('0', '2'): 10000.0 / 10000,
                ('0', '4'): 10000.0 / 10000,
                ('1', '2'): 10000.0 / 10000,
                ('3', '4'): 6800.0 / 10000,
                ('4', '0'): 10000.0 / 10000
            },
            "test_weights_dict": {
                'AT': 0,
                'BT': 0,
                'CT': 7000,
                'HT': 0,
                'VT': 6800 # 4800 + 2000
            }
        },
        {
            "community_to_merge": 3,
            "new_community": 4,
            "merge_communities_test": {
                'AT': 0,
                'BT': 1,
                'CT': 2,
                'HT': 4,
                'VT': 4
            },
            "test_HHI": {
                ('0', '1'): 10000.0 / 10000,
                ('0', '2'): 10000.0 / 10000,
                ('0', '4'): 10000.0 / 10000,
                ('1', '2'): 10000.0 / 10000,
                ('3', '4'): 6800.0 / 10000,
                ('4', '0'): 10000.0 / 10000
            },
            "test_weights_dict": {
                'AT': 0,
                'BT': 0,
                'CT': 0,
                'HT': 7000,
                'VT': 8000
            }
        },
    ],
    ids=[
        'no change in scores',
        'change in airline scores',
        'change in all scores'
    ]
)
def example_merge(request, example_case):
    MG, nodes_airlines_dict, \
    test_HHI, test_weights_dict, \
    test_weights_individual_dict, \
    init_communities_test = example_case

    param = request.param
    community_to_merge = param["community_to_merge"]
    new_community = param["new_community"]
    merge_communities_test = param["merge_communities_test"]
    test_HHI = param["test_HHI"]
    test_weights_dict = param["test_weights_dict"]

    return MG, \
           community_to_merge, new_community, \
           merge_communities_test, test_HHI, \
           test_weights_dict, test_weights_individual_dict

@pytest.fixture(scope="class",
    params=[
        {
            "weights": (1.0,0.0),
            "test_communities_dict": {
                'AT': 0,
                'BT': 1,
                'CT': 2,
                'HT': 3,
                'VT': 4
            },
        },
        {
            "weights": (-1.0,0.0),
            "test_communities_dict": {
                'AT': 0,
                'BT': 1,
                'CT': 2,
                'HT': 2,
                'VT': 2
            },
        },
        {
            "weights": (0.0,1.0),
            "test_communities_dict": {
                'AT': 0,
                'BT': 0,
                'CT': 0,
                'HT': 0,
                'VT': 0
            },
        },
    ],
    ids=[
        'minimize HHI',
        'maximize HHI',
        'maximize MAI',
    ]
)
def optimization_cases(request, example_case):
    MG, nodes_airlines_dict, \
    test_HHI, test_weights_dict, \
    test_weights_individual_dict, \
    init_communities_test = example_case

    param = request.param
    weights = param["weights"]
    test_communities_dict = param["test_communities_dict"]

    return MG, weights, test_communities_dict

@pytest.mark.parametrize("dict1, dict2, expected", [
    ({'AT': 0, 'BT': 1, 'CT': 0, 'HT': 1, 'VT': 0}, {'AT': 1, 'BT': 0, 'CT': 1, 'HT': 0, 'VT': 1}, True),
    ({'AT': 0, 'BT': 1, 'CT': 0, 'HT': 1, 'VT': 0}, {'AT': 2, 'BT': 0, 'CT': 2, 'HT': 0, 'VT': 2}, True),
    ({'AT': 0, 'BT': 1, 'CT': 0, 'HT': 1, 'VT': 0}, {'AT': 2, 'BT': 1, 'CT': 2, 'HT': 1, 'VT': 2}, True),
    ({'AT': 0, 'BT': 1, 'CT': 0, 'HT': 1, 'VT': 0}, {'AT': 1, 'BT': 2, 'CT': 1, 'HT': 0, 'VT': 1}, False),
], ids=['case 1', 'case 2', 'case 3', 'case 4'])
def test_same_community(dict1, dict2, expected):
    assert same_community(dict1, dict2) == expected

@pytest.mark.parametrize(
    'input_dict, test_dict, test_unique, test_n_communities', [
        (
            {
                '1' : 1,
                '2' : 1,
                '3' : 2,
                '4' : 5,
                '5' : 6,
                '6' : 6,
            },
            {
                '1' : 0,
                '2' : 0,
                '3' : 1,
                '4' : 2,
                '5' : 3,
                '6' : 3,
            },
            [
                0,1,2,3
            ],
            4
        ),
        (
            {
                '1' : 4,
                '2' : 4,
                '3' : 2,
                '4' : 5,
                '5' : 6,
                '6' : 6,
            },
            {
                '1' : 1,
                '2' : 1,
                '3' : 0,
                '4' : 2,
                '5' : 3,
                '6' : 3,
            },
            [
                0,1,2,3
            ],
            4
        )
    ], ids=[
        'case 1',
        'case 2'
    ]
)
def test_reorder(input_dict, test_dict, test_unique, test_n_communities):
    orig2sorted, sorted2orig, reorder_dict, unique_list, n_communities = resort_communities(input_dict)
    assert n_communities == test_n_communities
    assert reorder_dict == test_dict
    assert unique_list == test_unique
    assert same_community(input_dict,reorder_dict) # check community structure is preserved

@pytest.mark.parametrize("test_vector, threshold, expected", [
    (np.array([0, 0, 0, 40, 60]), 0.3, 3),
    (np.array([0, 40, 0, 0, 60]), 0.3, 1),
    (np.array([0, 0, 0, 40, 60]), 0.4, 4),
    (np.array([0, 40, 0, 0, 60]), 0.5, 4),
], ids=['case 1', 'case 2', 'case 3', 'case 4'])
def test_sampler_deterministic(test_vector, threshold, expected):

    from ..utilities_cuda import cumsum_h, searchsorted_h

    cumsum = np.zeros_like(test_vector)

    cumsum_h(test_vector,cumsum)
    chosen = searchsorted_h(cumsum,threshold)

    assert chosen == expected

@pytest.mark.parametrize("test_vector, expected", [
    (np.array([0, 0, 0, 40, 60]), np.array([0,0,0,0.4,0.6])),
    (np.array([0, 40, 0, 0, 60]), np.array([0,0.4,0,0.0,0.6])),
], ids=['case 1', 'case 2'])
def test_sampler(test_vector, expected):

    from ..utilities_cuda import cumsum_h, searchsorted_h
    import random
    random.seed(seed)
    n_samples = 50

    cumsum = np.zeros_like(test_vector)
    counts = np.zeros_like(test_vector)

    cumsum_h(test_vector,cumsum)

    for _ in range(n_samples):
        threshold = random.random()
        chosen = searchsorted_h(cumsum,threshold)
        counts[chosen] += 1

    p = counts / counts.sum()

    assert np.allclose(p,expected,rtol=1e-1)

def test_cuda_blocks():

    dims = (500, 500, 500)
    TPB = (8, 8, 8)
    dims = (5000, 50)
    TPB = (32, 32)

    nstates, threadsperblock, blockspergrid = cuda_blocks(dims=dims, TPB=TPB)

    return

@pytest.mark.parametrize("reduced_counts_base, reduced_counts", [
    (
        np.array(
            [
                [  0,   0,   0, 191, 309],
                [500,   0,   0,   0,   0],
                [  0,   0, 500,   0,   0],
                [  0, 500,   0,   0,   0],
                [  0,   0, 399, 101,   0],
                [  0,   0,   0,   0, 500]
            ], dtype = np.int32
        ),
        np.array(
            [
                [  0,   0,   0, 191, 309],
                [500,   0,   0,   0,   0],
                [  0,   0, 500,   0,   0],
                [  0, 500,   0,   0,   0],
                [  0,   0,   0, 399, 101],
                [  0,   0,   0,   0, 500]
            ], dtype = np.int32
        )
    ),
    (
        np.array(
            [
                [  0,   0,   0, 191, 309],
                [500,   0,   0,   0,   0],
                [  0,   0, 500,   0,   0],
                [  0, 500,   0,   0,   0],
                [  0,   0, 399, 101,   0],
                [  0,   0,   0,   0, 500]
            ], dtype = np.int32
        ),
        np.array(
            [
                [  0,   0,   0, 191, 309],
                [500,   0,   0,   0,   0],
                [  0,   0, 500,   0,   0],
                [  0, 500,   0,   0,   0],
                [  0,   0,   0, 500,   0],
                [  0,   0,   0,   0, 500]
            ], dtype = np.int32
        )
    )
], ids=['no difference', 'difference'])
def test_hist_diff(reduced_counts_base, reduced_counts):

    kernels = define_cuda_kernels_stochastic(1, 1, 500, 1, 1)

    set_membership             = kernels[0]
    spatial_random_walk        = kernels[1]
    attr_sampler               = kernels[2]
    attr_source_postprocess    = kernels[3]
    edge_sampler               = kernels[4]
    init_edge_counts           = kernels[5]
    edge_postprocess           = kernels[6]
    check_edge_distribution    = kernels[7]
    load_counts                = kernels[8]
    edge_histogram             = kernels[9]
    copy_reduced_counts        = kernels[10]
    init_attr                  = kernels[11]
    init_node_weights          = kernels[12]
    init_community             = kernels[13]
    node_weight                = kernels[14]
    attr_weight                = kernels[15]
    community_weight           = kernels[16]
    attr_postprocess           = kernels[17]

    reduced_counts_base_d = cuda.to_device(reduced_counts_base)
    reduced_counts_d = cuda.to_device(reduced_counts)
    distributions_changed_d = cuda.device_array(reduced_counts.shape[0], dtype=np.int32)

    reduced_counts_base_h = reduced_counts_base
    reduced_counts_h = reduced_counts
    # Assuming your matrices are named matrix1 and matrix2
    hhi_new = np.sort(reduced_counts_h, axis=1)
    hhi_base = np.sort(reduced_counts_base_h, axis=1)
    distributions_changed_test = np.any(hhi_new != hhi_base, axis=1)

    nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(reduced_counts.shape[0], ), TPB = 8)

    check_edge_distribution[threadsperblock,blockspergrid](
        reduced_counts_base_d,
        reduced_counts_d,
        distributions_changed_d
    )

    cuda.synchronize()

    distributions_changed = distributions_changed_d.copy_to_host()
    distributions_changed = distributions_changed > 0

    assert np.all(distributions_changed_test == distributions_changed)

class TestDeterministic():
    n_samples_edge=500
    walk_length=10
    n_samples_source_node = 50
    use_gpu = False

    @pytest.mark.dependency()
    def test_initialization_deterministic(self,example_case_param):

        MG, nodes_airlines_dict, \
        test_HHI, test_weights_dict, \
        test_weights_individual_dict, \
        init_communities_test = example_case_param

        test_class = DeterministicCommunity(MG, communities_dict=init_communities_test, use_gpu=self.use_gpu)

        # check all keys in test dictionary are present
        assert all(key in test_HHI.keys() for key in test_class.hhi)

        for key,value in test_class.hhi.items():
            assert abs(test_HHI[key] - value) < tolerance

        assert all(key in test_weights_dict.keys() for key in test_class.bfs_degree)

        for key,value in test_class.bfs_degree.items():
            assert value == test_weights_dict[key] + test_weights_individual_dict[key]

        assert same_community(init_communities_test,test_class.communities_dict)

    @pytest.fixture(scope="function")
    def init_class(self, example_case) -> DeterministicCommunity:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = DeterministicCommunity(MG, seed=seed, use_gpu=self.use_gpu, num_streams=100)
        test_class.init_random()

        return test_class

    def test_save_load(self, example_case, tmp_path):

        directory = tmp_path / "test_save"
        directory.mkdir()

        MG, _, _, _, _, _ = example_case

        test_class = DeterministicCommunity(MG, seed=seed, use_gpu=self.use_gpu, num_streams=100)
        test_class.init_random()

        test_class.greedy_optimize()

        test_class.save(directory=directory)

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

        test_data = {}
        for name, attr in data_to_load.items():
            test_data[name] = getattr(test_class,attr["name"])

        del test_class

        test_class = DeterministicCommunity(MG, seed=seed, use_gpu=self.use_gpu)
        test_class.load(directory=directory)

        for name, attr in data_to_load.items():
            if attr["dtype"] == np.ndarray:
                assert np.allclose(getattr(test_class,attr["name"]), test_data[name], rtol=1e-1)
            elif attr["dtype"] == list:
                for value, test_value in zip(getattr(test_class,attr["name"]), test_data[name]):
                    assert abs(value - test_value) < 1e-6
            elif attr["dtype"] == List[Dict[str,int]]:
                for value, test_value in zip(getattr(test_class,attr["name"]), test_data[name]):
                    assert value == test_value
            elif attr["dtype"] == List[Tuple[float,float]]:
                for value, test_value in zip(getattr(test_class,attr["name"]), test_data[name]):
                    for value_i, test_value_i in zip(value, test_value):
                        assert abs(value_i - test_value_i) < 1e-6
            else:
                assert getattr(test_class,attr["name"]) == test_data[name]

    def test_optimize(self, init_class: DeterministicCommunity, optimization_cases):

        _, weights, test_communities_dict = optimization_cases

        test_class = init_class

        test_class.greedy_optimize(weights=weights)

        assert same_community(test_class.communities_dict, test_communities_dict)

        print(test_class.f_history)
        print(test_class.communities_history)


class TestDeterministicCPU(TestDeterministic):
    use_gpu = False

    @pytest.mark.dependency()
    def test_sampler(self,example_case_param):

        MG, nodes_airlines_dict, \
        test_HHI, test_weights_dict, \
        test_weights_individual_dict, \
        init_communities_test = example_case_param

        test_class = DeterministicCommunity(MG, communities_dict=init_communities_test)
        n_omit_samples = int(test_class.n_communities*(test_class.n_communities+1)/2)
        n_test_samples = test_class.n_communities**2 - n_omit_samples

        assert test_class.n_communities**2 - len(test_class._sampled_communities_pairs) == n_test_samples

        while samples:=test_class._sample_community_pairs(3):
            assert len(samples) <= n_test_samples

    @pytest.mark.dependency()
    def test_sampler_dist(self,example_case_param):

        n_samples = 1000

        MG, nodes_airlines_dict, \
        test_HHI, test_weights_dict, \
        test_weights_individual_dict, \
        init_communities_test = example_case_param

        test_class = DeterministicCommunity(MG, communities_dict=init_communities_test)

        all_samples = list(itertools.product(range(test_class.n_communities), range(test_class.n_communities)))
        all_samples_dict = {pair : i for i,pair in enumerate(all_samples)}
        diag_mask = np.array([True if i==j else False for i,j in all_samples])

        probabilities = test_class.bfs_degree_community[all_samples] / np.sum(test_class.bfs_degree_community)
        probabilities_pairs = probabilities[:,0] * probabilities[:,1]
        probabilities_pairs[diag_mask] = 0.0
        probabilities_pairs = probabilities_pairs / np.sum(probabilities_pairs)


        samples = test_class._sample_community_pairs(n_samples,replace=True)
        samples_indices = [all_samples_dict[pair] for pair in samples]

        # Compute the empirical PMF
        unique, counts = np.unique(samples_indices, return_counts=True)
        counts_full = np.zeros_like(probabilities_pairs)
        counts_full[unique] = counts
        empirical_pmf = counts_full / len(samples_indices)

        kl_divergence = kl_div(empirical_pmf + 1e-10, probabilities_pairs + 1e-10)

        assert np.mean(empirical_pmf - probabilities_pairs) <= 1e-3

        return

    def test_merge_D_CPU(self, init_class: DeterministicCommunity, example_merge):

        test_class = init_class

        _, \
        community_to_merge, new_community, \
        merge_communities_test, test_HHI, \
        test_weights_dict, test_weights_individual_dict = example_merge

        test_class.update_scores(community_to_merge,new_community)

        # check all keys in test dictionary are present
        assert all(key in test_HHI.keys() for key in test_class.hhi)

        assert same_community(test_class.communities_dict, merge_communities_test)

        for key,value in test_class.hhi.items():
            assert abs(test_HHI[key] - value) < tolerance

        for key,value in test_class.bfs_degree.items():
            assert value == test_weights_dict[key] + test_weights_individual_dict[key]


class TestDeterministicGPU(TestDeterministic):
    num_streams = 4
    tile_size = 2
    threads_per_block=8
    use_gpu = True

    @pytest.fixture(scope="class")
    def init_class_iteration(self, example_case) -> DeterministicCommunity:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = DeterministicCommunity(MG, seed=seed, use_gpu=self.use_gpu, num_streams=100)
        test_class.init_random()

        membership_pairs, hhi_pairs, weights_pairs, score_pairs = test_class.greedy_optimize(
            max_iterations=1,
            batch_size=1,
            debug=True
        )

        return test_class, membership_pairs, hhi_pairs, weights_pairs

    def test_merge_D_GPU(
            self,
            init_class_iteration: Tuple[DeterministicCommunity,np.ndarray,np.ndarray,np.ndarray,np.ndarray],
            example_merge
        ):

        test_class, membership_pairs, hhi_pairs, weights_pairs = init_class_iteration

        _, \
        community_to_merge, new_community, \
        merge_communities_test, test_HHI, \
        test_weights_dict, _ = example_merge

        for key,value in zip(test_class.attr_list,membership_pairs[community_to_merge,new_community]):
            assert value == merge_communities_test[key]

        for key,value in zip(test_class.edge_unique_list,hhi_pairs[community_to_merge,new_community]):
            assert abs(test_HHI[key] - value) < tolerance

        for key,value in zip(test_class.attr_list,weights_pairs[community_to_merge,new_community]):
            assert test_weights_dict[key] == value


class TestStochastic():

    n_samples_edge_low=10
    n_samples_edge=500
    n_samples_edge_high = 5000
    walk_length=10
    threads_per_block=8
    n_samples_source_node = 50
    num_streams = 4
    tile_size = 2
    parallel_reduce = False
    use_gpu = False

    @pytest.mark.dependency()
    def test_initialization_stochastic(self,example_case_param):

        MG, nodes_airlines_dict, \
        test_HHI, test_weights_dict, \
        test_weights_individual_dict, \
        init_communities_test = example_case_param

        test_class = StochasticCommunity(
            MG, seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge,
            walk_length=self.walk_length,
            communities_dict=init_communities_test
        )

        test_class.init_random()

        # check all keys in test dictionary are present
        assert all(key in test_HHI.keys() for key in test_class.hhi)

        for key,value in test_class.hhi.items():
            assert abs(test_HHI[key] - value) / test_HHI[key] <= 1e-1

        assert all(key in test_weights_dict.keys() for key in test_class.bfs_degree)

        assert same_community(init_communities_test,test_class.communities_dict)

    @pytest.fixture(scope="function")
    def init_class_less_samples(self, example_case) -> StochasticCommunity:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = StochasticCommunity(
            MG, seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge_low,
            walk_length=self.walk_length,
            num_streams=self.num_streams
        )
        test_class.init_random(tile_size=self.tile_size)

        return test_class


    @pytest.fixture(scope="function")
    def init_class(self, example_case) -> StochasticCommunity:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = StochasticCommunity(
            MG, seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge,
            walk_length=self.walk_length,
            num_streams=self.num_streams
        )

        test_class.init_random(tile_size=self.tile_size)

        return test_class

    @pytest.fixture(scope="function")
    def init_class_many_samples(self, example_case) -> StochasticCommunity:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = StochasticCommunity(
            MG, seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge_high,
            walk_length=self.walk_length
        )

        test_class.init_random(tile_size=self.tile_size)

        return test_class

    def test_probabilities(self, init_class: StochasticCommunity, example_case, example_conditionals):
        MG, _, _, _, _, _ = example_case
        test_posterior, test_likelihood, test_prior, test_marginal, \
            test_source, test_edge_source_cond = example_conditionals

        test_class = init_class

        posterior = get_conditional_prob(test_class)
        likelihood = get_likelihood_prob(test_class)

        prior = get_prior(test_class)
        marginal = get_marginal(test_class)

        edge_p, edge_source_cond_p = get_edge_prob(test_class)
        source_p = get_source_prob(test_class)

        assert np.allclose(test_posterior, posterior, 1e-3)
        assert np.allclose(test_likelihood, likelihood, 1e-3)
        assert np.allclose(test_prior, prior, 1e-3)
        assert np.allclose(test_marginal, marginal, 1e-3)
        assert np.allclose(test_edge_source_cond, edge_source_cond_p, 1e-3)
        assert np.allclose(test_marginal, edge_p, 1e-3)
        assert np.allclose(test_source, source_p, 1e-3)

class TestStochasticGPU(TestStochastic):

    use_gpu = True

    @pytest.fixture(scope="class")
    def baseline_stochastic(self, example_case) -> Tuple[np.ndarray, np.ndarray]:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = StochasticCommunity(
            MG, seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge,
            walk_length=self.walk_length,
            num_streams=1
        )
        test_class.init_random(tile_size=self.tile_size)

        membership_pairs = test_class._calculate_members_gpu()

        edge_samples, walks_spatial, walks_edges_spatial, \
            walks_attr, walks_attr_cp, \
                source_attr_counts, attr_weights, node_weights = test_class._randomwalk_gpu(testing=True)

        _, _, _, _, cum_weight, hhi_estimate, mai_estimate, _ = test_class._stochastic_eval_gpu(
            0,
            0,
            edge_samples,
            walks_attr_cp,
            walks_attr,
            membership_pairs,
            stream = test_class.cuda.stream(),
            stream_id = 0,
            testing = True,
            parallel_reduce=self.parallel_reduce
        )

        estimate_bfs = get_estimate_node_bfs_probabilities(
            test_class,
            n_samples_edge=self.n_samples_edge,
            walk_length=self.walk_length,
            walks_attr=walks_attr,
            walks_nodes_spatial=walks_spatial
        )

        bfs = get_bfs_probabilities(test_class,self.walk_length-1)

        return cum_weight, hhi_estimate, mai_estimate, bfs, estimate_bfs

    def test_counters(self, init_class_less_samples: StochasticCommunity):
        test_class = init_class_less_samples
        community_to_merge = 0
        new_community = 0

        membership_pairs = test_class._calculate_members_gpu()

        edge_samples, walks_spatial, walks_edges_spatial, \
            walks_attr, walks_attr_cp, \
                source_attr_counts, attr_weights, node_weights = test_class._randomwalk_gpu(testing=True)

        # Convert edge_samples and membership_pairs to device arrays
        membership_pairs = test_class._calculate_members_gpu()

        edge_samples, walks_spatial, walks_edges_spatial, \
            walks_attr, walks_attr_cp, \
                source_attr_counts, attr_weights, node_weights = test_class._randomwalk_gpu(testing=True)

        _, _, _, edge_counts, _, _, _, _ = test_class._stochastic_eval_gpu(
            community_to_merge,
            new_community,
            edge_samples,
            walks_attr_cp,
            walks_attr,
            membership_pairs,
            stream = test_class.cuda.stream(),
            stream_id = 0,
            testing = True,
            parallel_reduce = self.parallel_reduce
        )

        # Expected counts calculation on host for verification
        expected_counts = np.zeros((test_class.n_unique_edges, test_class.n_communities), dtype=int)
        for i, sample_row in enumerate(edge_samples):
            for sample in sample_row:
                expected_counts[i, membership_pairs[community_to_merge, new_community, sample]] += 1

        # Assert the results
        np.testing.assert_array_equal(edge_counts * test_class.n_samples_edge, expected_counts)

    def test_merge_GPU(
            self,
            init_class: StochasticCommunity,
            example_merge,
            baseline_stochastic
        ):

        _, \
        community_to_merge, new_community, \
        merge_communities_test, test_HHI, \
        test_weights_dict, test_weights_individual_dict = example_merge

        test_class = init_class

        cum_weight_baseline, hhi_estimate_baseline, \
            mai_estimate_baseline, bfs_baseline, estimate_bfs_baseline = baseline_stochastic

        membership_pairs = test_class._calculate_members_gpu()

        edge_samples, walks_attr, walks_attr_cp, edge_counts, cum_weight, hhi_estimate, mai_estimate, log_mai = test_class._stochastic_eval_gpu(
            community_to_merge,
            new_community,
            test_class._edge_samples_d,
            test_class._walks_attr_cp_d,
            test_class._walks_attr_d,
            membership_pairs,
            stream = test_class.cuda.stream(),
            stream_id = 0,
            testing = True,
            parallel_reduce = self.parallel_reduce
        )

        test_class.update_scores(community_to_merge, new_community, stream_id=0)
        test_class.communities_dict = test_class._communities_dict

        bfs = get_bfs_probabilities(test_class,self.walk_length-1)

        for key,value in zip(test_class.attr_list,membership_pairs[community_to_merge,new_community]):
            assert value == merge_communities_test[key]

        for key,value in zip(test_class.edge_unique_list,hhi_estimate):
            assert math.isclose(test_HHI[key],value,rel_tol=1e-1)

        assert np.all(np.sign(cum_weight_baseline - cum_weight) == np.sign(bfs_baseline.sum(axis=2) - bfs.sum(axis=2)))

    @pytest.mark.parametrize("n_sample_pairs", [12, 8], ids=['no sampling', '8 samples'])
    def test_optimize_GPU(self, init_class: StochasticCommunity, optimization_cases, n_sample_pairs):
        _, weights, test_communities_dict = optimization_cases

        test_class = init_class

        membership_pairs, hhi_pairs, weights_pairs = test_class.greedy_optimize(
                weights=weights,
                testing_opt=True,
                n_sample_pairs=n_sample_pairs,
                epsilon = 0.5,
                rtol = 0.0
            )

        assert same_community(test_class.communities_dict, test_communities_dict)

        print(test_class.f_history)
        print(test_class.communities_history)

    def test_randomwalker(self, example_case, init_class_many_samples: StochasticCommunity):

        MG, _, _, _, _, _ = example_case
        test_class = init_class_many_samples

        membership_pairs = test_class._calculate_members_gpu()

        edge_samples, walks_spatial, walks_edges_spatial, \
            walks_attr, walks_attr_cp, \
                source_counts, attr_weights, node_weights = test_class._randomwalk_gpu(testing=True)

        edge_samples, walks_attr, walks_attr_cp, edge_counts, cum_weight, _, _, _ = test_class._stochastic_eval_gpu(
            0,
            0,
            edge_samples,
            walks_attr_cp,
            walks_attr,
            membership_pairs,
            stream = test_class.cuda.stream(),
            testing = True,
            parallel_reduce = self.parallel_reduce
        )

        assert np.allclose(test_class._adj3/100,edge_counts,rtol=1e-1)

class TestStochasticCPU(TestStochastic):

    use_gpu = False

    @pytest.fixture(scope="function")
    def baseline_stochastic(self, example_case) -> Tuple[StochasticCommunity, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = StochasticCommunity(
            MG, seed=seed,
            use_gpu=False,
            n_samples_edge=self.n_samples_edge,
            walk_length=self.walk_length
        )
        test_class.init_random(tile_size=self.tile_size)

        membership_pairs = test_class._calculate_members_cpu()

        edge_samples, walks_spatial, walks_edges_spatial, \
            walks_attr, walks_attr_cp, \
                source_attr_counts, attr_weights, node_weights = test_class._randomwalk_cpu(testing=True)

        _, _, _, community_weight, _, cum_weight, hhi_estimate, mai_estimate, _ = test_class._stochastic_eval_cpu(
            0,
            0,
            edge_samples,
            walks_attr_cp,
            walks_attr,
            membership_pairs,
            testing = True,
        )

        estimate_bfs = get_estimate_node_bfs_probabilities(
            test_class,
            n_samples_edge=self.n_samples_edge,
            walk_length=self.walk_length,
            walks_attr=walks_attr,
            walks_nodes_spatial=walks_spatial
        )

        bfs = get_bfs_probabilities(test_class,self.walk_length-1)

        return test_class, community_weight, cum_weight, hhi_estimate, mai_estimate, bfs, estimate_bfs

    def test_counters(
            self,
            init_class_less_samples: StochasticCommunity,
        ):
        test_class = init_class_less_samples
        community_to_merge = 0
        new_community = 0

        membership_pairs = test_class._calculate_members_cpu()

        edge_samples, walks_spatial, walks_edges_spatial, \
            walks_attr, walks_attr_cp, \
                source_attr_counts, attr_weights, node_weights = test_class._randomwalk_cpu(testing=True)

        # Expected counts should be calculated based on the logic of edge_postprocess_cpu
        # Assuming each number in samples corresponds to an index in the membership array
        expected_counts = np.zeros((len(edge_samples), test_class.n_communities), dtype=int)
        for i, sample_row in enumerate(edge_samples):
            for sample in sample_row:
                expected_counts[i, membership_pairs[community_to_merge, new_community, sample]] += 1

        reduced_counts = edge_postprocess_cpu(
            ci = community_to_merge,
            cj = new_community,
            samples = edge_samples,
            membership = membership_pairs,
            n_communities = test_class.n_communities
        )

        np.testing.assert_array_equal(reduced_counts, expected_counts)

    def test_merge_CPU(
            self,
            init_class: StochasticCommunity,
            example_merge,
            baseline_stochastic: Tuple[StochasticCommunity, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ):

        _, \
        community_to_merge, new_community, \
        merge_communities_test, test_HHI, \
        test_weights_dict, test_weights_individual_dict = example_merge

        test_class = init_class

        test_class, community_weight_baseline, cum_weight_baseline, hhi_estimate_baseline, \
            mai_estimate_baseline, bfs_baseline, estimate_bfs_baseline = baseline_stochastic

        membership_pairs = test_class._calculate_members_cpu()

        edge_samples, walks_attr, walks_attr_cp, community_weight, edge_counts, cum_weight, hhi_estimate, mai_estimate, log_mai = test_class._stochastic_eval_cpu(
            community_to_merge,
            new_community,
            test_class._edge_samples,
            test_class._walks_attr_cp,
            test_class._walks_attr,
            membership_pairs,
            stream_id = 0,
            testing = True,
        )

        test_class.update_scores(community_to_merge, new_community)
        test_class.communities_dict = test_class._communities_dict

        bfs = get_bfs_probabilities(test_class,self.walk_length-1)

        for key,value in zip(test_class.attr_list,membership_pairs[community_to_merge,new_community]):
            assert value == merge_communities_test[key]

        for key,value in zip(test_class.edge_unique_list,hhi_estimate):
            assert math.isclose(test_HHI[key],value,rel_tol=100e-1)

        assert np.all(np.sign(cum_weight_baseline - cum_weight) == np.sign(bfs_baseline.sum(axis=2) - bfs.sum(axis=2)))

    @pytest.mark.parametrize("n_sample_pairs", [12, 8], ids=['no sampling', '8 samples'])
    def test_optimize_CPU(self, init_class: StochasticCommunity, optimization_cases, n_sample_pairs):
        _, weights, test_communities_dict = optimization_cases

        test_class = init_class

        membership_pairs, hhi_pairs, weights_pairs = test_class.greedy_optimize(
                weights=weights,
                testing_opt=True,
                n_sample_pairs=n_sample_pairs,
                epsilon = 0.5,
                rtol = 0.0
            )

        assert same_community(test_class.communities_dict, test_communities_dict)

        print(test_class.f_history)
        print(test_class.communities_history)

    def test_randomwalker(self, example_case, init_class_many_samples: StochasticCommunity):

        MG, _, _, _, _, _ = example_case
        test_class = init_class_many_samples

        membership_pairs = test_class._calculate_members_cpu()

        edge_samples, walks_spatial, walks_edges_spatial, walks_attr, walks_attr_cp, source_counts, attr_weights, node_weights = test_class._randomwalk_cpu(testing=True)

        edge_samples, walks_attr, walks_attr_cp, community_weight, edge_counts, cum_weight, _, _, _ = test_class._stochastic_eval_cpu(
            0,
            0,
            edge_samples = edge_samples,
            walks_nodes_weights = walks_attr_cp,
            walks_nodes = walks_attr,
            membership_pairs = membership_pairs,
            testing = True
        )

        assert np.allclose(test_class._adj3/100,edge_counts,rtol=1e-1)


def test_attr_postprocess():
    # This setup will run once for the entire class

    n_samples_edge_low=10
    walk_length=10
    num_streams = 4
    tile_size = 2

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
    data_dir = os.path.join('..','test_data')
    MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL_blank.graphml')) # store graph

    test_class_gpu = StochasticCommunity(
        MG, seed=seed,
        use_gpu=True,
        n_samples_edge=n_samples_edge_low,
        walk_length=walk_length,
        num_streams=num_streams
    )
    test_class_gpu.init_random(tile_size=tile_size)

    kernels = define_cuda_kernels_stochastic(
        n_attr = test_class_gpu.n_attr, 
        n_nodes = test_class_gpu.n_nodes_spatial, 
        n_samples_edge = n_samples_edge_low, 
        tile_size = 1, 
        max_size = 1
    )

    set_membership             = kernels[0]
    spatial_random_walk        = kernels[1]
    attr_sampler               = kernels[2]
    attr_source_postprocess    = kernels[3]
    edge_sampler               = kernels[4]
    init_edge_counts           = kernels[5]
    edge_postprocess           = kernels[6]
    check_edge_distribution    = kernels[7]
    load_counts                = kernels[8]
    edge_histogram             = kernels[9]
    copy_reduced_counts        = kernels[10]
    init_attr                  = kernels[11]
    init_node_weights          = kernels[12]
    init_community             = kernels[13]
    node_weight                = kernels[14]
    attr_weight                = kernels[15]
    community_weight_gpu       = kernels[16]
    attr_postprocess_gpu       = kernels[17]

    community_to_merge = 0
    new_community = 0

    membership_pairs = test_class_gpu._calculate_members_gpu()

    edge_samples, walks_spatial, walks_edges_spatial, \
        walks_attr, walks_attr_cp, \
            source_attr_counts, attr_weights, node_weights = test_class_gpu._randomwalk_gpu(testing=True)

    nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(test_class_gpu.n_unique_edges, n_samples_edge_low), TPB = 8)

    community_weight_d = np.zeros((test_class_gpu.n_nodes_spatial,test_class_gpu.n_communities), dtype=np.float32)
    community_weight_d = cuda.to_device(community_weight_d)

    community_weight_gpu[blockspergrid,threadsperblock](
        community_to_merge,
        new_community,
        test_class_gpu.n_samples_edge, 
        test_class_gpu.walk_length, 
        test_class_gpu.n_communities,
        walks_attr,
        walks_attr_cp, 
        membership_pairs, 
        community_weight_d
    )

    nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(test_class_gpu.n_unique_edges, n_samples_edge_low), TPB = 8)

    cum_weight_d = np.zeros((test_class_gpu.n_nodes_spatial,test_class_gpu.n_attr), dtype=np.float32)
    cum_weight_d = cuda.to_device(cum_weight_d)

    attr_postprocess_gpu[threadsperblock,blockspergrid](
        community_to_merge,
        new_community,
        test_class_gpu.n_attr,
        attr_weights,
        community_weight_d,
        node_weights,
        membership_pairs,
        cum_weight_d,
    )

    walks_attr_cp = walks_attr_cp.copy_to_host()
    walks_attr = walks_attr.copy_to_host()
    membership_pairs = membership_pairs.copy_to_host()
    attr_weights = attr_weights.copy_to_host()
    node_weights = node_weights.copy_to_host()

    bfs_estimate_d = cum_weight_d.copy_to_host()

    community_weight_h = community_weight_cpu(
        ci = community_to_merge,
        cj = new_community,
        samples = walks_attr,
        samples_weights = walks_attr_cp,
        membership = membership_pairs,
        n_communities = test_class_gpu.n_communities
    )

    cum_weight_h = attr_postprocess_cpu(
        ci = community_to_merge,
        cj = new_community,
        attr_weight = attr_weights,
        community_weight = community_weight_h,
        node_weight = node_weights,
        membership = membership_pairs,
        n_nodes = test_class_gpu.n_nodes_spatial,
        n_attr = test_class_gpu.n_attr
    )

    np.testing.assert_array_equal(cum_weight_h, cum_weight_d.copy_to_host())

    test_class_cpu = StochasticCommunity(
        MG, seed=seed,
        use_gpu=False,
        n_samples_edge=n_samples_edge_low,
        walk_length=walk_length,
        num_streams=num_streams
    )
    test_class_cpu.init_random(tile_size=tile_size)

    membership_pairs = test_class_cpu._calculate_members_cpu()

    edge_samples, walks_spatial, walks_edges_spatial, \
        walks_attr, walks_attr_cp, \
            source_attr_counts, attr_weights, node_weights = test_class_cpu._randomwalk_cpu(testing=True)

    community_weight_h = community_weight_cpu(
        ci = community_to_merge,
        cj = new_community,
        samples = walks_attr,
        samples_weights = walks_attr_cp,
        membership = membership_pairs,
        n_communities = test_class_gpu.n_communities
    )

    cum_weight_h = attr_postprocess_cpu(
        ci = community_to_merge,
        cj = new_community,
        attr_weight = attr_weights,
        community_weight = community_weight_h,
        node_weight = node_weights,
        membership = membership_pairs,
        n_nodes = test_class_gpu.n_nodes_spatial,
        n_attr = test_class_gpu.n_attr
    )

    bfs_estimate_h = cum_weight_h

    np.allclose(bfs_estimate_h, bfs_estimate_d, rtol=1e-1)