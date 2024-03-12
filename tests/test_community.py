from cgi import test
from doctest import debug
from optparse import Option
import pytest
import os

from sklearn.covariance import log_likelihood
from ..community_detection.alliance_builder import AllianceBuilder
from ..community_detection.utilities import resort_communities, same_community
# from ..utilities_cpu import searchsorted, cumsum

import numpy as np
import networkx as nx
import json
import os

from typing import Optional, Tuple, Dict

os.environ['NUMBA_ENABLE_CUDASIM'] = "0"

seed = 1234

def get_conditional_prob(test_class: AllianceBuilder):

    cond_prob_node_edge = np.zeros((test_class.n_unique_edges,test_class.n_nodes))
    for i, (u,v) in enumerate(test_class.edge_unique_list):

        for j,node in enumerate(test_class.node_list):
            cond_prob_node_edge[i,j] = test_class._cond_probs_nodes[(u,v)].get(node,0.0)/100

    return cond_prob_node_edge

def get_likelihood_prob(test_class: AllianceBuilder):

    likelihood_edge_node = np.zeros((test_class.n_nodes,test_class.n_unique_edges))
    for i,node in enumerate(test_class.node_list):
        node_weight = test_class.node_key2w[node]

        for u,v,key in test_class.edge_node2keypair_list[node]:
            j = test_class.edge_pairkey2ui[(u,v,key)]
            edge_weight = test_class.edge_pairkey2atrr[(u,v,key)]["weight"]
            likelihood_edge_node[i,j] += edge_weight / node_weight

    return likelihood_edge_node

def get_prior(test_class: AllianceBuilder):

    prior_node = np.zeros(test_class.n_nodes)
    node_weights = list(test_class.node_key2w.values())
    for i,node in enumerate(test_class.node_list):
        node_weight = test_class.node_key2w[node]
        prior_node[i] = node_weight / sum(node_weights)

    return prior_node

def get_marginal(test_class: AllianceBuilder):

    marginal_edge = np.zeros(test_class.n_unique_edges)
    edge_weights = list(test_class.edge_pair2w.values())
    for i,edge in enumerate(test_class.edge_unique_list):
        edge_weight = test_class.edge_pair2w[edge]
        marginal_edge[i] = edge_weight / sum(edge_weights)

    return marginal_edge

def get_source_prob(test_class:AllianceBuilder):

    source_p = np.array(list(test_class.node_spatial_key2w.values()),dtype=np.float64)
    total = source_p.sum()
    source_p /= total

    return source_p

def get_edge_prob(test_class: AllianceBuilder):

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

def get_node_bfs_probabilities(
        test_class: AllianceBuilder,
        source_u: str,
        target_node: str,
        max_depth:Optional[int] = 3,
        current_depth:Optional[int]=1,
        current_prob:Optional[int]=1,
        probabilities:Optional[Dict[str,float]]=None,
        previous_edge:Optional[Tuple[str,str, int]]=None
    ):

    if probabilities is None:
        probabilities = [0,] * max_depth

    if current_depth > max_depth:
        return probabilities

    edges = test_class.edge_u2pairkey_list[source_u]
    total_weight = sum(test_class.edge_pairkey2atrr[edge]['weight'] for edge in edges)

    for u, v, key in edges:
        curr_edge = (u, v, key)
        edge_prob = test_class.edge_pair2w[(u,v)] / total_weight
        edge_type_prob = test_class.edge_pairkey2atrr[curr_edge]['market_share_percent'] / 100.0
        edge_type = test_class.edge_pairkey2atrr[curr_edge]['edge_type']

        previous_edge_type = test_class.edge_pairkey2atrr[previous_edge]['edge_type'] if previous_edge is not None else None

        if edge_type == target_node and (previous_edge_type is None or previous_edge_type == target_node):
            new_prob = current_prob * edge_prob * edge_type_prob
        else:
            new_prob = 0.0

        probabilities[current_depth - 1] += new_prob

        get_node_bfs_probabilities(
            test_class,
            v,
            target_node,
            max_depth,
            current_depth + 1,
            new_prob,
            probabilities,
            curr_edge
        )

    return probabilities

def get_bfs_probabilities(
        test_class: AllianceBuilder,
        max_depth:Optional[int] = 3,
    ):

    node_bfs_p = np.zeros((test_class.n_nodes_spatial,test_class.n_nodes,max_depth))

    for i, source_node in enumerate(test_class.node_spatial_list):
        for j, target_node in enumerate(test_class.node_list):

            p = get_node_bfs_probabilities(
                test_class,
                source_node,
                target_node,
                max_depth
            )

            node_bfs_p[i,j,:] = p

    return node_bfs_p

def get_estimate_edge_prob(test_class: AllianceBuilder, walks_edges_spatial: np.ndarray):

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
        test_class: AllianceBuilder,
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
        test_class: AllianceBuilder,
        n_samples_edge: int,
        walk_length: int,
        walks_nodes: np.ndarray,
        walks_nodes_spatial: np.ndarray,
    ):

    # get sampled edge probabilities (marginal)
    bfs = np.zeros((test_class.n_nodes_spatial,test_class.n_nodes,walk_length-1))
    bfs_compound = np.zeros((test_class.n_nodes_spatial,test_class.n_nodes,walk_length-1))
    counts_bfs = np.zeros((test_class.n_nodes_spatial,test_class.n_nodes,walk_length-1))
    counts_spatial = np.zeros((test_class.n_nodes_spatial,walk_length-1))

    for i_sample in range(test_class.n_nodes_spatial):
        for j_sample in range(n_samples_edge):
            sample_walk_spatial_nodes = walks_nodes_spatial[i_sample, j_sample, :]
            sample_walk_nodes = walks_nodes[i_sample, j_sample, :]

            i = sample_walk_spatial_nodes[0]
            j_target = sample_walk_nodes[0] # sample node for the chain
            for k in range(walk_length-1):
                j = sample_walk_nodes[k]
                if j in test_class.node_i2key.keys():
                    counts_spatial[i,k] += 1
                    if j==j_target:
                        counts_bfs[i,j,k] += 1
                    else:
                        break
                    
    for i in range(test_class.n_nodes_spatial):
        for k in range(walk_length-1):

            bfs[i,:,k] = counts_bfs[i,:,k] / counts_spatial[i,k].sum() if counts_spatial[i,k].sum() > 0 else 0.0

    bfs_compound[:,:,0] = bfs[:,:,0]
    for k in range(1,walk_length-1):
        bfs_compound[:,:,k] = bfs[:,:,k] * bfs[:,:,k-1]

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

    bfs_source_nodal_probabilities_k1 = np.array(
        [
            [0.4677737 , 0.        , 0.10644526, 0.17031242, 0.25546863],
            [0.        , 1.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.8       , 0.2       , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 1.        ]
        ]
    )

    bfs_source_nodal_probabilities_k2 = np.array(
        [
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.25546863]
        ]
    )

    bfs_source_nodal_probabilities_k3 = np.array(
        [
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.        , 0.        , 0.        , 0.        , 0.        ]
        ]
    )

    bfs_source_nodal_probabilities = np.dstack(
        (
            bfs_source_nodal_probabilities_k1,
            bfs_source_nodal_probabilities_k2,
            bfs_source_nodal_probabilities_k3
        )
    )

    return posterior, likelihood, prior, marginal, source, edge_source_cond, bfs_source_nodal_probabilities

@pytest.fixture(scope="class")
def example_case():

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
    data_dir = os.path.join('..','test_data')
    MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph
    with open(os.path.join(script_dir,data_dir,'test_airline_nodes.json'),'r') as file:
        nodes_airlines_dict = json.load(file)

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

    communities_test = {
        'AT': 0,
        'BT': 1,
        'CT': 2,
        'HT': 3,
        'VT': 4
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
                ('0', '1'): 5200.0,
                ('0', '2'): 10000.0,
                ('0', '4'): 10000.0,
                ('1', '2'): 10000.0,
                ('3', '4'): 6800.0,
                ('4', '0'): 10000.0
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
                ('0', '1'): 5200.0,
                ('0', '2'): 10000.0,
                ('0', '4'): 10000.0,
                ('1', '2'): 10000.0,
                ('3', '4'): 6800.0,
                ('4', '0'): 10000.0
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
                ('0', '1'): 10000.0,
                ('0', '2'): 10000.0,
                ('0', '4'): 10000.0,
                ('1', '2'): 10000.0,
                ('3', '4'): 6800.0,
                ('4', '0'): 10000.0
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
           test_weights_dict

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
            "n_samples": None
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
            "n_samples": None
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
            "n_samples": None
        },
        # {
        #     "weights": (0.5,0.5),
        #     "test_communities_dict": {
        #         'AT': 0,
        #         'BT': 1,
        #         'CT': 0,
        #         'HT': 1,
        #         'VT': 0
        #     },
        #     "n_samples": 5
        # },
    ],
    ids=[
        'minimize HHI',
        'maximize HHI',
        'maximize MAI',
        # 'test sampling',
    ]
)
def optimization_cases(request, example_case):
    MG, nodes_airlines_dict, \
    test_HHI, test_weights_dict, \
    test_weights_individual_dict, \
    init_communities_test = example_case

    param = request.param
    weights = param["weights"]
    n_samples = param["n_samples"]
    test_communities_dict = param["test_communities_dict"]

    return MG, weights, n_samples, test_communities_dict

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
    key, reorder_dict, unique_list, n_communities = resort_communities(input_dict)
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
    numba_sim = os.environ.get("NUMBA_ENABLE_CUDASIM")
    os.environ['NUMBA_ENABLE_CUDASIM'] = "1"

    from ..utilities_cuda import cumsum_d, searchsorted_d
    
    cumsum = np.zeros_like(test_vector)

    cumsum_d(test_vector,cumsum)
    chosen = searchsorted_d(cumsum,threshold)

    assert chosen == expected

    os.environ['NUMBA_ENABLE_CUDASIM'] = numba_sim # reset environment to default
    
@pytest.mark.parametrize("test_vector, expected", [
    (np.array([0, 0, 0, 40, 60]), np.array([0,0,0,0.4,0.6])),
    (np.array([0, 40, 0, 0, 60]), np.array([0,0.4,0,0.0,0.6])),
], ids=['case 1', 'case 2'])
def test_sampler(test_vector, expected):
    numba_sim = os.environ.get("NUMBA_ENABLE_CUDASIM")
    os.environ['NUMBA_ENABLE_CUDASIM'] = "1"

    from ..utilities_cuda import cumsum_d, searchsorted_d
    import random
    random.seed(seed)
    n_samples = 50
    
    cumsum = np.zeros_like(test_vector)
    counts = np.zeros_like(test_vector)

    cumsum_d(test_vector,cumsum)
    
    for _ in range(n_samples):
        threshold = random.random()
        chosen = searchsorted_d(cumsum,threshold)
        counts[chosen] += 1

    p = counts / counts.sum()

    assert np.allclose(p,expected,rtol=1e-1)

    os.environ['NUMBA_ENABLE_CUDASIM'] = numba_sim # reset environment to default

class TestAllianceBuilder:

    @pytest.mark.dependency()
    def test_initialization(self,example_case):

        MG, nodes_airlines_dict, \
        test_HHI, test_weights_dict, \
        test_weights_individual_dict, \
        init_communities_test = example_case

        test_class = AllianceBuilder(MG)

        # check all keys in test dictionary are present
        assert all(key in test_HHI.keys() for key in test_class.HHI)

        for key,value in test_class.HHI.items():
            assert test_HHI[key] == value

        assert all(key in test_weights_dict.keys() for key in test_class.weights_dict)

        for key,value in test_class.weights_dict.items():
            assert value == test_weights_dict[key]

        for key,value in test_class.weights_individual_dict.items():
            assert value == test_weights_individual_dict[key]

        for key,value in test_class.communities_dict.items():
            assert value == init_communities_test[key]

    @pytest.fixture(scope="function")
    def init_class(self, example_case) -> AllianceBuilder:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = AllianceBuilder(MG, seed=seed)

        return test_class

    @pytest.fixture(scope="function")
    def init_class_gpu(self, example_case) -> AllianceBuilder:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = AllianceBuilder(MG, seed=seed, use_gpu=True)

        return test_class

    @pytest.fixture(scope="class")
    def init_class_gpu_iteration(self, example_case) -> AllianceBuilder:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = AllianceBuilder(MG, seed=seed, use_gpu=True)
        membership_pairs, hhi_pairs, weights_pairs, score_pairs = test_class.greedy_optimize(
            max_iterations=1,
            num_streams=test_class.n_communities,
            batch_size=1,
            debug=True
        )

        return test_class, membership_pairs, hhi_pairs, weights_pairs, score_pairs

    def test_probabilities(self, init_class: AllianceBuilder, example_case, example_conditionals):
        MG, _, _, _, _, _ = example_case
        test_posterior, test_likelihood, test_prior, test_marginal, \
            test_source, test_edge_source_cond, test_bfs = example_conditionals

        test_class = init_class

        posterior = get_conditional_prob(test_class)
        likelihood = get_likelihood_prob(test_class)

        prior = get_prior(test_class)
        marginal = get_marginal(test_class)

        edge_p, edge_source_cond_p = get_edge_prob(test_class)
        source_p = get_source_prob(test_class)

        bfs = get_bfs_probabilities(test_class,3)

        assert np.allclose(test_posterior, posterior, 1e-3)
        assert np.allclose(test_likelihood, likelihood, 1e-3)
        assert np.allclose(test_prior, prior, 1e-3)
        assert np.allclose(test_marginal, marginal, 1e-3)
        assert np.allclose(test_edge_source_cond, edge_source_cond_p, 1e-3)
        assert np.allclose(test_marginal, edge_p, 1e-3)
        assert np.allclose(test_source, source_p, 1e-3)
        assert np.allclose(test_bfs, bfs, 1e-3)

    def test_merge(self, init_class: AllianceBuilder, example_merge):

        test_class = init_class

        _, \
        community_to_merge, new_community, \
        merge_communities_test, test_HHI, \
        test_weights_dict = example_merge

        test_class.update_scores(community_to_merge,new_community)

        # check all keys in test dictionary are present
        assert all(key in test_HHI.keys() for key in test_class.HHI)

        for key,value in test_class.communities_dict.items():
            assert value == merge_communities_test[key]

        for key,value in test_class.HHI.items():
            assert test_HHI[key] == value

        for key,value in test_class.weights_dict.items():
            assert value == test_weights_dict[key]

    def test_optimize(self, init_class: AllianceBuilder, optimization_cases):

        MG, weights, n_samples, test_communities_dict = optimization_cases

        test_class = init_class

        test_class.greedy_optimize(weights=weights,n_samples=n_samples)

        assert same_community(test_class.communities_dict, test_communities_dict)

        print(test_class.f_history)
        print(test_class.communities_history)

    def test_merge_gpu(
            self,
            init_class_gpu_iteration: Tuple[AllianceBuilder,np.ndarray,np.ndarray,np.ndarray,np.ndarray],
            example_merge
        ):

        test_class, membership_pairs, hhi_pairs, weights_pairs, score_pairs = init_class_gpu_iteration

        _, \
        community_to_merge, new_community, \
        merge_communities_test, test_HHI, \
        test_weights_dict = example_merge

        for key,value in zip(test_class.node_list,membership_pairs[community_to_merge,new_community]):
            assert value == merge_communities_test[key]

        for key,value in zip(test_class.edge_unique_list,hhi_pairs[community_to_merge,new_community]):
            assert test_HHI[key] == value

        for key,value in zip(test_class.node_list,weights_pairs[community_to_merge,new_community]):
            assert test_weights_dict[key] == value

    def test_optimize_gpu(self, init_class_gpu: AllianceBuilder, optimization_cases):

        _, weights, n_samples, test_communities_dict = optimization_cases

        test_class = init_class_gpu

        test_class.greedy_optimize(weights=weights)

        assert same_community(test_class.communities_dict, test_communities_dict)

        print(test_class.f_history)
        print(test_class.communities_history)

    def test_randomwalker(self, example_case, init_class_gpu: AllianceBuilder):

        MG, _, _, _, _, _ = example_case
        test_class = init_class_gpu

        n_samples_edge = 5000
        n_samples_source_node = 50
        walk_length = 10
        TPB = (8,8,8)

        walks_spatial_weighted_h, walks_edges_spatial_weighted_h = test_class.weighted_sampling(
                n_samples_edge=n_samples_edge,
                n_samples_source_node=n_samples_source_node,
                walk_length=walk_length,
                TPB = (8,8),
                testing=True
            )

        edge_p, cond_edge_p = get_edge_prob(test_class)
        estimate_edge_p, estimate_cond_edge_p = get_estimate_edge_weighted_prob(
            test_class,
            n_samples_edge=n_samples_edge,
            n_samples_source_node=n_samples_source_node,
            walks_nodes_spatial=walks_spatial_weighted_h,
            walks_edges_spatial=walks_edges_spatial_weighted_h
        )

        edge_samples_h, walks_spatial_h, walks_edges_spatial_h, \
            walks_nodes_h, walks_nodes_weights_h, \
            edge_counts_h, cum_weight_h = test_class.greedy_optimize_stochastic(
                n_samples_edge=n_samples_edge,
                walk_length=walk_length,
                threads_per_block = 8,
                testing=True
            )

        bfs = get_bfs_probabilities(test_class,walk_length-1)
        estimate_bfs = get_estimate_node_bfs_probabilities(
            test_class,
            n_samples_edge=n_samples_edge,
            walk_length=walk_length,
            walks_nodes=walks_nodes_h,
            walks_nodes_spatial=walks_spatial_h
        )
        
        assert np.allclose(test_class._adj3/100,edge_counts_h,rtol=1e-1)
        assert np.allclose(cond_edge_p,estimate_cond_edge_p,rtol=1e-1)
        assert np.allclose(edge_p,estimate_edge_p,rtol=1e-1)
        assert np.allclose(bfs,estimate_bfs,rtol=1e-1)

if __name__ == "__main__":
    MG, nodes_airlines_dict, \
    test_HHI, test_weights_dict, \
    test_weights_individual_dict, \
    init_communities_test = example_case

    param = {
        "weights": (1.0,0.0),
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

    test_class = AllianceBuilder(MG,seed=seed, use_gpu=True)

    test_class.greedy_optimize(weights=weights,n_samples=n_samples)
