from cgi import test
from collections import defaultdict
from doctest import debug
from ipaddress import v4_int_to_packed
from optparse import Option
from igraph import Graph
import pytest
import os

from sklearn.covariance import log_likelihood

from ..community_detection.alliance_builder import GraphLoader, NeighbourSampler, ExactEvaluator, SampleEvaluator, GreedyOptimizer
from ..community_detection.utilities import same_community
from ..community_detection.utilities_cuda import define_cuda_kernels_sampling, define_cuda_kernels_deterministic, define_cuda_kernels_stochastic
from ..utilities import cuda_blocks

import numpy as np
import networkx as nx
import json
import os
import math
import itertools
from scipy.special import kl_div
from numba import cuda

from typing import Optional, Tuple, Dict, List

seed = 1234
tolerance = 1e-6


def get_attr_bfs_probabilities(
        test_class: NeighbourSampler,
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
        test_class: NeighbourSampler,
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
                ('0', '1'): 0.52,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
            },
            "w_ik": {
                0 : {0: 8789,   1: 0,       2: 2000,    3: 3200,    4: 4800},
                1 : {0: 0,      1: 6000,    2: 0,       3: 0,       4: 0},
                2 : {0: 0,      1: 0,       2: 0,       3: 0,       4: 0},
                3 : {0: 0,      1: 0,       2: 4800,    3: 1200,    4: 0},
                4 : {0: 0,      1: 0,       2: 0,       3: 0,       4: 7000},
            },
            "n_communities": 5,
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
                ('0', '1'): 0.52,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
            },
            "w_ik": {
                0 : {0: 0, 1: 8789, 2: 2000, 3: 3200, 4: 4800},
                1 : {0: 0, 1: 6000, 2: 0, 3: 0, 4: 0},
                2 : {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
                3 : {0: 0, 1: 0, 2: 4800, 3: 1200, 4: 0},
                4 : {0: 0, 1: 0, 2: 0, 3: 0, 4: 7000},
            },
            "n_communities": None,
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
                ('0', '1'): 0.52,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
            },
            "w_ik": {
                0 : {0: 8789,   1: 0,       2: 0,       3: 3200,    4: 6800},
                1 : {0: 0,      1: 6000,    2: 0,       3: 0,       4: 0},
                2 : {0: 0,      1: 0,       2: 0,       3: 0,       4: 0},
                3 : {0: 0,      1: 0,       2: 0,       3: 1200,    4: 4800},
                4 : {0: 0,      1: 0,       2: 0,       3: 0,       4: 7000},
            },
            "n_communities": None,
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
                ('0', '1'): 1.0,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
            },
            "w_ik": {
                0 : {0: 8789,   1: 0,       2: 2000,    3: 0,       4: 8000},
                1 : {0: 0,      1: 6000,    2: 0,       3: 0,       4: 0},
                2 : {0: 0,      1: 0,       2: 0,       3: 0,       4: 0},
                3 : {0: 0,      1: 0,       2: 4800,    3: 0,       4: 1200},
                4 : {0: 0,      1: 0,       2: 0,       3: 0,       4: 7000},
            },
            "n_communities": None,
        },
        {
            "merge_communities_test": {
                'AT': 0,
                'BT': 1,
                'CT': 2,
                'HT': 3,
                'VT': 3
            },
            "test_HHI": {
                ('0', '1'): 1.0,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
            },
            "w_ik": {
                0 : {0: 8789,   1: 0,       2: 2000,    3: 8000},
                1 : {0: 0,      1: 6000,    2: 0,       3: 0},
                2 : {0: 0,      1: 0,       2: 0,       3: 0},
                3 : {0: 0,      1: 0,       2: 4800,    3: 1200},
                4 : {0: 0,      1: 0,       2: 0,       3: 7000},
            },
            "n_communities": 4,
        },
    ],
    ids=[
        'default init',
        'nondefault init 1',
        'nondefault init 2',
        'nondefault init 3',
        'n_communities < n_attr'
    ]
)
def example_case_param(request):

    w_it = {
        0 : {'AT': 8789,    'BT': 0,        'CT': 2000,     'HT': 3200, 'VT': 4800},
        1 : {'AT': 0,       'BT': 6000,     'CT': 0,        'HT': 0,    'VT': 0},
        2 : {'AT': 0,       'BT': 0,        'CT': 0,        'HT': 0,    'VT': 0},
        3 : {'AT': 0,       'BT': 0,        'CT': 4800,     'HT': 1200, 'VT': 0},
        4 : {'AT': 0,       'BT': 0,        'CT': 0,        'HT': 0,    'VT': 7000},
    }

    w_i = {
        0 : 18_789,
        1 : 6000,
        2 : 0.0,
        3 : 6000,
        4 : 7000
    }

    p_it = defaultdict(dict)
    for key, value in w_it.items():
        for attr, weight in value.items():
            if w_i[key] > 0:
                p_it[key][attr] = weight / w_i[key]
            else:
                p_it[key][attr] = 0.0

    param = request.param
    communities_test = param["merge_communities_test"]
    test_HHI = param["test_HHI"]
    w_ik = param["w_ik"]
    n_communities = param["n_communities"]

    return communities_test, n_communities, test_HHI, w_it, w_i, w_ik, p_it

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
def optimization_cases(request):

    param = request.param
    weights = param["weights"]
    test_communities_dict = param["test_communities_dict"]

    return weights, test_communities_dict

@pytest.mark.parametrize("dict1, dict2, expected", [
    ({'AT': 0, 'BT': 1, 'CT': 0, 'HT': 1, 'VT': 0}, {'AT': 1, 'BT': 0, 'CT': 1, 'HT': 0, 'VT': 1}, True),
    ({'AT': 0, 'BT': 1, 'CT': 0, 'HT': 1, 'VT': 0}, {'AT': 2, 'BT': 0, 'CT': 2, 'HT': 0, 'VT': 2}, True),
    ({'AT': 0, 'BT': 1, 'CT': 0, 'HT': 1, 'VT': 0}, {'AT': 2, 'BT': 1, 'CT': 2, 'HT': 1, 'VT': 2}, True),
    ({'AT': 0, 'BT': 1, 'CT': 0, 'HT': 1, 'VT': 0}, {'AT': 1, 'BT': 2, 'CT': 1, 'HT': 0, 'VT': 1}, False),
], ids=['case 1', 'case 2', 'case 3', 'case 4'])
def test_same_community(dict1, dict2, expected):
    assert same_community(dict1, dict2) == expected

def test_cuda_blocks():

    dims = (500, 500, 500)
    TPB = (8, 8, 8)
    dims = (5000, 50)
    TPB = (32, 32)

    nstates, threadsperblock, blockspergrid = cuda_blocks(dims=dims, TPB=TPB)

    return

# @pytest.mark.parametrize("test_vector, threshold, expected", [
#     (np.array([0, 0, 0, 40, 60]), 0.3, 3),
#     (np.array([0, 40, 0, 0, 60]), 0.3, 1),
#     (np.array([0, 0, 0, 40, 60]), 0.4, 4),
#     (np.array([0, 40, 0, 0, 60]), 0.5, 4),
# ], ids=['case 1', 'case 2', 'case 3', 'case 4'])
# def test_sampler_deterministic(test_vector, threshold, expected):

#     from ..utilities_cuda import cumsum_h, searchsorted_h

#     cumsum = np.zeros_like(test_vector)

#     cumsum_h(test_vector,cumsum)
#     chosen = searchsorted_h(cumsum,threshold)

#     assert chosen == expected

# @pytest.mark.parametrize("test_vector, expected", [
#     (np.array([0, 0, 0, 40, 60]), np.array([0,0,0,0.4,0.6])),
#     (np.array([0, 40, 0, 0, 60]), np.array([0,0.4,0,0.0,0.6])),
# ], ids=['case 1', 'case 2'])
# def test_sampler(test_vector, expected):

#     from ..utilities_cuda import cumsum_h, searchsorted_h
#     import random
#     random.seed(seed)
#     n_samples = 50

#     cumsum = np.zeros_like(test_vector)
#     counts = np.zeros_like(test_vector)

#     cumsum_h(test_vector,cumsum)

#     for _ in range(n_samples):
#         threshold = random.random()
#         chosen = searchsorted_h(cumsum,threshold)
#         counts[chosen] += 1

#     p = counts / counts.sum()

#     assert np.allclose(p,expected,rtol=1e-1)

# @pytest.mark.parametrize("reduced_counts_base, reduced_counts", [
#     (
#         np.array(
#             [
#                 [  0,   0,   0, 191, 309],
#                 [500,   0,   0,   0,   0],
#                 [  0,   0, 500,   0,   0],
#                 [  0, 500,   0,   0,   0],
#                 [  0,   0, 399, 101,   0],
#                 [  0,   0,   0,   0, 500]
#             ], dtype = np.int32
#         ),
#         np.array(
#             [
#                 [  0,   0,   0, 191, 309],
#                 [500,   0,   0,   0,   0],
#                 [  0,   0, 500,   0,   0],
#                 [  0, 500,   0,   0,   0],
#                 [  0,   0,   0, 399, 101],
#                 [  0,   0,   0,   0, 500]
#             ], dtype = np.int32
#         )
#     ),
#     (
#         np.array(
#             [
#                 [  0,   0,   0, 191, 309],
#                 [500,   0,   0,   0,   0],
#                 [  0,   0, 500,   0,   0],
#                 [  0, 500,   0,   0,   0],
#                 [  0,   0, 399, 101,   0],
#                 [  0,   0,   0,   0, 500]
#             ], dtype = np.int32
#         ),
#         np.array(
#             [
#                 [  0,   0,   0, 191, 309],
#                 [500,   0,   0,   0,   0],
#                 [  0,   0, 500,   0,   0],
#                 [  0, 500,   0,   0,   0],
#                 [  0,   0,   0, 500,   0],
#                 [  0,   0,   0,   0, 500]
#             ], dtype = np.int32
#         )
#     )
# ], ids=['no difference', 'difference'])
# def test_hist_diff(reduced_counts_base, reduced_counts):

#     kernels = define_cuda_kernels_stochastic(1, 1, 500, 1, 1)

#     set_membership             = kernels[0]
#     spatial_random_walk        = kernels[1]
#     attr_sampler               = kernels[2]
#     attr_source_postprocess    = kernels[3]
#     edge_sampler               = kernels[4]
#     init_edge_counts           = kernels[5]
#     edge_postprocess           = kernels[6]
#     check_edge_distribution    = kernels[7]
#     load_counts                = kernels[8]
#     edge_histogram             = kernels[9]
#     copy_reduced_counts        = kernels[10]
#     init_attr                  = kernels[11]
#     init_node_weights          = kernels[12]
#     init_community             = kernels[13]
#     node_weight                = kernels[14]
#     attr_weight                = kernels[15]
#     community_weight           = kernels[16]
#     attr_postprocess           = kernels[17]

#     reduced_counts_base_d = cuda.to_device(reduced_counts_base)
#     reduced_counts_d = cuda.to_device(reduced_counts)
#     distributions_changed_d = cuda.device_array(reduced_counts.shape[0], dtype=np.int32)

#     reduced_counts_base_h = reduced_counts_base
#     reduced_counts_h = reduced_counts
#     # Assuming your matrices are named matrix1 and matrix2
#     hhi_new = np.sort(reduced_counts_h, axis=1)
#     hhi_base = np.sort(reduced_counts_base_h, axis=1)
#     distributions_changed_test = np.any(hhi_new != hhi_base, axis=1)

#     nstates, threadsperblock, blockspergrid = cuda_blocks(dims=(reduced_counts.shape[0], ), TPB = 8)

#     check_edge_distribution[threadsperblock,blockspergrid](
#         reduced_counts_base_d,
#         reduced_counts_d,
#         distributions_changed_d
#     )

#     cuda.synchronize()

#     distributions_changed = distributions_changed_d.copy_to_host()
#     distributions_changed = distributions_changed > 0

#     assert np.all(distributions_changed_test == distributions_changed)

class _TestDeterministic():
    num_streams = 4
    tile_size = 2
    threads_per_block = 8
    use_gpu = False

    @pytest.fixture(scope="function")
    def read_graph(self) -> GraphLoader:
        # This setup will run once for the entire class

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
        data_dir = os.path.join('..','test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph

        graph = GraphLoader(
            MG = MG,
            seed = seed,
            use_gpu = self.use_gpu
        )

        return graph

    @pytest.mark.dependency()
    def test_evaluator(self, read_graph: GraphLoader, example_case_param):

        graph = read_graph
        init_communities_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

        test_class = ExactEvaluator(
            gloader=graph, 
            seed=seed, 
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        test_class.update_scores(communities_dict=init_communities_test)
        
        # check all keys in test dictionary are present
        assert all(key in test_HHI.keys() for key in test_class.hhi)

        for key,value in test_class.hhi.items():
            assert abs(test_HHI[key] - value) < tolerance

        for node_id,(node,attrs) in enumerate(test_w_it.items()):
            for attr in attrs.keys():
                attr_id = test_class._gloader.attr_key2i[attr]
                assert abs(test_w_it[node][attr] - test_class.w_it[node_id][attr_id]) < tolerance

        for node_id,(node,ks) in enumerate(test_w_ik.items()):
            for k in ks.keys():
                assert abs(test_w_ik[node][k] - test_class.w_ik[node_id][k]) < tolerance

        for node_id,(node,value) in enumerate(test_w_i.items()):
            assert abs(test_w_i[node] - test_class.w_i[node_id]) < tolerance

        assert same_community(init_communities_test,test_class.communities_dict)

    def test_save_load(self, read_graph: GraphLoader, example_case_param, tmp_path):

        graph = read_graph
        init_communities_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

        test_class = ExactEvaluator(
            gloader=graph, 
            seed=seed, 
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        directory = tmp_path / "test_save"
        directory.mkdir()

        test_class.update_scores(communities_dict=init_communities_test)

        test_class.save(directory=directory)

        data_to_load = {
            'communities_dict':     {"name" : '_communities_dict', "dtype" : dict},
            'log_mpc':              {"name" : '_log_mpc', "dtype" : dict},
            'mpc':                  {"name" : '_mpc', "dtype" : dict},
            'hhi':                  {"name" : '_hhi', "dtype" : Dict[Tuple[str,str],int]},
        }

        test_data = {}
        for name, attr in data_to_load.items():
            test_data[name] = getattr(test_class,attr["name"])

        del test_class

        test_class = ExactEvaluator(
            gloader=graph, 
            seed=seed, 
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )
        
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


class TestDeterministicCPU(_TestDeterministic):
    use_gpu = False


class TestDeterministicGPU(_TestDeterministic):
    use_gpu = True


class _TestStochastic():

    n_samples_edge=5000
    walk_length=1
    threads_per_block=8
    n_walks = 10000
    use_gpu = False
    tolerance = 1e-1

    @pytest.fixture(scope="function")
    def read_graph(self) -> Tuple[GraphLoader, NeighbourSampler]:
        # This setup will run once for the entire class

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
        data_dir = os.path.join('..','test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph

        graph = GraphLoader(
            MG = MG,
            seed = seed,
            use_gpu = self.use_gpu
        )

        sampler = NeighbourSampler(
            gloader=graph,
            seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge,
            n_walks=self.n_walks,
            walk_length=self.walk_length,
        )

        sampler.execute_randomwalk()

        return graph, sampler

    @pytest.mark.dependency()
    def test_evaluator(self, read_graph: GraphLoader, example_case_param):

        graph, sampler = read_graph
        init_communities_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

        test_p_ik = defaultdict(dict)
        for node_id,(node,ks) in enumerate(test_w_ik.items()):
            for k in ks.keys():

                if test_w_i[node_id] > 0:
                    test_p_ik[node_id][k] = test_w_ik[node_id][k] / test_w_i[node_id]
                else:
                    test_p_ik[node_id][k] = 0.0

        test_class = SampleEvaluator(
            gloader=graph,
            sampler=sampler,
            seed=seed,
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        test_class.update_scores(communities_dict=init_communities_test)
        
        # check all keys in test dictionary are present
        assert all(key in test_HHI.keys() for key in test_class.hhi)

        for key,value in test_class.hhi.items():
            assert abs(test_HHI[key] - value) < self.tolerance

        for node_id,(node,attrs) in enumerate(test_w_it.items()):
            for attr in attrs.keys():
                attr_id = test_class._gloader.attr_key2i[attr]

                if test_w_i[node_id] > 0:
                    assert abs(test_p_it[node][attr] - test_class.w_it[node_id][attr_id] / test_class.w_i[node_id]) < self.tolerance
                else:
                    assert test_class.w_i[node_id] == 0

        for node_id,(node,ks) in enumerate(test_w_ik.items()):
            for k in ks.keys():
                
                if test_w_i[node_id] > 0:
                    assert abs(test_p_ik[node].get(k,0) - test_class.w_ik[node_id][k] / test_class.w_i[node_id]) < self.tolerance
                else:
                    assert test_class.w_i[node_id] == 0

        assert same_community(init_communities_test,test_class.communities_dict)

    def test_save_load(self, read_graph: GraphLoader, example_case_param, tmp_path):

        graph, sampler = read_graph
        init_communities_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

        test_class = SampleEvaluator(
            gloader=graph,
            sampler=sampler,
            seed=seed, 
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        directory = tmp_path / "test_save"
        directory.mkdir()

        test_class.update_scores(communities_dict=init_communities_test)

        test_class.save(directory=directory)

        data_to_load = {
            'communities_dict':     {"name" : '_communities_dict', "dtype" : dict},
            'log_mpc':              {"name" : '_log_mpc', "dtype" : dict},
            'mpc':                  {"name" : '_mpc', "dtype" : dict},
            'hhi':                  {"name" : '_hhi', "dtype" : Dict[Tuple[str,str],int]},
        }

        test_data = {}
        for name, attr in data_to_load.items():
            test_data[name] = getattr(test_class,attr["name"])

        del test_class

        test_class = SampleEvaluator(
            gloader=graph,
            sampler=sampler,
            seed=seed, 
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )
        
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


class TestStochasticCPU(_TestStochastic):

    use_gpu = False


class TestStochasticGPU(_TestStochastic):

    use_gpu = True


class TestOptimize():

    epsilon = 0.5
    rtol = 0.0
    n_samples_edge=500
    walk_length=1
    threads_per_block=8
    n_walks = 1000
    use_gpu = False

    @pytest.fixture(scope="function")
    def read_graph(self) -> Tuple[GraphLoader, NeighbourSampler]:
        # This setup will run once for the entire class

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
        data_dir = os.path.join('..','test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph

        graph = GraphLoader(
            MG = MG,
            seed = seed,
            use_gpu = self.use_gpu
        )

        sampler = NeighbourSampler(
            gloader=graph,
            seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge,
            n_walks=self.n_walks,
            walk_length=self.walk_length,
        )

        sampler.execute_randomwalk()

        return graph, sampler

    @pytest.mark.parametrize("n_sample_pairs", [100, 8], ids=['no sampling', '8 samples'])
    def test_optimize(self, read_graph: Tuple[GraphLoader, NeighbourSampler], optimization_cases, n_sample_pairs):

        graph, sampler = read_graph
        weights, test_communities_dict = optimization_cases

        evaluator = SampleEvaluator(
            gloader=graph,
            sampler=sampler,
            seed=seed,
            use_gpu=self.use_gpu,
            n_communities=None, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        optimizer = GreedyOptimizer(
            evaluator=evaluator,
            seed=seed,
            num_streams=1,
            weights=weights,
            use_gpu=self.use_gpu
        )

        optimizer.greedy_optimize(
            max_iterations = 1000,
            epsilon = self.epsilon,
            rtol = self.rtol,
            n_sample_pairs = n_sample_pairs
        )

        assert same_community(optimizer.communities_dict, test_communities_dict)




# @pytest.mark.dependency()
# def test_sampler(self,example_case_param):

#     MG, nodes_airlines_dict, \
#     test_HHI, test_weights_dict, \
#     test_weights_individual_dict, \
#     init_communities_test = example_case_param

#     test_class = ExactEvaluator(MG, communities_dict=init_communities_test)
#     n_omit_samples = int(test_class.n_communities*(test_class.n_communities+1)/2)
#     n_test_samples = test_class.n_communities**2 - n_omit_samples

#     assert test_class.n_communities**2 - len(test_class._sampled_communities_pairs) == n_test_samples

#     while samples:=test_class._sample_community_pairs(3):
#         assert len(samples) <= n_test_samples

# @pytest.mark.dependency()
# def test_sampler_dist(self,example_case_param):

#     n_samples = 1000

#     MG, nodes_airlines_dict, \
#     test_HHI, test_weights_dict, \
#     test_weights_individual_dict, \
#     init_communities_test = example_case_param

#     test_class = ExactEvaluator(MG, communities_dict=init_communities_test)

#     all_samples = list(itertools.product(range(test_class.n_communities), range(test_class.n_communities)))
#     all_samples_dict = {pair : i for i,pair in enumerate(all_samples)}
#     diag_mask = np.array([True if i==j else False for i,j in all_samples])

#     probabilities = test_class.bfs_degree_community[all_samples] / np.sum(test_class.bfs_degree_community)
#     probabilities_pairs = probabilities[:,0] * probabilities[:,1]
#     probabilities_pairs[diag_mask] = 0.0
#     probabilities_pairs = probabilities_pairs / np.sum(probabilities_pairs)


#     samples = test_class._sample_community_pairs(n_samples,replace=True)
#     samples_indices = [all_samples_dict[pair] for pair in samples]

#     # Compute the empirical PMF
#     unique, counts = np.unique(samples_indices, return_counts=True)
#     counts_full = np.zeros_like(probabilities_pairs)
#     counts_full[unique] = counts
#     empirical_pmf = counts_full / len(samples_indices)

#     kl_divergence = kl_div(empirical_pmf + 1e-10, probabilities_pairs + 1e-10)

#     assert np.mean(empirical_pmf - probabilities_pairs) <= 1e-3

#     return