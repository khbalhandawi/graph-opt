from doctest import debug
import pytest
import os
from ..community_detection.alliance_builder import AllianceBuilder
from ..community_detection.utilities import resort_communities, same_community

import numpy as np
import networkx as nx
import json
import os

from typing import Tuple

os.environ['NUMBA_ENABLE_CUDASIM'] = "1"

seed = 1234

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
        'VT': 5000
    }

    test_weights_individual_dict = {
        'AT': 8789,
        'BT': 6000,
        'CT': 6000,
        'HT': 5000,
        'VT': 12000
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
                'VT': 5000
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
                'CT': 14000,
                'HT': 0,
                'VT': 7000 # 5000 + 2000
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
        {
            "weights": (0.5,0.5),
            "test_communities_dict": {
                'AT': 0,
                'BT': 1,
                'CT': 0,
                'HT': 1,
                'VT': 0
            },
            "n_samples": 5
        },
    ],
    ids=[
        'minimize HHI',
        'maximize HHI',
        'maximize MAI',
        'test sampling',
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

    @pytest.fixture(scope="class")
    def init_class_gpu_iteration(self, example_case) -> AllianceBuilder:
        # This setup will run once for the entire class

        MG, _, _, _, _, _ = example_case

        test_class = AllianceBuilder(MG, seed=seed, use_gpu=True)
        membership_pairs, hhi_pairs, market_shares_pairs, weights_pairs, score_pairs = test_class.greedy_optimize(max_iterations=1,debug=True)

        return test_class, membership_pairs, hhi_pairs, market_shares_pairs, weights_pairs, score_pairs

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

        test_class, membership_pairs, hhi_pairs, market_shares_pairs, weights_pairs, score_pairs = init_class_gpu_iteration

        _, \
        community_to_merge, new_community, \
        merge_communities_test, test_HHI, \
        test_weights_dict = example_merge

        for key,value in zip(test_class.node_list,membership_pairs[community_to_merge,new_community]):
            assert value == merge_communities_test[key]

        for key,value in zip(test_class.unique_edges,hhi_pairs[community_to_merge,new_community]):
            assert test_HHI[key] == value

        for key,value in zip(test_class.node_list,weights_pairs[community_to_merge,new_community]):
            assert test_weights_dict[key] == value


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
