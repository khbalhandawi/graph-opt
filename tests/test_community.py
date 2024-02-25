import pytest
from ..community_detection.alliance_builder import AllianceBuilder

import numpy as np
import networkx as nx
import json
import os

seed = 1234

@pytest.fixture
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

@pytest.fixture(
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
                'CT': 7000,
                'HT': 0,
                'VT': 7000 # 5000 + 2000
            }
        },
        # case of a merger where there is a change in all scores
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

@pytest.fixture(
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
                'CT': 4,
                'HT': 4,
                'VT': 4
            },
            "n_samples": None
        },
        {
            "weights": (0.0,1.0),
            "test_communities_dict": {
                'AT': 4,
                'BT': 4,
                'CT': 4,
                'HT': 4,
                'VT': 4
            },
            "n_samples": None
        },
        {
            "weights": (0.5,5.0),
            "test_communities_dict": {
                'AT': 2,
                'BT': 3,
                'CT': 2,
                'HT': 3,
                'VT': 2
            },
            "n_samples": 2
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


@pytest.mark.dependency()
def test_initialization(example_case):

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

# @pytest.mark.dependency(depends=["test_initialization"])
def test_merge(example_merge):

    MG, \
    community_to_merge, new_community, \
    merge_communities_test, test_HHI, \
    test_weights_dict = example_merge

    test_class = AllianceBuilder(MG)

    test_class.update_scores(community_to_merge,new_community)

    # check all keys in test dictionary are present
    assert all(key in test_HHI.keys() for key in test_class.HHI)

    for key,value in test_class.communities_dict.items():
        assert value == merge_communities_test[key]

    for key,value in test_class.HHI.items():
        assert test_HHI[key] == value

    for key,value in test_class.weights_dict.items():
        assert value == test_weights_dict[key]

def test_optimize(optimization_cases):

    MG, weights, n_samples, test_communities_dict = optimization_cases

    test_class = AllianceBuilder(MG,seed=seed)

    test_class.greedy_optimize(weights=weights,n_samples=n_samples)

    assert test_class.communities_dict == test_communities_dict

    print(test_class.f_history)
    print(test_class.communities_history)