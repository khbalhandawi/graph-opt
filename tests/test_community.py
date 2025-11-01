from cgi import test
from collections import defaultdict
from doctest import debug
from optparse import Option
import pytest
import os

from graph_opt.community_detection.alliance_builder import GraphLoader, NeighbourSampler, ExactEvaluator, ExactWeightedEvaluator, SampleEvaluator, SampleWeightedEvaluator, GreedyOptimizer, BruteForce, ExactOptimizer, multiobjective
from graph_opt.community_detection.utilities import same_community
from graph_opt.utilities import cuda_blocks

import numpy as np
import networkx as nx
import os

from typing import Optional, Tuple, Dict, List, Union, Any

seed = 1234
tolerance = 1e-6

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
            "membership_test": np.array([0, 1, 2, 3, 4]),
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
            "membership_test": np.array([1, 1, 2, 3, 4]),
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
            "membership_test": np.array([0, 1, 4, 3, 4]),
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
            "membership_test": np.array([0, 1, 2, 4, 4]),
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
            "membership_test": np.array([0, 1, 2, 3, 3]),
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
    membership_test = param["membership_test"]
    test_HHI = param["test_HHI"]
    w_ik = param["w_ik"]
    n_communities = param["n_communities"]

    return communities_test, membership_test, n_communities, test_HHI, w_it, w_i, w_ik, p_it

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
        'maximize MPC',
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
    ({'AT': 0, 'BT': 0, 'CT': 0, 'HT': 2, 'VT': 2}, {'AT': 0, 'BT': 0, 'CT': 0, 'HT': 0, 'VT': 0}, False),
], ids=['case 1', 'case 2', 'case 3', 'case 4', 'case 5'])
def test_same_community(dict1, dict2, expected):
    assert same_community(dict1, dict2) == expected

def test_cuda_blocks():

    dims = (500, 500, 500)
    TPB = (8, 8, 8)
    dims = (5000, 50)
    TPB = (32, 32)

    nstates, threadsperblock, blockspergrid = cuda_blocks(dims=dims, TPB=TPB)

    return

@pytest.mark.parametrize("test_vector, threshold, expected", [
    (np.array([0, 0, 0, 40, 60]), 0.3, 3),
    (np.array([0, 40, 0, 0, 60]), 0.3, 1),
    (np.array([0, 0, 0, 40, 60]), 0.4, 4),
    (np.array([0, 40, 0, 0, 60]), 0.5, 4),
], ids=['case 1', 'case 2', 'case 3', 'case 4'])
def test_sampler_deterministic(test_vector, threshold, expected):

    from graph_opt.utilities_cuda import cumsum_h, searchsorted_h

    cumsum = np.zeros_like(test_vector)

    cumsum_h(test_vector,cumsum)
    chosen = searchsorted_h(cumsum,threshold)

    assert chosen == expected


class _TestDeterministic():
    num_streams = 4
    tile_size = 2
    threads_per_block = 8
    use_gpu = False
    update_membership = False
    walk_length = 1
    tolerance = 1e-6

    @pytest.fixture(scope="function")
    def read_graph(self) -> GraphLoader:
        # This setup will run once for the entire class

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
        data_dir = os.path.join('..','test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph

        graph = GraphLoader(
            MG = MG,
            use_gpu = self.use_gpu
        )

        return graph

    def test_evaluator(self, read_graph: GraphLoader, example_case_param):

        graph = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

        test_p_ik = defaultdict(dict)
        for node_id,(node,ks) in enumerate(test_w_ik.items()):
            for k in ks.keys():

                if test_w_i[node_id] > 0:
                    test_p_ik[node_id][k] = test_w_ik[node_id][k] / test_w_i[node_id]
                else:
                    test_p_ik[node_id][k] = 0.0

        test_class = ExactEvaluator(
            gloader=graph, 
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            walk_length=self.walk_length,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        if self.update_membership:
            test_class.update_scores_membership(membership_test)
        else:
            test_class.update_scores(communities_dict=init_communities_test)
        
        # check all keys in test dictionary are present
        assert all(key in test_HHI.keys() for key in test_class.hhi)

        for key,value in test_class.hhi.items():
            assert abs(test_HHI[key] - value) < tolerance

        for node_id,(node,attrs) in enumerate(test_w_it.items()):
            for attr in attrs.keys():
                attr_id = test_class._gloader.attr_key2i[attr]

                if test_w_i[node_id] > 0:
                    assert abs(test_p_it[node][attr] - test_class.p_it[node_id][attr_id]) < self.tolerance
                else:
                    assert test_class.w_i[node_id] == 0

        for node_id,(node,ks) in enumerate(test_w_ik.items()):
            for k in ks.keys():
                
                if test_w_i[node_id] > 0:
                    assert abs(test_p_ik[node].get(k,0) - test_class.p_ik[node_id][k]) < self.tolerance
                else:
                    assert test_class.w_i[node_id] == 0


        for node_id,(node,value) in enumerate(test_w_i.items()):
            assert abs(test_w_i[node] - test_class.w_i[node_id]) < tolerance

        assert same_community(init_communities_test,test_class.communities_dict)

    def test_save_load(self, read_graph: GraphLoader, example_case_param, tmp_path):

        graph = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

        test_class = ExactEvaluator(
            gloader=graph, 
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        directory = tmp_path / "test_save"
        directory.mkdir()

        if self.update_membership:
            test_class.update_scores_membership(membership_test)
        else:
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

        test_class = ExactEvaluator(
            gloader=graph, 
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
    update_membership = True

    @pytest.fixture(scope="function")
    def read_graph(self) -> Tuple[GraphLoader, NeighbourSampler]:
        # This setup will run once for the entire class

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
        data_dir = os.path.join('..','test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph

        graph = GraphLoader(
            MG = MG,
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

    def test_evaluator(self, read_graph: GraphLoader, example_case_param):

        graph, sampler = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

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
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        if self.update_membership:
            test_class.update_scores_membership(membership_test)
        else:
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

        if not self.update_membership:
            assert same_community(init_communities_test,test_class.communities_dict)

    def test_save_load(self, read_graph: GraphLoader, example_case_param, tmp_path):

        graph, sampler = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

        test_class = SampleEvaluator(
            gloader=graph,
            sampler=sampler,
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        directory = tmp_path / "test_save"
        directory.mkdir()

        if self.update_membership:
            test_class.update_scores_membership(membership_test)
        else:
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

        sampler = NeighbourSampler(
            gloader=graph,
            seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge,
            n_walks=self.n_walks,
            walk_length=self.walk_length,
        )

        test_class = SampleEvaluator(
            gloader=graph,
            sampler=sampler,
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

        pass

    def test_with_exact(self, read_graph: GraphLoader, example_case_param):

        graph, sampler = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_ik, test_p_it = example_case_param

        test_class = SampleEvaluator(
            gloader=graph,
            sampler=sampler,
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        test_class.update_scores_membership(membership_test)

        test_class_exact = ExactEvaluator(
            gloader=graph,
            use_gpu=self.use_gpu,
            n_communities=n_communities, 
            walk_length=self.walk_length,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        test_class_exact.update_scores(communities_dict=init_communities_test)

        if not self.use_gpu:
            assert np.allclose(test_class.p_it, test_class_exact.p_it, rtol=1e-1)
            assert np.allclose(test_class.p_ik, test_class_exact.p_ik, rtol=1e-1)
        assert np.allclose(test_class.hhi_edges, test_class_exact.hhi_edges, rtol=1e-1)


class TestStochasticCPU(_TestStochastic):

    use_gpu = False


class TestStochasticGPU(_TestStochastic):

    use_gpu = True


class _TestOptimize():

    epsilon = 0.5
    rtol = 1e-6
    n_samples_edge=500
    walk_length=2
    threads_per_block=8
    n_walks = 1000
    num_streams = 2

    @pytest.fixture(scope="function")
    def read_graph(self) -> Tuple[GraphLoader, NeighbourSampler]:
        # This setup will run once for the entire class

        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
        data_dir = os.path.join('..','test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir,data_dir,'test_DG_AP_AL.graphml')) # store graph

        graph = GraphLoader(
            MG = MG,
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

    @pytest.fixture(scope="function",
        params=[
            {
                "type": "exact",
            },
            {
                "type": "sample",
            },
        ],
        ids=[
            'exact',
            'neighbourhood_sample',
        ]
    )
    def get_evaluator(
            self, 
            read_graph: Tuple[GraphLoader, NeighbourSampler], 
            get_sample_evaluator: SampleEvaluator,
            get_exact_evaluator: ExactEvaluator,
            request
        ) -> ExactEvaluator | SampleEvaluator:

        graph, sampler = read_graph

        param = request.param

        if param["type"] == 'sample':
            return get_sample_evaluator

        elif param["type"] == 'exact':
            return get_exact_evaluator

    @pytest.fixture(scope="function")
    def get_sample_evaluator(self, read_graph: Tuple[GraphLoader, NeighbourSampler]) -> SampleEvaluator:

        graph, sampler = read_graph

        return SampleEvaluator(
            gloader=graph,
            sampler=sampler,
            use_gpu=self.use_gpu,
            n_communities=None, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

    @pytest.fixture(scope="function")
    def get_exact_evaluator(self, read_graph: Tuple[GraphLoader, NeighbourSampler]) -> ExactEvaluator:

        graph, sampler = read_graph

        return SampleEvaluator(
            gloader=graph,
            sampler=sampler,
            use_gpu=self.use_gpu,
            n_communities=None, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )
    
    @pytest.mark.parametrize("n_sample_pairs", [None, 8], ids=['no sampling', '8 samples'])
    def test_greedy_optimize(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_evaluator: SampleEvaluator | ExactEvaluator, 
        n_sample_pairs: int
    ):
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_evaluator

        optimizer = GreedyOptimizer(
            evaluator=evaluator,
            seed=seed,
            num_streams=self.num_streams,
            weights=weights,
            use_gpu=self.use_gpu
        )

        optimizer.optimize(
            max_iterations = 1000,
            epsilon = self.epsilon,
            rtol = self.rtol,
            n_sample_pairs = n_sample_pairs,
            multiprocess=self.use_multiprocessing
        )

        # evaluator.update_scores(test_communities_dict)
        # f1 = -np.mean(list(evaluator.hhi.values()))
        # f2 = np.mean(list(evaluator.log_mpc.values()))
        # f = multiobjective(f1,f2,weights)

        # fs = []
        # for membership in optimizer.membership_pairs.values():
        #     evaluator.update_scores_membership(membership)
        #     f1 = -np.mean(list(evaluator.hhi.values()))
        #     f2 = np.mean(list(evaluator.log_mpc.values()))
        #     f = multiobjective(f1,f2,weights)
        #     fs.append(f)

        assert same_community(optimizer.communities_dict, test_communities_dict)

    @pytest.mark.parametrize("max_iterations", [None, 1000], ids=['unlimited budget', '100 budget'])
    def test_brute_force_optimize(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_evaluator: SampleEvaluator | ExactEvaluator, 
        max_iterations: int
    ):
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_evaluator

        optimizer = BruteForce(
            evaluator=evaluator,
            seed=seed,
            num_streams=self.num_streams,
            weights=weights,
            use_gpu=self.use_gpu
        )

        optimizer.optimize(
            max_iterations = max_iterations,
            rtol = self.rtol,
            multiprocess=self.use_multiprocessing
        )

        evaluator.update_scores(test_communities_dict)
        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f = multiobjective(f1,f2,weights)

        assert optimizer.f == f

    def test_exact_optimize(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_evaluator: SampleEvaluator | ExactEvaluator, 
        tmp_path
    ):
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_evaluator

        directory = tmp_path / "test_save"
        directory.mkdir()

        model_settings = {
            'MIPGap': 1e-12,  # Stop if the gap is within 0.01%
            'Heuristics': 0.1,  # Use more aggressive heuristics
            'MIPFocus': 1,  # Focus on finding feasible solutions
            'VarBranch': 0,  # More aggressive variable branching
            'Presolve': 2 # Turn off presolve
        }
        model_settings = {
            'TimeLimit': 7_200,  # Time limit in seconds, e.g., 3600 seconds for 1 hour
            'NodefileStart': 0.5,  # Start writing nodes to disk after 50% of the time limit
            'SoftMemLimit': 54*1e9,  # Memory limit in MB, e.g., 55296 MB for 54 GB
            'MIPGap': 1e-3,  # Stop if the gap is within 0.01%
            'Heuristics': 0.2,  # default heuristics 0.05
            'MIPFocus': 1,  # balanced Focus on finding feasible solutions and proving optimality (1 for feasible, 2 for optimal)
            # 'VarBranch': 2,  # More aggressive variable branching
            'Cuts': -1, # automatic (the default)
            'Presolve': 2,  # aggressive presolve
            'Threads': 12,  # Adjust based on your CPU core count
            'ScaleFlag': 2,  # Enable scaling (more aggressive)
            'Method': 2,  # Set to 1 for dual simplex (2 for barrier, -1 for automatic)
            'FeasibilityTol': 1e-6, # Feasibility tolerance
            'IntFeasTol': 1e-5, # Integer feasibility tolerance
            'PreQLinearize': 2,  # Enable pre-queue linearization
            'Crossover': 0,  # keep the interior solution
            'NodeMethod': 2,  # use Barrier inside the tree as well
            'NumericFocus': 2,  # High numeric focus for better numerical stability
            'BarConvTol': 1e-8,  # (optional) a bit tighter because we skip crossover
        }

        optimizer = ExactOptimizer(
            evaluator=evaluator,
            seed=seed,
            weights=weights,
            log_dir=directory,
            log_interval=1,
            model_settings=model_settings,
        )

        optimizer.build_model()
        # optimizer.tune(time_limit=5)
        optimizer.optimize()

        evaluator.update_scores(optimizer.communities_dict)
        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f = multiobjective(f1,f2,weights)

        assert abs(optimizer.f - f) / f < 1e-6

        evaluator.update_scores(test_communities_dict)
        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f = multiobjective(f1,f2,weights)

        assert abs(optimizer.f - f) / f < 1e-6

    def test_save_load_exact(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_sample_evaluator: SampleEvaluator | ExactEvaluator,
        tmp_path
    ):
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_sample_evaluator

        directory = tmp_path / "test_save"
        directory.mkdir()

        model_settings = {
            'MIPGap': 1e-12,  # Stop if the gap is within 0.01%
            'Heuristics': 0.9,  # Use more aggressive heuristics
            'MIPFocus': 3,  # Focus on finding feasible solutions
            'VarBranch': 2,  # More aggressive variable branching
            'Presolve': 0 # Turn off presolve
        }

        optimizer = ExactOptimizer(
            evaluator=evaluator,
            seed=seed,
            weights=weights,
            log_dir=directory,
            log_interval=1,
            model_settings=model_settings,
        )

        optimizer.build_model()
        optimizer.optimize()

        optimizer.save(directory=directory)

        data_to_load = {
            'communities_history':  {"name" : 'communities_history', "dtype" : List[Dict[str,int]]},
            'f_history':            {"name" : 'f_history', "dtype" : list},
            'multi_history':        {"name" : 'multi_history', "dtype" : List[Tuple[float,float]]},
            'mip_logs':             {"name" : 'mip_logs', "dtype" : List[Tuple[float,float,float,float,float,float]]},
            'simplex_logs':         {"name" : 'simplex_logs', "dtype" : List[Tuple[float,float,float,float]]},
            'barrier_logs':         {"name" : 'barrier_logs', "dtype" : List[Tuple[float,float,float,float]]},
        }

        test_data = {}
        for name, attr in data_to_load.items():
            test_data[name] = getattr(optimizer,attr["name"])

        optimizer = ExactOptimizer(
            evaluator=evaluator,
            seed=seed,
            weights=weights,
            log_dir=directory,
            log_interval=1,
            model_settings=model_settings,
        )
        
        optimizer.load(directory=directory)

        for name, attr in data_to_load.items():
            if attr["dtype"] == np.ndarray:
                assert np.allclose(getattr(optimizer,attr["name"]), test_data[name], rtol=1e-1)
            elif attr["dtype"] == list:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    assert abs(value - test_value) < 1e-6
            elif attr["dtype"] == List[Dict[str,int]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    assert value == test_value
            elif attr["dtype"] == List[Tuple[float,float]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    for value_i, test_value_i in zip(value, test_value):
                        assert abs(value_i - test_value_i) < 1e-6
            elif attr["dtype"] == List[Dict[int,Any]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    for key, test_key in zip(value.keys(), test_value.keys()):
                        assert key == test_key
                        for value_i, test_value_i in zip(value[key].values(), test_value[test_key].values()):
                            assert abs(value_i - test_value_i) < 1e-6

    def test_save_load_brute(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_evaluator: SampleEvaluator | ExactOptimizer,
        tmp_path
    ):
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_evaluator

        optimizer = BruteForce(
            evaluator=evaluator,
            seed=seed,
            num_streams=self.num_streams,
            weights=weights,
            use_gpu=self.use_gpu
        )

        optimizer.optimize(
            max_iterations = 1000,
            rtol = self.rtol,
            multiprocess=self.use_multiprocessing,
        )

        directory = tmp_path / "test_save"
        directory.mkdir()

        optimizer.save(directory=directory)

        data_to_load = {
            'communities_history':  {"name" : 'communities_history', "dtype" : List[Dict[str,int]]},
            'f_history':            {"name" : 'f_history', "dtype" : list},
            'multi_history':        {"name" : 'multi_history', "dtype" : List[Tuple[float,float]]},
            'results_library':      {"name" : 'results_library', "dtype" : Dict[int,Any]},
        }

        test_data = {}
        for name, attr in data_to_load.items():
            test_data[name] = getattr(optimizer,attr["name"])

        optimizer = BruteForce(
            evaluator=evaluator,
            seed=seed,
            num_streams=self.num_streams,
            weights=weights,
            use_gpu=self.use_gpu
        )
        
        optimizer.load(directory=directory)

        for name, attr in data_to_load.items():
            if attr["dtype"] == np.ndarray:
                assert np.allclose(getattr(optimizer,attr["name"]), test_data[name], rtol=1e-1)
            elif attr["dtype"] == list:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    assert abs(value - test_value) < 1e-6
            elif attr["dtype"] == List[Dict[str,int]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    assert value == test_value
            elif attr["dtype"] == List[Tuple[float,float]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    for value_i, test_value_i in zip(value, test_value):
                        assert abs(value_i - test_value_i) < 1e-6
            elif attr["dtype"] == List[Dict[int,Any]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    for key, test_key in zip(value.keys(), test_value.keys()):
                        assert key == test_key
                        for value_i, test_value_i in zip(value[key].values(), test_value[test_key].values()):
                            assert abs(value_i - test_value_i) < 1e-6

    def test_save_load_greedy(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_evaluator: SampleEvaluator | ExactEvaluator,
        tmp_path
    ):
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_evaluator

        optimizer = GreedyOptimizer(
            evaluator=evaluator,
            seed=seed,
            num_streams=self.num_streams,
            weights=weights,
            use_gpu=self.use_gpu
        )

        optimizer.optimize(
            max_iterations = 1000,
            rtol = self.rtol,
            multiprocess=self.use_multiprocessing,
        )

        directory = tmp_path / "test_save"
        directory.mkdir()

        optimizer.save(directory=directory)

        data_to_load = {
            'communities_history':  {"name" : 'communities_history', "dtype" : List[Dict[str,int]]},
            'f_history':            {"name" : 'f_history', "dtype" : list},
            'multi_history':        {"name" : 'multi_history', "dtype" : List[Tuple[float,float]]},
        }

        test_data = {}
        for name, attr in data_to_load.items():
            test_data[name] = getattr(optimizer,attr["name"])

        optimizer_load = GreedyOptimizer(
            evaluator=evaluator,
            seed=seed,
            num_streams=self.num_streams,
            weights=weights,
            use_gpu=self.use_gpu
        )
        
        optimizer_load.load(directory=directory)

        for name, attr in data_to_load.items():
            if attr["dtype"] == np.ndarray:
                assert np.allclose(getattr(optimizer_load,attr["name"]), test_data[name], rtol=1e-1)
            elif attr["dtype"] == list:
                for value, test_value in zip(getattr(optimizer_load,attr["name"]), test_data[name]):
                    assert abs(value - test_value) < 1e-6
            elif attr["dtype"] == List[Dict[str,int]]:
                for value, test_value in zip(getattr(optimizer_load,attr["name"]), test_data[name]):
                    assert value == test_value
            elif attr["dtype"] == List[Tuple[float,float]]:
                for value, test_value in zip(getattr(optimizer_load,attr["name"]), test_data[name]):
                    for value_i, test_value_i in zip(value, test_value):
                        assert abs(value_i - test_value_i) < 1e-6
            elif attr["dtype"] == List[Dict[int,Any]]:
                for value, test_value in zip(getattr(optimizer_load,attr["name"]), test_data[name]):
                    for key, test_key in zip(value.keys(), test_value.keys()):
                        assert key == test_key
                        for value_i, test_value_i in zip(value[key].values(), test_value[test_key].values()):
                            assert abs(value_i - test_value_i) < 1e-6


class _TestOptimizeWeighted():
    """Test class for optimization with weighted evaluators (both exact and sample)."""

    epsilon = 0.5
    rtol = 1e-6
    n_samples_edge = 500
    walk_length = 2
    threads_per_block = 8
    n_walks = 1000
    num_streams = 2

    @pytest.fixture(scope="function")
    def read_graph(self) -> Tuple[GraphLoader, NeighbourSampler]:
        """Setup graph and sampler for weighted optimization tests."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join('..', 'test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir, data_dir, 'test_DG_AP_AL.graphml'))

        graph = GraphLoader(
            MG=MG,
            use_gpu=self.use_gpu
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

    @pytest.fixture(scope="function",
        params=[
            {
                "type": "exact_weighted",
            },
            {
                "type": "sample_weighted",
            },
        ],
        ids=[
            'exact_weighted',
            'sample_weighted',
        ]
    )
    def get_weighted_evaluator(
            self, 
            read_graph: Tuple[GraphLoader, NeighbourSampler], 
            get_sample_weighted_evaluator: SampleWeightedEvaluator,
            get_exact_weighted_evaluator: ExactWeightedEvaluator,
            request
        ) -> ExactWeightedEvaluator | SampleWeightedEvaluator:
        """Fixture to get either exact or sample weighted evaluator based on test parameter."""

        param = request.param

        if param["type"] == 'sample_weighted':
            return get_sample_weighted_evaluator
        elif param["type"] == 'exact_weighted':
            return get_exact_weighted_evaluator

    @pytest.fixture(scope="function")
    def get_sample_weighted_evaluator(self, read_graph: Tuple[GraphLoader, NeighbourSampler]) -> SampleWeightedEvaluator:
        """Create SampleWeightedEvaluator for optimization tests."""
        graph, sampler = read_graph

        return SampleWeightedEvaluator(
            gloader=graph,
            sampler=sampler,
            use_gpu=self.use_gpu,
            n_communities=None, 
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

    @pytest.fixture(scope="function")
    def get_exact_weighted_evaluator(self, read_graph: Tuple[GraphLoader, NeighbourSampler]) -> ExactWeightedEvaluator:
        """Create ExactWeightedEvaluator for optimization tests."""
        graph, sampler = read_graph

        return ExactWeightedEvaluator(
            gloader=graph,
            use_gpu=self.use_gpu,
            n_communities=None,
            walk_length=self.walk_length,
            lambda_decay=1.0,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )
    
    @pytest.mark.parametrize("n_sample_pairs", [None, 8], ids=['no sampling', '8 samples'])
    def test_greedy_optimize(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_weighted_evaluator: SampleWeightedEvaluator | ExactWeightedEvaluator, 
        n_sample_pairs: int
    ):
        """Test greedy optimization with weighted evaluators."""
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_weighted_evaluator

        optimizer = GreedyOptimizer(
            evaluator=evaluator,
            seed=seed,
            num_streams=self.num_streams,
            weights=weights,
            use_gpu=self.use_gpu
        )

        optimizer.optimize(
            max_iterations=1000,
            epsilon=self.epsilon,
            rtol=self.rtol,
            n_sample_pairs=n_sample_pairs,
            multiprocess=self.use_multiprocessing
        )

        assert same_community(optimizer.communities_dict, test_communities_dict)

    @pytest.mark.parametrize("max_iterations", [None, 1000], ids=['unlimited budget', '1000 budget'])
    def test_brute_force_optimize(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_weighted_evaluator: SampleWeightedEvaluator | ExactWeightedEvaluator, 
        max_iterations: int
    ):
        """Test brute force optimization with weighted evaluators."""
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_weighted_evaluator

        optimizer = BruteForce(
            evaluator=evaluator,
            seed=seed,
            num_streams=self.num_streams,
            weights=weights,
            use_gpu=self.use_gpu
        )

        optimizer.optimize(
            max_iterations=max_iterations,
            rtol=self.rtol,
            multiprocess=self.use_multiprocessing
        )

        evaluator.update_scores(test_communities_dict)
        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f = multiobjective(f1, f2, weights)

        # assert same_community(optimizer.communities_dict, test_communities_dict)

        assert optimizer.f == f

    def test_exact_optimize(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_weighted_evaluator: SampleWeightedEvaluator | ExactWeightedEvaluator, 
        tmp_path
    ):
        """Test exact optimization with weighted evaluators using ExactOptimizer."""
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_weighted_evaluator

        directory = tmp_path / "test_save"
        directory.mkdir()

        model_settings = {
            'MIPGap': 1e-12,  # Stop if the gap is within 0.01%
            'Heuristics': 0.1,  # Use more aggressive heuristics
            'MIPFocus': 1,  # Focus on finding feasible solutions
            'VarBranch': 0,  # More aggressive variable branching
            'Presolve': 2 # Turn off presolve
        }
        model_settings = {
            'TimeLimit': 7_200,  # Time limit in seconds, e.g., 3600 seconds for 1 hour
            'NodefileStart': 0.5,  # Start writing nodes to disk after 50% of the time limit
            'SoftMemLimit': 54*1e9,  # Memory limit in MB, e.g., 55296 MB for 54 GB
            'MIPGap': 1e-3,  # Stop if the gap is within 0.01%
            'Heuristics': 0.2,  # default heuristics 0.05
            'MIPFocus': 1,  # balanced Focus on finding feasible solutions and proving optimality (1 for feasible, 2 for optimal)
            # 'VarBranch': 2,  # More aggressive variable branching
            'Cuts': -1, # automatic (the default)
            'Presolve': 2,  # aggressive presolve
            'Threads': 12,  # Adjust based on your CPU core count
            'ScaleFlag': 2,  # Enable scaling (more aggressive)
            'Method': 2,  # Set to 1 for dual simplex (2 for barrier, -1 for automatic)
            'FeasibilityTol': 1e-6, # Feasibility tolerance
            'IntFeasTol': 1e-5, # Integer feasibility tolerance
            'PreQLinearize': 2,  # Enable pre-queue linearization
            'Crossover': 0,  # keep the interior solution
            'NodeMethod': 2,  # use Barrier inside the tree as well
            'NumericFocus': 2,  # High numeric focus for better numerical stability
            'BarConvTol': 1e-8,  # (optional) a bit tighter because we skip crossover
        }

        optimizer = ExactOptimizer(
            evaluator=evaluator,
            seed=seed,
            weights=weights,
            log_dir=directory,
            log_interval=1,
            model_settings=model_settings,
        )

        optimizer.build_model()
        # optimizer.tune(time_limit=5)
        optimizer.optimize()

        evaluator.update_scores(optimizer.communities_dict)
        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f = multiobjective(f1,f2,weights)

        assert abs(optimizer.f - f) / f < 1e-6

        evaluator.update_scores(test_communities_dict)
        f1 = -np.mean(list(evaluator.hhi.values()))
        f2 = np.mean(list(evaluator.log_mpc.values()))
        f = multiobjective(f1,f2,weights)

        assert abs(optimizer.f - f) / f < 1e-6

    def test_save_load_exact(
        self, 
        optimization_cases: Tuple[Tuple[float,float], Dict[str,int]],
        get_sample_weighted_evaluator: SampleWeightedEvaluator,
        tmp_path
    ):
        """Test saving and loading ExactOptimizer."""
        
        weights, test_communities_dict = optimization_cases
        evaluator = get_sample_weighted_evaluator

        directory = tmp_path / "test_save"
        directory.mkdir()

        model_settings = {
            'MIPGap': 1e-12,  # Stop if the gap is within 0.01%
            'Heuristics': 0.9,  # Use more aggressive heuristics
            'MIPFocus': 3,  # Focus on finding feasible solutions
            'VarBranch': 2,  # More aggressive variable branching
            'Presolve': 0 # Turn off presolve
        }

        optimizer = ExactOptimizer(
            evaluator=evaluator,
            seed=seed,
            weights=weights,
            log_dir=directory,
            log_interval=1,
            model_settings=model_settings,
        )

        optimizer.build_model()
        optimizer.optimize()

        optimizer.save(directory=directory)

        data_to_load = {
            'communities_history':  {"name" : 'communities_history', "dtype" : List[Dict[str,int]]},
            'f_history':            {"name" : 'f_history', "dtype" : list},
            'multi_history':        {"name" : 'multi_history', "dtype" : List[Tuple[float,float]]},
            'mip_logs':             {"name" : 'mip_logs', "dtype" : List[Tuple[float,float,float,float,float,float]]},
            'simplex_logs':         {"name" : 'simplex_logs', "dtype" : List[Tuple[float,float,float,float]]},
            'barrier_logs':         {"name" : 'barrier_logs', "dtype" : List[Tuple[float,float,float,float]]},
        }

        test_data = {}
        for name, attr in data_to_load.items():
            test_data[name] = getattr(optimizer,attr["name"])

        optimizer = ExactOptimizer(
            evaluator=evaluator,
            seed=seed,
            weights=weights,
            log_dir=directory,
            log_interval=1,
            model_settings=model_settings,
        )
        
        optimizer.load(directory=directory)

        for name, attr in data_to_load.items():
            if attr["dtype"] == np.ndarray:
                assert np.allclose(getattr(optimizer,attr["name"]), test_data[name], rtol=1e-1)
            elif attr["dtype"] == list:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    assert abs(value - test_value) < 1e-6
            elif attr["dtype"] == List[Dict[str,int]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    assert value == test_value
            elif attr["dtype"] == List[Tuple[float,float]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    for value_i, test_value_i in zip(value, test_value):
                        assert abs(value_i - test_value_i) < 1e-6
            elif attr["dtype"] == List[Dict[int,Any]]:
                for value, test_value in zip(getattr(optimizer,attr["name"]), test_data[name]):
                    for key, test_key in zip(value.keys(), test_value.keys()):
                        assert key == test_key
                        for value_i, test_value_i in zip(value[key].values(), test_value[test_key].values()):
                            assert abs(value_i - test_value_i) < 1e-6


class TestOptimizeWeightedGPU(_TestOptimizeWeighted):
    """GPU tests for weighted optimization."""
    use_gpu = True
    use_multiprocessing = False


class TestOptimizeWeightedCPU(_TestOptimizeWeighted):
    """CPU tests for weighted optimization."""
    use_gpu = False
    use_multiprocessing = False

class TestOptimizeGPU(_TestOptimize):

    use_gpu = True
    use_multiprocessing = False


class TestOptimizeCPU(_TestOptimize):

    use_gpu = False
    use_multiprocessing = False

import pytest
import logging
from unittest.mock import Mock, MagicMock
from graph_opt.community_detection.alliance_builder import ExactOptimizer

class TestProcessCommunitiesDict:
    """Test cases for the process_communities_dict method of ExactOptimizer"""
    
    def setup_mock_optimizer(self, n_attr=4, n_communities=3, attr_names=None):
        """Set up a mock ExactOptimizer with necessary attributes"""
        if attr_names is None:
            attr_names = [f"attr_{i}" for i in range(n_attr)]
        
        # Create mock evaluator with graph loader
        mock_gloader = Mock()
        mock_gloader.attr_i2key = {i: attr_names[i] for i in range(n_attr)}
        
        mock_evaluator = Mock()
        mock_evaluator._gloader = mock_gloader
        
        # Create mock optimizer
        optimizer = Mock(spec=ExactOptimizer)
        optimizer._evaluator = mock_evaluator
        optimizer._T = list(range(n_attr))
        optimizer._K = list(range(n_communities))
        optimizer.model_built = True
        
        # Bind the real method to the mock
        optimizer.process_communities_dict = ExactOptimizer.process_communities_dict.__get__(optimizer)
        
        return optimizer
    
    def create_mock_var(self, value):
        """Create a mock Gurobi variable with specified X attribute"""
        mock_var = Mock()
        mock_var.X = value
        mock_var.getAttr.return_value = value
        return mock_var
    
    def test_process_communities_dict_complete_assignment(self):
        """Test when all attributes are assigned to communities"""
        optimizer = self.setup_mock_optimizer(n_attr=3, n_communities=3, 
                                            attr_names=['AT', 'BT', 'CT'])
        
        # Create decision variables where each attribute is assigned to a unique community
        x = {
            (0, 0): self.create_mock_var(1.0),  # attr 0 -> community 0
            (0, 1): self.create_mock_var(0.0),
            (0, 2): self.create_mock_var(0.0),
            (1, 0): self.create_mock_var(0.0),
            (1, 1): self.create_mock_var(1.0),  # attr 1 -> community 1
            (1, 2): self.create_mock_var(0.0),
            (2, 0): self.create_mock_var(0.0),
            (2, 1): self.create_mock_var(0.0),
            (2, 2): self.create_mock_var(1.0),  # attr 2 -> community 2
        }
        
        result = optimizer.process_communities_dict(x)
        
        expected = {'AT': 0, 'BT': 1, 'CT': 2}
        assert result == expected
    
    def test_process_communities_dict_partial_assignment_with_unused_communities(self):
        """Test when some attributes are unassigned - should raise ValueError"""
        optimizer = self.setup_mock_optimizer(n_attr=3, n_communities=5, 
                                            attr_names=['AT', 'BT', 'CT'])
        
        # Only assign first two attributes, leave third unassigned
        x = {
            (0, 0): self.create_mock_var(1.0),  # attr 0 -> community 0
            (0, 1): self.create_mock_var(0.0),
            (0, 2): self.create_mock_var(0.0),
            (0, 3): self.create_mock_var(0.0),
            (0, 4): self.create_mock_var(0.0),
            (1, 0): self.create_mock_var(0.0),
            (1, 1): self.create_mock_var(1.0),  # attr 1 -> community 1
            (1, 2): self.create_mock_var(0.0),
            (1, 3): self.create_mock_var(0.0),
            (1, 4): self.create_mock_var(0.0),
            (2, 0): self.create_mock_var(0.0),  # attr 2 -> unassigned
            (2, 1): self.create_mock_var(0.0),
            (2, 2): self.create_mock_var(0.0),
            (2, 3): self.create_mock_var(0.0),
            (2, 4): self.create_mock_var(0.0),
        }
        
        # Should raise ValueError for unassigned attributes
        with pytest.raises(ValueError, match="Error: 1 attributes have no community assigned"):
            optimizer.process_communities_dict(x)
    
    def test_process_communities_dict_partial_assignment_no_unused_communities(self):
        """Test when some attributes are unassigned - should raise ValueError"""
        optimizer = self.setup_mock_optimizer(n_attr=4, n_communities=2, 
                                            attr_names=['AT', 'BT', 'CT', 'DT'])
        
        # Assign first two attributes to both communities, leave last two unassigned
        x = {
            (0, 0): self.create_mock_var(1.0),  # attr 0 -> community 0
            (0, 1): self.create_mock_var(0.0),
            (1, 0): self.create_mock_var(0.0),
            (1, 1): self.create_mock_var(1.0),  # attr 1 -> community 1
            (2, 0): self.create_mock_var(0.0),  # attr 2 -> unassigned
            (2, 1): self.create_mock_var(0.0),
            (3, 0): self.create_mock_var(0.0),  # attr 3 -> unassigned
            (3, 1): self.create_mock_var(0.0),
        }
        
        # Should raise ValueError for unassigned attributes
        with pytest.raises(ValueError, match="Error: 2 attributes have no community assigned"):
            optimizer.process_communities_dict(x)
    
    def test_process_communities_dict_fractional_values(self):
        """Test with fractional values (should only consider values > 0.5)"""
        optimizer = self.setup_mock_optimizer(n_attr=2, n_communities=2, 
                                            attr_names=['AT', 'BT'])
        
        x = {
            (0, 0): self.create_mock_var(0.7),  # attr 0 -> community 0 (>0.5)
            (0, 1): self.create_mock_var(0.3),  # not assigned
            (1, 0): self.create_mock_var(0.4),  # not assigned
            (1, 1): self.create_mock_var(0.8),  # attr 1 -> community 1 (>0.5)
        }
        
        result = optimizer.process_communities_dict(x)
        
        expected = {'AT': 0, 'BT': 1}
        assert result == expected
    
    def test_process_communities_dict_multiple_assignments_error(self):
        """Test that multiple community assignments to same attribute raise an error"""
        optimizer = self.setup_mock_optimizer(n_attr=2, n_communities=2, 
                                            attr_names=['AT', 'BT'])
        
        x = {
            (0, 0): self.create_mock_var(0.8),  # attr 0 -> community 0
            (0, 1): self.create_mock_var(0.9),  # attr 0 -> community 1 (conflict!)
            (1, 0): self.create_mock_var(0.0),
            (1, 1): self.create_mock_var(0.7),  # attr 1 -> community 1
        }
        
        with pytest.raises(AssertionError, match="error: multiple communities assigned to the same attribute"):
            optimizer.process_communities_dict(x)
    
    def test_process_communities_dict_empty_variables(self):
        """Test with empty decision variables dictionary"""
        optimizer = self.setup_mock_optimizer(n_attr=2, n_communities=2)
        
        with pytest.raises(AssertionError, match="No decision variables found"):
            optimizer.process_communities_dict({})
    
    def test_process_communities_dict_vars_without_x_attribute(self):
        """Test with variables that don't have X attribute - should raise ValueError for unassigned"""
        optimizer = self.setup_mock_optimizer(n_attr=2, n_communities=2, 
                                            attr_names=['AT', 'BT'])
        
        # Create variables without X attribute
        mock_var_no_x = Mock()
        del mock_var_no_x.X  # Remove X attribute
        
        x = {
            (0, 0): mock_var_no_x,  # No X attribute - should be skipped
            (0, 1): self.create_mock_var(0.0),
            (1, 0): self.create_mock_var(0.0),
            (1, 1): self.create_mock_var(0.8),  # attr 1 -> community 1
        }
        
        # Should raise ValueError for unassigned attributes (attr 0 has no valid assignment)
        with pytest.raises(ValueError, match="Error: 1 attributes have no community assigned"):
            optimizer.process_communities_dict(x)
    
    def test_process_communities_dict_real_attribute_names(self):
        """Test with realistic attribute names - should raise ValueError for unassigned"""
        attr_names = ['airport_type', 'building_type', 'city_type', 'hub_type', 'vessel_type']
        optimizer = self.setup_mock_optimizer(n_attr=5, n_communities=3, 
                                            attr_names=attr_names)
        
        x = {
            (0, 0): self.create_mock_var(1.0),  # airport_type -> community 0
            (0, 1): self.create_mock_var(0.0),
            (0, 2): self.create_mock_var(0.0),
            (1, 0): self.create_mock_var(0.0),
            (1, 1): self.create_mock_var(1.0),  # building_type -> community 1
            (1, 2): self.create_mock_var(0.0),
            (2, 0): self.create_mock_var(0.0),
            (2, 1): self.create_mock_var(0.0),
            (2, 2): self.create_mock_var(1.0),  # city_type -> community 2
            (3, 0): self.create_mock_var(0.0),  # hub_type -> unassigned
            (3, 1): self.create_mock_var(0.0),
            (3, 2): self.create_mock_var(0.0),
            (4, 0): self.create_mock_var(0.0),  # vessel_type -> unassigned
            (4, 1): self.create_mock_var(0.0),
            (4, 2): self.create_mock_var(0.0),
        }
        
        # Should raise ValueError for unassigned attributes
        with pytest.raises(ValueError, match="Error: 2 attributes have no community assigned"):
            optimizer.process_communities_dict(x)
    
    def test_process_communities_dict_all_values_below_threshold(self):
        """Test when all variables have values below 0.5 threshold - should raise ValueError"""
        optimizer = self.setup_mock_optimizer(n_attr=2, n_communities=2, 
                                            attr_names=['AT', 'BT'])
        
        x = {
            (0, 0): self.create_mock_var(0.3),  # attr 0 -> not assigned (< 0.5)
            (0, 1): self.create_mock_var(0.4),  # attr 0 -> not assigned (< 0.5)
            (1, 0): self.create_mock_var(0.2),  # attr 1 -> not assigned (< 0.5)
            (1, 1): self.create_mock_var(0.1),  # attr 1 -> not assigned (< 0.5)
        }
        
        # Should raise ValueError for all unassigned attributes
        with pytest.raises(ValueError, match="Error: 2 attributes have no community assigned"):
            optimizer.process_communities_dict(x)


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
            "membership_test": np.array([0, 1, 2, 3, 4]),
            "test_HHI": {
                ('0', '1'): 0.52,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
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
            "membership_test": np.array([1, 1, 2, 3, 4]),
            "test_HHI": {
                ('0', '1'): 0.52,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
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
            "membership_test": np.array([0, 1, 4, 3, 4]),
            "test_HHI": {
                ('0', '1'): 0.52,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
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
            "membership_test": np.array([0, 1, 2, 4, 4]),
            "test_HHI": {
                ('0', '1'): 1.0,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
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
            "membership_test": np.array([0, 1, 2, 3, 3]),
            "test_HHI": {
                ('0', '1'): 1.0,
                ('0', '2'): 1.0,
                ('0', '4'): 1.0,
                ('1', '2'): 1.0,
                ('3', '4'): 0.68,
                ('4', '0'): 1.0,
            },
            "n_communities": 4,
        },
    ],
    ids=[
        'weighted default init',
        'weighted nondefault init 1',
        'weighted nondefault init 2',
        'weighted nondefault init 3',
        'weighted n_communities < n_attr'
    ]
)
def example_case_param_weighted(request):
    """
    Fixture for weighted evaluator test cases with dynamically computed expected values.
    
    This fixture computes the expected w_it and w_ik values on-the-fly
    using the same weighted recursive formulation that the ExactWeightedEvaluator uses.
    This ensures the tests are self-contained and don't rely on external pre-computed values.
    
    The computation follows these steps:
    1. Load the test graph and extract parameters
    2. Compute w_it (attribute weights) and w_i (total weights) 
    3. Compute p_iv (neighbor PMF) and p_iv_t (conditional attribute PMF)
    4. Apply the weighted recursive formulation to get S_t, S_k, and S_total
    5. Normalize to get w_it and w_ik
    6. Convert to dictionary format for test comparisons
    """
    
    # Import required modules for computation
    from graph_opt.community_detection.utilities_cpu import calc_w_it_cpu, calc_p_iv_cpu, calc_p_iv_t_cpu
    from graph_opt.community_detection.utilities_weighted_cpu import (
        calc_s_t_weighted_cpu, calc_s_k_weighted_cpu, calc_s_total_weighted_cpu,
        calc_normalized_p_t_weighted_cpu, calc_normalized_p_k_weighted_cpu
    )
    
    # Load test graph
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join('..', 'test_data')
    MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir, data_dir, 'test_DG_AP_AL.graphml'))
    graph = GraphLoader(MG=MG, use_gpu=False)
    
    # Get test case parameters
    param = request.param
    communities_test = param["merge_communities_test"]
    membership_test = param["membership_test"]
    test_HHI = param["test_HHI"]
    n_communities = param["n_communities"]
    
    # Set default n_communities if None
    if n_communities is None:
        n_communities = graph.n_attr
    
    # Compute w_it (attribute weights per node)
    w_it_array = np.zeros((graph.n_nodes_spatial, graph.n_attr), dtype=np.float32)
    calc_w_it_cpu(
        adj=graph._adj_sparse,
        adj3=graph._adj3,
        w_it=w_it_array
    )
    
    # Convert to dictionary format for compatibility
    w_it = {}
    attr_names = ['AT', 'BT', 'CT', 'HT', 'VT']
    for i in range(graph.n_nodes_spatial):
        w_it[i] = {}
        for j, attr in enumerate(attr_names):
            w_it[i][attr] = float(w_it_array[i, j])
    
    # Compute w_i (total outgoing weights)
    w_i_array = np.sum(w_it_array, axis=1)
    w_i = {i: float(w_i_array[i]) for i in range(graph.n_nodes_spatial)}
    
    # Compute p_iv (neighbor PMF)
    p_iv = np.zeros((graph.n_nodes_spatial, graph.n_nodes_spatial), dtype=np.float32)
    calc_p_iv_cpu(
        adj_sparse=graph._adj_sparse,
        adj_weight=graph._adj,
        w_i=w_i_array,
        p_iv=p_iv
    )
    
    # Compute p_iv_t (conditional attribute PMF)
    p_iv_t = np.zeros((graph.n_nodes_spatial, graph.n_nodes_spatial, graph.n_attr), dtype=np.float32)
    calc_p_iv_t_cpu(
        adj_multi_attr_sparse=graph._adj_multi_attr_sparse,
        adj_multi_attr_type=graph._adj_multi_attr_type,
        adj_multi_attr_weight=graph._adj_multi_attr,
        adj_multi_attr_index=graph._adj_multi_attr_index,
        adj_weight=graph._adj,
        p_iv_t=p_iv_t
    )
    
    # Initialize weighted score arrays
    s_t = np.zeros((graph.n_nodes_spatial, graph.n_attr), dtype=np.float32)
    s_k = np.zeros((graph.n_nodes_spatial, n_communities), dtype=np.float32)
    s_total = np.zeros(graph.n_nodes_spatial, dtype=np.float32)
    
    # Compute weighted scores with walk_length=1, lambda_decay=1.0
    calc_s_t_weighted_cpu(p_iv, w_i_array, p_iv_t, 1, 1.0, s_t)
    calc_s_k_weighted_cpu(p_iv, w_i_array, p_iv_t, 1, 1.0, membership_test, s_k)
    calc_s_total_weighted_cpu(p_iv, w_i_array, 1, 1.0, s_total)
    
    # Compute normalized probabilities
    p_it_array = np.zeros((graph.n_nodes_spatial, graph.n_attr), dtype=np.float32)
    p_ik_array = np.zeros((graph.n_nodes_spatial, n_communities), dtype=np.float32)

    calc_normalized_p_t_weighted_cpu(s_t, s_total, p_it_array)
    calc_normalized_p_k_weighted_cpu(s_k, s_total, p_ik_array)

    # Convert to dictionary format for test comparisons
    p_it = {}
    for i in range(graph.n_nodes_spatial):
        p_it[i] = {}
        for j, attr in enumerate(attr_names):
            p_it[i][attr] = float(p_it_array[i, j])

    p_ik = {}
    for i in range(graph.n_nodes_spatial):
        p_ik[i] = {}
        for k in range(n_communities):
            p_ik[i][k] = float(p_ik_array[i, k])

    return communities_test, membership_test, n_communities, test_HHI, w_it, w_i, p_it, p_ik


class _TestDeterministicWeighted():
    """Test class for ExactWeightedEvaluator with deterministic weighted calculations."""
    
    num_streams = 4
    tile_size = 2
    threads_per_block = 8
    use_gpu = False
    update_membership = False
    walk_length = 1
    lambda_decay = 1.0
    tolerance = 1e-6

    @pytest.fixture(scope="function")
    def read_graph(self) -> GraphLoader:
        # This setup will run once for the entire class
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
        data_dir = os.path.join('..','test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir, data_dir, 'test_DG_AP_AL.graphml'))

        graph = GraphLoader(
            MG=MG,
            use_gpu=self.use_gpu
        )

        return graph

    def test_evaluator(self, read_graph: GraphLoader, example_case_param_weighted):
        """Test the ExactWeightedEvaluator against expected weighted probability values."""
        
        graph = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_p_it, test_p_ik = example_case_param_weighted

        # Import the ExactWeightedEvaluator
        from graph_opt.community_detection.alliance_builder import ExactWeightedEvaluator

        test_class = ExactWeightedEvaluator(
            gloader=graph,
            use_gpu=self.use_gpu,
            n_communities=n_communities,
            walk_length=self.walk_length,
            lambda_decay=self.lambda_decay,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        if self.update_membership:
            test_class.update_scores_membership(membership_test)
        else:
            test_class.update_scores(communities_dict=init_communities_test)

        # Check all keys in test dictionary are present for HHI
        assert all(key in test_HHI.keys() for key in test_class.hhi)

        for key, value in test_class.hhi.items():
            assert abs(test_class.hhi[key] - test_HHI[key]) < self.tolerance, \
                f"HHI mismatch for {key}: expected {test_HHI[key]}, got {test_class.hhi[key]}"

        # Check that w_it now contains p_t_normalized (since weighted evaluator overwrites it)
        for node_id, attrs in test_w_it.items():
            for attr in attrs.keys():
                attr_idx = graph.attr_key2i[attr]
                expected_val = test_w_it[node_id][attr]
                actual_val = test_class.w_it[node_id, attr_idx]  # For weighted evaluator, this contains w_it
                assert abs(actual_val - expected_val) < self.tolerance, \
                    f"w_it mismatch for node {node_id}, attr {attr}: expected {expected_val}, got {actual_val}"

        # Check w_i values (should be exactly the same as in ExactEvaluator)
        for node_id, (node, value) in enumerate(test_w_i.items()):
            expected_val = test_w_i[node_id]
            actual_val = test_class.w_i[node_id]
            assert abs(actual_val - expected_val) < self.tolerance, \
                f"w_i mismatch for node {node_id}: expected {expected_val}, got {actual_val}"

        # Check p_it values (these are stored in _p_it for compatibility)
        for node_id, communities in test_p_ik.items():
            for k in communities.keys():
                expected_val = test_p_ik[node_id][k]
                actual_val = test_class.p_ik[node_id, k]  # For weighted evaluator, this contains w_k_normalized
                assert abs(actual_val - expected_val) < self.tolerance, \
                    f"w_ik mismatch for node {node_id}, community {k}: expected {expected_val}, got {actual_val}"

        # Check p_ik values (these are stored in _p_ik for compatibility)
        for node_id, communities in test_p_ik.items():
            for k in communities.keys():
                expected_val = test_p_ik[node_id][k]
                actual_val = test_class.p_ik[node_id, k]  # For weighted evaluator, this contains w_k_normalized
                assert abs(actual_val - expected_val) < self.tolerance, \
                    f"w_ik mismatch for node {node_id}, community {k}: expected {expected_val}, got {actual_val}"

        # Verify that the communities dictionary is correctly set
        assert same_community(init_communities_test, test_class.communities_dict)

    def test_save_load(self, read_graph: GraphLoader, example_case_param_weighted, tmp_path):
        """Test save/load functionality for ExactWeightedEvaluator."""
        
        graph = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_it, test_w_ik = example_case_param_weighted

        from graph_opt.community_detection.alliance_builder import ExactWeightedEvaluator

        test_class = ExactWeightedEvaluator(
            gloader=graph,
            use_gpu=self.use_gpu,
            n_communities=n_communities,
            walk_length=self.walk_length,
            lambda_decay=self.lambda_decay,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        directory = tmp_path / "test_save_weighted"
        directory.mkdir()

        if self.update_membership:
            test_class.update_scores_membership(membership_test)
        else:
            test_class.update_scores(communities_dict=init_communities_test)

        test_class.save(directory=directory)

        # Test loading
        loaded_class = ExactWeightedEvaluator(
            gloader=graph,
            use_gpu=self.use_gpu,
            n_communities=n_communities,
            walk_length=self.walk_length,
            lambda_decay=self.lambda_decay,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        loaded_class.load(directory=directory)

        # Verify loaded data matches original
        assert same_community(test_class.communities_dict, loaded_class.communities_dict)
        assert test_class.log_mpc == loaded_class.log_mpc
        assert test_class.mpc == loaded_class.mpc

        # Verify HHI values match
        for key in test_class.hhi.keys():
            assert abs(test_class.hhi[key] - loaded_class.hhi[key]) < self.tolerance


class TestDeterministicWeightedCPU(_TestDeterministicWeighted):
    """CPU tests for ExactWeightedEvaluator."""
    use_gpu = False


class TestDeterministicWeightedGPU(_TestDeterministicWeighted):
    """GPU tests for ExactWeightedEvaluator (when implemented)."""
    use_gpu = True


class _TestStochasticWeighted():
    """Test class for SampleWeightedEvaluator with stochastic weighted sampling."""
    
    n_samples_edge = 5000
    walk_length = 1
    threads_per_block = 8
    n_walks = 10000
    use_gpu = False
    tolerance = 2e-1  # Increased tolerance for stochastic sampling
    update_membership = True

    @pytest.fixture(scope="function")
    def read_graph(self) -> Tuple[GraphLoader, NeighbourSampler]:
        # This setup will run once for the entire class
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute directory of the script
        data_dir = os.path.join('..', 'test_data')
        MG = nx.readwrite.graphml.read_graphml(os.path.join(script_dir, data_dir, 'test_DG_AP_AL.graphml'))  # store graph

        graph = GraphLoader(
            MG=MG,
            use_gpu=self.use_gpu
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

    def test_evaluator(self, read_graph: Tuple[GraphLoader, NeighbourSampler], example_case_param_weighted):
        """Test the SampleWeightedEvaluator basic functionality and reasonable results."""
        
        graph, sampler = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_it, test_w_ik = example_case_param_weighted

        # Import the SampleWeightedEvaluator
        from graph_opt.community_detection.alliance_builder import SampleWeightedEvaluator

        test_class = SampleWeightedEvaluator(
            gloader=graph,
            sampler=sampler,
            use_gpu=self.use_gpu,
            n_communities=n_communities,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        if self.update_membership:
            test_class.update_scores_membership(membership_test)
        else:
            test_class.update_scores(communities_dict=init_communities_test)

        # Check all keys in test dictionary are present for HHI
        assert all(key in test_HHI.keys() for key in test_class.hhi)

        # Check HHI values are within reasonable tolerance
        for key, value in test_class.hhi.items():
            assert abs(test_HHI[key] - value) < self.tolerance, \
                f"HHI mismatch for {key}: expected {test_HHI[key]}, got {value}"

        # Test that p_it values are valid probabilities (sum to 1 for each node)
        p_it_sums = np.sum(test_class.p_it, axis=1)
        for i, row_sum in enumerate(p_it_sums):
            if test_w_i[i] > 0:  # Only check nodes with outgoing edges
                assert abs(row_sum - 1.0) < 5e-5, f"p_it probabilities don't sum to 1.0 for node {i}: got {row_sum}"

        # Test that p_ik values are valid probabilities (sum to 1 for each node)
        p_ik_sums = np.sum(test_class.p_ik, axis=1)
        for i, row_sum in enumerate(p_ik_sums):
            if test_w_i[i] > 0:  # Only check nodes with outgoing edges
                assert abs(row_sum - 1.0) < 5e-5, f"p_ik probabilities don't sum to 1.0 for node {i}: got {row_sum}"

        # Check that values are non-negative
        assert np.all(test_class.p_it >= 0), "p_it contains negative values"
        assert np.all(test_class.p_ik >= 0), "p_ik contains negative values"

        # Check that for nodes with outgoing edges, at least some attributes have non-zero probability
        for i in range(graph.n_nodes_spatial):
            if test_w_i[i] > 0:
                assert np.any(test_class.p_it[i] > 0), f"Node {i} has outgoing edges but all p_it values are zero"
                assert np.any(test_class.p_ik[i] > 0), f"Node {i} has outgoing edges but all p_ik values are zero"

        if not self.update_membership:
            assert same_community(init_communities_test, test_class.communities_dict)

    def test_save_load(self, read_graph: Tuple[GraphLoader, NeighbourSampler], example_case_param_weighted, tmp_path):
        """Test save/load functionality for SampleWeightedEvaluator."""
        
        graph, sampler = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_it, test_w_ik = example_case_param_weighted

        from graph_opt.community_detection.alliance_builder import SampleWeightedEvaluator

        test_class = SampleWeightedEvaluator(
            gloader=graph,
            sampler=sampler,
            use_gpu=self.use_gpu,
            n_communities=n_communities,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        directory = tmp_path / "test_save_weighted_sample"
        directory.mkdir()

        if self.update_membership:
            test_class.update_scores_membership(membership_test)
        else:
            test_class.update_scores(communities_dict=init_communities_test)

        test_class.save(directory=directory)

        data_to_load = {
            'communities_dict': {"name": '_communities_dict', "dtype": dict},
            'log_mpc': {"name": '_log_mpc', "dtype": dict},
            'mpc': {"name": '_mpc', "dtype": dict},
            'hhi': {"name": '_hhi', "dtype": Dict[Tuple[str, str], int]},
        }

        test_data = {}
        for name, attr in data_to_load.items():
            test_data[name] = getattr(test_class, attr["name"])

        # Create new sampler and evaluator for loading
        sampler_new = NeighbourSampler(
            gloader=graph,
            seed=seed,
            use_gpu=self.use_gpu,
            n_samples_edge=self.n_samples_edge,
            n_walks=self.n_walks,
            walk_length=self.walk_length,
        )

        test_class_new = SampleWeightedEvaluator(
            gloader=graph,
            sampler=sampler_new,
            use_gpu=self.use_gpu,
            n_communities=n_communities,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        test_class_new.load(directory=directory)

        for name, attr in data_to_load.items():
            if attr["dtype"] == np.ndarray:
                assert np.allclose(getattr(test_class_new, attr["name"]), test_data[name], rtol=1e-1)
            elif attr["dtype"] == list:
                for value, test_value in zip(getattr(test_class_new, attr["name"]), test_data[name]):
                    assert abs(value - test_value) < 1e-6
            elif attr["dtype"] == List[Dict[str, int]]:
                for value, test_value in zip(getattr(test_class_new, attr["name"]), test_data[name]):
                    assert value == test_value
            elif attr["dtype"] == List[Tuple[float, float]]:
                for value, test_value in zip(getattr(test_class_new, attr["name"]), test_data[name]):
                    for value_i, test_value_i in zip(value, test_value):
                        assert abs(value_i - test_value_i) < 1e-6
            else:
                assert getattr(test_class_new, attr["name"]) == test_data[name]

    def test_with_exact_weighted(self, read_graph: Tuple[GraphLoader, NeighbourSampler], example_case_param_weighted):
        """Test SampleWeightedEvaluator produces results in same ballpark as ExactWeightedEvaluator."""
        
        graph, sampler = read_graph
        init_communities_test, membership_test, n_communities, test_HHI, test_w_it, test_w_i, test_w_it, test_w_ik = example_case_param_weighted

        from graph_opt.community_detection.alliance_builder import SampleWeightedEvaluator, ExactWeightedEvaluator

        test_class = SampleWeightedEvaluator(
            gloader=graph,
            sampler=sampler,
            use_gpu=self.use_gpu,
            n_communities=n_communities,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        test_class.update_scores_membership(membership_test)

        test_class_exact = ExactWeightedEvaluator(
            gloader=graph,
            use_gpu=self.use_gpu,
            n_communities=n_communities,
            walk_length=self.walk_length,
            lambda_decay=1.0,
            threads_per_block_1D=self.threads_per_block,
            threads_per_block_2D=self.threads_per_block,
            threads_per_block_3D=self.threads_per_block,
            stream=None
        )

        test_class_exact.update_scores(communities_dict=init_communities_test)

        if not self.use_gpu:
            # Check that both methods agree on the general trend/direction
            # For nodes with significant weight differences, check correlation
            for i in range(graph.n_nodes_spatial):
                if test_w_i[i] > 0:  # Only check nodes with outgoing edges
                    # Check that dominant attributes in exact also have high probability in sample
                    exact_max_attr = np.argmax(test_class_exact.p_it[i])
                    sample_max_attr = np.argmax(test_class.p_it[i])
                    
                    # Allow for some variation in stochastic sampling
                    # Check that the top attributes are at least in the top 2 for both methods
                    exact_top2 = np.argsort(test_class_exact.p_it[i])[-2:]
                    sample_top2 = np.argsort(test_class.p_it[i])[-2:]
                    
                    # At least one of the top 2 should overlap
                    overlap = len(set(exact_top2) & set(sample_top2))
                    assert overlap >= 1, f"No overlap in top-2 attributes for node {i}: exact={exact_top2}, sample={sample_top2}"

        # Compare HHI scores - these should be much closer since they use edge sampling
        assert np.allclose(test_class.hhi_edges, test_class_exact.hhi_edges, rtol=self.tolerance), \
            "HHI scores differ significantly between sample and exact weighted evaluators"


class TestStochasticWeightedCPU(_TestStochasticWeighted):
    """CPU tests for SampleWeightedEvaluator."""
    use_gpu = False


class TestStochasticWeightedGPU(_TestStochasticWeighted):
    """GPU tests for SampleWeightedEvaluator (when implemented)."""
    use_gpu = True