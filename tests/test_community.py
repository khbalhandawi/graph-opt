from cgi import test
from collections import defaultdict
from doctest import debug
from optparse import Option
import pytest
import os

from graph_opt.community_detection.alliance_builder import GraphLoader, NeighbourSampler, ExactEvaluator, SampleEvaluator, GreedyOptimizer, BruteForce, ExactOptimizer, multiobjective
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
                    assert abs(test_p_it[node][attr] - test_class.w_it[node_id][attr_id]) < self.tolerance
                else:
                    assert test_class.w_i[node_id] == 0

        for node_id,(node,ks) in enumerate(test_w_ik.items()):
            for k in ks.keys():
                
                if test_w_i[node_id] > 0:
                    assert abs(test_p_ik[node].get(k,0) - test_class.w_ik[node_id][k]) < self.tolerance
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
            assert np.allclose(test_class.w_it / (self.walk_length * self.n_walks), test_class_exact.w_it, rtol=1e-1)
            assert np.allclose(test_class.w_ik / (self.walk_length * self.n_walks), test_class_exact.w_ik, rtol=1e-1)
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
        get_evaluator: SampleEvaluator | ExactEvaluator,
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

class TestOptimizeGPU(_TestOptimize):

    use_gpu = True
    use_multiprocessing = False


class TestOptimizeCPU(_TestOptimize):

    use_gpu = False
    use_multiprocessing = False