import numpy as np
from ..utilities import dict2array
import pytest

from typing import Dict, Union, Any
from ..community_detection.utilities_cpu import calc_membership_pairs_cpu

DictType = Dict[Any,Union[Any,'DictType']]

@pytest.mark.parametrize(
    'test_data, test_array', [
        (
            {
                ('0', '1'): {0: 0.0, 1: 0.0, 2: 0.0, 3: 40.0, 4: 60.0},
                ('0', '2'): {0: 100.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                ('0', '4'): {0: 0.0, 1: 0.0, 2: 100.0, 3: 0.0, 4: 0.0},
                ('1', '2'): {0: 0.0, 1: 100.0, 2: 0.0, 3: 0.0, 4: 0.0},
                ('3', '4'): {0: 0.0, 1: 0.0, 2: 80.0, 3: 20.0, 4: 0.0},
                ('4', '0'): {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 100.0}
            },
            np.array([
                [0.0, 0.0, 0.0, 40.0, 60.0],
                [100.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 100.0, 0.0, 0.0],
                [0.0, 100.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 80.0, 20.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 100.0],
            ])
        ),
        (
            {
                ('0', '1'): {0: 0.0, 1: 0.0, 2: 40.0, 3: 60.0}, 
                ('0', '2'): {0: 100.0, 1: 0.0, 2: 0.0, 3: 0.0}, 
                ('0', '4'): {0: 0.0, 1: 0.0, 2: 0.0, 3: 100.0}, 
                ('1', '2'): {0: 0.0, 1: 100.0, 2: 0.0, 3: 0.0}, 
                ('3', '4'): {0: 0.0, 1: 0.0, 2: 20.0, 3: 80.0}, 
                ('4', '0'): {0: 0.0, 1: 0.0, 2: 0.0, 3: 100.0}
            },
            np.array([
                [0.0, 0.0, 40.0, 60.0],
                [100.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 100.0],
                [0.0, 100.0, 0.0, 0.0],
                [0.0, 0.0, 20.0, 80.0],
                [0.0, 0.0, 0.0, 100.0],
            ])
        ),

        (
            {
                'a': {
                    'a1': {
                        'a11': {0: 1, 1: 2, 2: 3},
                        'a12': {0: 1, 1: 2, 2: 3}
                    },
                    'a2': {
                        'a21': {0: 4, 1: 5, 2: 6},
                        'a22': {0: 4, 1: 5, 2: 6}
                    }
                },
                'b': {
                    'b1': {
                        'b11': {0: 7, 1: 8, 2: 9},
                        'b12': {0: 13, 1: 14, 2: 15}
                    },
                    'b2': {
                        'b21': {0: 10, 1: 11, 2: 12},
                        'b22': {0: 13, 1: 14, 2: 15}
                    }
                }
            },
            np.array([
                [
                    [
                        [1, 2, 3],
                        [1, 2, 3]
                    ], 
                    [
                        [4, 5, 6],
                        [4, 5, 6],
                    ]
                ], 
                [
                    [
                        [7, 8, 9], 
                        [13, 14, 15]
                    ], 
                    [
                        [10, 11, 12], 
                        [13, 14, 15]
                    ]
                ]
            ])
        )
    ],
    ids=[
        '2D array case 1',
        '2D array case 2',
        '3D array'
    ]
)
def test_dict2array(test_data:DictType, test_array:np.ndarray):
    assert np.all(dict2array(test_data) == test_array)

def test_calc_membership_pairs_cpu_shape_and_type():
    # Test input parameters
    n_attr = 5
    n_communities = 3
    membership = np.array([0, 1, 2, 0, 1])
    membership_pairs = np.zeros((n_communities, n_communities, n_attr), dtype=np.int32)

    # Expected shape
    expected_shape = (n_communities, n_communities, n_attr)
    
    # Run the function
    result = calc_membership_pairs_cpu(membership, membership_pairs, n_communities)

    # Check results
    assert result.dtype == np.int32, "Resulting dtype should be int32"
    assert result.shape == expected_shape, "Resulting shape should match expected shape"

def test_calc_membership_pairs_cpu_output():
    # Define inputs
    n_attr = 4
    n_communities = 2
    membership = np.array([0, 1, 0, 1])
    membership_pairs = np.zeros((n_communities, n_communities, n_attr), dtype=np.int32)
    
    # Calculate result
    result = calc_membership_pairs_cpu(membership, membership_pairs, n_communities)

    # Expected result
    expected_result = np.array([
        [[0, 1, 0, 1], [1, 1, 1, 1]],
        [[0, 0, 0, 0], [0, 1, 0, 1]]
    ], dtype=np.int32)

    # Verify the output
    np.testing.assert_array_equal(result, expected_result, "The output array does not match the expected array")

def test_calc_membership_pairs_cpu_edge_case_single_community():
    # Single community scenario
    n_attr = 3
    n_communities = 1
    membership = np.array([0, 0, 0])
    membership_pairs = np.zeros((n_communities, n_communities, n_attr), dtype=np.int32)
    
    # Expected result: All elements should be the same since only one community is present
    expected_result = np.zeros((n_communities, n_communities, n_attr), dtype=np.int32)
    
    # Run the function
    result = calc_membership_pairs_cpu(membership, membership_pairs, n_communities)

    # Verify the output
    np.testing.assert_array_equal(result, expected_result, "The output should be all zeros for a single community case")
