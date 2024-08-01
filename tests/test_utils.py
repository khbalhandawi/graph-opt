import numpy as np
from ..utilities import dict2array
import pytest

from typing import Dict, Union, Any
from ..community_detection.utilities import generate_membership_combinations, partition_to_set_of_sets

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

def test_no_possible_merges():
    # All members are in the same community
    membership = np.array([0, 0, 0, 0, 0])
    max_community = 1
    expected = {}  # No merges should occur
    assert generate_membership_combinations(membership, max_community) == expected

def test_valid_merges():
    # Members belong to different communities
    membership = np.array([0, 1, 2])
    max_community = 2
    results = generate_membership_combinations(membership, max_community)
    # Verify that the results contain the expected merges and that partitions are correct
    assert (0, 1) in results
    assert (0, 2) in results
    assert (1, 2) in results
    assert len(results) == 3
    # Each result should reflect the proper merging
    assert np.all(results[(0, 1)] == np.array([0, 0, 2]))
    assert np.all(results[(0, 2)] == np.array([0, 1, 0]))
    assert np.all(results[(1, 2)] == np.array([0, 1, 1]))

def test_with_large_max_community():
    # Include a large max_community that exceeds any present
    membership = np.array([0, 1, 2, 4])
    max_community = 10
    results = generate_membership_combinations(membership, max_community)
    assert len(results) > 0  # Should handle higher community indices gracefully

def test_empty_membership():
    # Edge case: Empty membership array
    membership = np.array([])
    max_community = 0
    expected = {}
    assert generate_membership_combinations(membership, max_community) == expected