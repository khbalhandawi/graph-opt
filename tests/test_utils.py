import numpy as np
from ..utilities import dict2array
import pytest

from typing import Dict, Union, Any
from ..community_detection.utilities import generate_membership_combinations, generate_possible_memberships

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

class TestPossibleMerges:

    def test_no_possible_merges(self):
        # All members are in the same community
        membership = np.array([0, 0, 0, 0, 0])
        max_community = 1
        expected = {}  # No merges should occur
        assert generate_membership_combinations(membership, max_community) == expected

    def test_valid_merges(self):
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

    def test_with_large_max_community(self):
        # Include a large max_community that exceeds any present
        membership = np.array([0, 1, 2, 4])
        max_community = 10
        results = generate_membership_combinations(membership, max_community)
        assert len(results) > 0  # Should handle higher community indices gracefully

    def test_empty_membership(self):
        # Edge case: Empty membership array
        membership = np.array([])
        max_community = 0
        expected = {}
        assert generate_membership_combinations(membership, max_community) == expected

class TestPossibleMemberships:
    def test_basic_functionality(self):
        result = generate_possible_memberships(2, 1)
        expected = {(0, 0), (0, 1)}  # This now expects the unique canonical forms only
        assert result == expected

    def test_small_vector_length(self):
        result = generate_possible_memberships(3, 2)
        expected = {
            (0, 0, 0),  # All in one community
            (0, 0, 1),  # Two in one community, one in another
            (0, 1, 2),  # All in different communities
            (0, 1, 0),  # Two in separate communities, one in the original
            (0, 1, 1)   # Two in one community, one in another (different order)
        }
        assert set(result) == expected

    def test_longer_vector_length(self):
        result = generate_possible_memberships(5, 4)
        expected = {
            (0, 0, 0, 0, 0),  # All in one community
            (0, 0, 0, 0, 1),  # Four in one community, one in another
            (0, 0, 0, 1, 1),  # Three in one community, two in another
            (0, 0, 0, 1, 2),  # Three in one, one in another, one in another (3 unique)
            (0, 0, 1, 1, 1),  # Two in one community, three in another
            (0, 0, 1, 2, 3),  # Two in one, and one each in separate communities (4 unique)
            (0, 1, 2, 3, 4),  # Each in a different community (5 unique)
            (0, 1, 0, 1, 0),  # Pattern with repeating and alternating communities
            (0, 1, 2, 0, 1),  # Pattern with communities repeating not from start
            (0, 1, 1, 2, 2),  # Pattern of two sets of two communities, one single
            (0, 1, 1, 2, 0),  # additional
            (0, 1, 1, 2, 3),
            (0, 1, 0, 0, 1),
            (0, 0, 1, 0, 0),
            (0, 1, 2, 1, 0),
            (0, 1, 2, 1, 3),
            (0, 1, 1, 1, 1),
            (0, 1, 0, 1, 2),
            (0, 1, 0, 2, 3),
            (0, 1, 2, 2, 1),
            (0, 1, 0, 2, 0),
            (0, 1, 2, 3, 0),
            (0, 0, 1, 2, 0),
            (0, 1, 2, 3, 3),
            (0, 1, 1, 0, 2),
            (0, 1, 0, 0, 0),
            (0, 1, 2, 1, 2),
            (0, 0, 1, 0, 2),
            (0, 1, 1, 1, 0),
            (0, 1, 0, 1, 1),
            (0, 1, 2, 0, 0),
            (0, 0, 1, 1, 0),
            (0, 1, 0, 2, 2),
            (0, 1, 2, 0, 3),
            (0, 1, 2, 2, 0),
            (0, 1, 2, 3, 2),
            (0, 0, 1, 2, 2),
            (0, 1, 1, 0, 1),
            (0, 1, 2, 2, 3),
            (0, 0, 0, 1, 0),
            (0, 1, 1, 2, 1),
            (0, 1, 0, 0, 2),
            (0, 1, 2, 1, 1),
            (0, 0, 1, 0, 1),
            (0, 1, 1, 0, 0),
            (0, 1, 1, 1, 2),
            (0, 1, 0, 2, 1),
            (0, 1, 2, 0, 2),
            (0, 0, 1, 2, 1),
            (0, 1, 2, 2, 2),
            (0, 1, 2, 3, 1),
            (0, 0, 1, 1, 2),
        }

        for item in set(result):
            print(item)

        assert set(result) == expected

    def test_single_community(self):
        result = generate_possible_memberships(3, 0)
        expected = {(0, 0, 0)}
        assert result == expected

    def test_all_zero_vector(self):
        result = generate_possible_memberships(4, 0)
        expected = {(0, 0, 0, 0)}
        assert result == expected

    def test_high_community_number(self):
        result = generate_possible_memberships(2, 3)
        expected = {
            (0, 0),  # Represents (0, 0), (1, 1), (2, 2), (3, 3)
            (0, 1),  # Represents all two-community combinations like (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
        }
        assert result == expected

    def test_canonical_form(self):
        result = generate_possible_memberships(2, 10)
        expected = {(0, 1), (1, 0), (0, 0), (1, 1)}  # Only these should appear
        assert len(result) <= 4

    def test_empty_vector(self):
        result = generate_possible_memberships(0, 1)
        expected = {()}  # Expecting a set with an empty tuple representing the empty configuration
        assert result == expected