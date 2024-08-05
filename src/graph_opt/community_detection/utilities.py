from typing import Dict, Any, List, Set, Tuple
import numpy as np
from itertools import product

def distinct_communities(community_dict: Dict[str,int]) -> Tuple[List[int], int]:
    distinct_communities_list = list(set(community_dict.values()))
    n_communities = len(distinct_communities_list)

    return distinct_communities_list, n_communities

def same_community(dict1: dict, dict2: dict) -> bool:
    """
    Check if two dictionaries map their keys to the same community ids.

    Parameters:
    - dict1 (dict): First dictionary of community assignments.
    - dict2 (dict): Second dictionary of community assignments.

    Returns:
    - bool: True if both dictionaries represent the same community structure, False otherwise.
    """
    if set(dict1.keys()) != set(dict2.keys()):
        return False  # Ensures both dictionaries have the same keys

    value_map = {}
    reverse_map = {}

    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        if val1 in value_map:
            if value_map[val1] != val2:
                return False
        else:
            value_map[val1] = val2

        if val2 in reverse_map:
            if reverse_map[val2] != val1:
                return False
        else:
            reverse_map[val2] = val1

    return True

def partition_to_set_of_sets(membership: np.ndarray) -> Set[frozenset]:
    """ 
    Convert membership array to a set of frozensets, where each frozenset represents a community. 
    
    Parameters
    ----------

    membership : np.ndarray
        the membership array to convert to a set of frozensets

    Returns
    -------
    Set[frozenset]
        a set of frozensets, where each frozenset represents a community
    """
    partition_dict = {}
    for index, community in enumerate(membership):
        if community not in partition_dict:
            partition_dict[community] = set()
        partition_dict[community].add(index)
    return frozenset(frozenset(indices) for indices in partition_dict.values())

def generate_membership_combinations(membership: np.ndarray, max_community: int) -> Dict[Tuple[int,int],np.ndarray]:
    """
    Generate all possible membership combinations by merging communities in the membership array.

    Parameters
    ----------
    membership : np.ndarray
        the membership array to generate combinations from
    max_community : int
        the maximum community index to consider

    Returns
    -------
    Dict[Tuple[int,int],np.ndarray]
        a dictionary of tuples, where the key is a tuple of the communities merged and the value is the new membership array
    """

    unique_communities = np.unique(membership)
    n = max(max_community + 1, len(unique_communities))  # Adjust the range to consider up to max_community
    seen_partitions = set()
    results = {}

    for i in range(n):
        if i not in unique_communities:
            continue  # Skip the loop if community `i` has no members
        for j in range(i + 1, n):
            if j not in unique_communities:
                continue  # Skip the loop if community `j` has no members

            new_membership = membership.copy()
            new_membership[membership == j] = i  # Merge community j into i
            partition_set = partition_to_set_of_sets(new_membership)

            if partition_set not in seen_partitions:
                seen_partitions.add(partition_set)
                results[(i, j)] = new_membership
    
    return results



def canonical_form(membership):
    """ Convert membership to its canonical form to treat equivalent patterns identically. """
    mapper = {}
    new_label = 0
    canonical = []
    for community in membership:
        if community not in mapper:
            mapper[community] = new_label
            new_label += 1
        canonical.append(mapper[community])
    return tuple(canonical)

def generate_possible_memberships(n: int, max_community: int) -> set:
    """
    Generate all possible unique membership combinations for a vector of length `n`
    considering up to `max_community` communities.

    Parameters
    ----------
    n : int
        Length of the membership vector.
    max_community : int
        Maximum index for communities, starting from 0.

    Returns
    -------
    set
        A set of tuples, each representing a unique community membership configuration.
    """

    # Generate all possible combinations of community memberships
    combinations = product(range(max_community + 1), repeat=n)

    # Use a set to store unique canonical forms
    unique_memberships = set()
    for combo in combinations:
        # Convert each combination to its canonical form and add to set
        unique_memberships.add(canonical_form(combo))

    return unique_memberships