from typing import Dict, Any, List, Set, Tuple
import numpy as np

def distinct_communities(community_dict: Dict[str,int]) -> Tuple[List[int], int]:
    distinct_communities_list = list(set(community_dict.values()))
    n_communities = len(distinct_communities_list)

    return distinct_communities_list, n_communities

def same_community(dict1: Dict[str,int], dict2: Dict[str,int]) -> bool:
    """_summary_

    Parameters
    ----------
    dict1 : Dict[str,int]
        the first dictionary
    dict2 : Dict[str,int]
        the second dictionary

    Returns
    -------
    bool
        whether the two dictionaries have the same communities
    """
    # Create a mapping from dict1 values to dict2 values
    mapping = {}
    for key in dict1:
        if key in dict2:
            if dict1[key] in mapping:
                if mapping[dict1[key]] != dict2[key]:
                    return False
            else:
                mapping[dict1[key]] = dict2[key]
        else:
            # Key is not in dict2, so grouping is inconsistent
            return False
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