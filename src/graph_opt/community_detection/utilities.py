from typing import Dict, Any, List, Set, Tuple
import numpy as np

def resort_communities(community_dict: Dict[str,int]) -> Tuple[Dict[int,int], Dict[str,int], List[int], int]:
    # retrieve iteration constants
    distinct_communities_set = set(community_dict.values())
    reordered_communities_list = list(range(len(distinct_communities_set)))

    orig2sorted = dict(zip(sorted(distinct_communities_set),reordered_communities_list))
    n_communities = len(orig2sorted)

    reordered_communities_dict = {key:orig2sorted[value] for key,value in community_dict.items()}

    sorted2orig = dict(zip(reordered_communities_list,sorted(distinct_communities_set)))

    return sorted2orig, reordered_communities_dict, reordered_communities_list, n_communities

def same_community(dict1: Dict[str,int], dict2: Dict[str,int]) -> bool:
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
