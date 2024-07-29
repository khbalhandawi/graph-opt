from typing import Dict, Any, List, Set, Tuple
import numpy as np

def distinct_communities(community_dict: Dict[str,int]) -> Tuple[List[int], int]:
    distinct_communities_list = list(set(community_dict.values()))
    n_communities = len(distinct_communities_list)

    return distinct_communities_list, n_communities

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
