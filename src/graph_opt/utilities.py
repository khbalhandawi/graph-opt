import numpy as np
import collections
import joblib
import pickle
import math
import multiprocessing as mp
import multiprocessing.pool as mpp
from itertools import repeat
import tqdm
import inspect
from networkx import Graph, DiGraph

from typing import Tuple, Callable, List, Any, Dict, Union

def _default_if_none(value, default, name, ensure_not_none=True):
    value = value if value is not None else default
    if ensure_not_none and value is None:
        raise ValueError(
            f"{name}: expected a value to be specified in either `__init__` or `run`, found None in both"
        )
    return value

def separated(values, *, limit, stringify, sep):
    """
    Print up to ``limit`` values with a separator.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
        sep (str): the separator to use between elements (and the "... (NNN more)" continuation)
    """
    count = len(values)
    if limit is not None and count > limit:
        values = values[:limit]
        continuation = f"{sep}... ({count - limit} more)" if count > limit else ""
    else:
        continuation = ""

    rendered = sep.join(stringify(x) for x in values)
    return rendered + continuation

def comma_sep(values, limit=20, stringify=repr):
    """
    Print up to ``limit`` values, comma separated.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
    """
    return separated(values, limit=limit, stringify=stringify, sep=", ")

def require_integer_in_range(x, variable_name, min_val=-np.inf, max_val=np.inf):
    """
    A function to verify that a variable is an integer in a specified closed range.
    Args:
        x: the variable to check
        variable_name (str): the name of the variable to print out in error messages
        min_val: the minimum value that `x` can attain
        min_val: the maximum value that `x` can attain
    """

    if not isinstance(x, int):
        raise TypeError(f"{variable_name}: expected int, found {type(x).__name__}")

    if x < min_val or x > max_val:

        if min_val == -np.inf:
            region = f"<= {max_val}"
        elif max_val == np.inf:
            region = f">= {min_val}"
        else:
            region = f"in the range [{min_val}, {max_val}]"

        raise ValueError(f"{variable_name}: expected integer {region}, found {x}")

def is_real_iterable(x):
    """
    Tests if x is an iterable and is not a string.

    Args:
        x: a variable to check for whether it is an iterable

    Returns:
        True if x is an iterable (but not a string) and False otherwise
    """
    return isinstance(x, collections.abc.Iterable) and not isinstance(x, (str, bytes))

# https://stackoverflow.com/a/7205107
def merge_dicts(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def pickle_trick(obj, max_depth=10):
    '''
    Check picklability of an object
    '''
    output = {}

    if max_depth <= 0:
        return output
    try:
        joblib.dump(obj,'test.pkl')
    except:
        failing_children = []
        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                result = pickle_trick(v, max_depth=max_depth - 1)
                if result:
                    failing_children.append(result)

        output = {
            "fail": obj, 
            # "err": e, 
            "depth": max_depth, 
            "failing_children": failing_children
        }

    return output

class myPool(mpp.Pool):

    def istarmap(self, func, iterable, chunksize=1):
        """
        starmap-version of imap
        https://stackoverflow.com/a/57364423
        """
        self._check_running()
        if chunksize < 1:
            raise ValueError(
                "Chunksize must be 1+, not {0:n}".format(
                    chunksize))

        task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
        result = mpp.IMapIterator(self)
        self._taskqueue.put(
            (
                self._guarded_task_generation(result._job,
                                            mpp.starmapstar,
                                            task_batches),
                result._set_length
            ))
        return (item for chunk in result for item in chunk)

def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    """
    https://stackoverflow.com/questions/45718523/pass-kwargs-to-starmap-while-using-pool-in-python
    """
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    # return pool.starmap(apply_args_and_kwargs, tqdm.tqdm(args_for_starmap, total=len(args_iter)))
    return list(tqdm.tqdm(pool.istarmap(apply_args_and_kwargs, args_for_starmap), total=len(args_iter)))

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

def parallel_sampling(func:Callable,
    vargs_iterator:List[List[Any]]=[]   ,vkwargs_iterator:List[Dict[str, Any]]=None,
    fargs:List[Any]=[]                  ,fkwargs:Dict[str, Any]=None,
    n_jobs:int=1):
    """
    function used to run parallel computations

    Parameters
    ----------
    func : Callable
        function to parallelize
    vargs_iterator : List[List[Any]], optional
        iterator of arguments, by default []
    vkwargs_iterator : List[Dict[str, Any]], optional
        iterator of keyword arguments, by default None
    fargs : List[Any], optional
        fixed arguments, by default []
    fkwargs : Dict[str, Any], optional
        fixed keyword arguments, by default None
    n_jobs : int, optional
        number of parallel threads, by default 1

    Returns
    -------
    _type_
        _description_
    """
    args_iter = []
    for vargs in vargs_iterator:
        args_iter += [[*vargs,*fargs]]

    kwargs_iter = []
    for vkwargs in vkwargs_iterator:
        fkwargs.update(vkwargs)
        kwargs_iter += [fkwargs.copy()]

    if n_jobs > 1:
        n_tasks_child = math.ceil(len(args_iter)/n_jobs)
        reset_every = math.floor(n_tasks_child*(1/100))
        print('Start method: %s\nMax tasks per child: %i' %(mp.get_start_method(),reset_every))
        pool = myPool(processes=n_jobs,maxtasksperchild=reset_every)

        results = starmap_with_kwargs(pool, func, args_iter, kwargs_iter)
        pool.terminate()

        """
        https://stackoverflow.com/a/26005535
        # Loop, terminate, and remove from the process list
        # Use a copy [:] of the list to remove items correctly
        for _curr_process in pool._pool[:]:
            print("Terminating process "+ str(_curr_process.pid))
            _curr_process.terminate()
            pool._pool.remove(_curr_process)
        """

    else:
        results = []
        for args,kwargs in zip(args_iter,kwargs_iter):
            result = func(*args, **kwargs)
            results.append(result)

    return results

def serialize(input:Union[list,np.ndarray,dict,type]):
    """
    Converts a dict or list with non-serializable objects to strings

    Parameters
    ----------
    input : dict, list
        input to serialize

    Returns
    -------
    dict, list
        string formatted
    """
    if isinstance(input, dict):
        return {serialize(key): serialize(value)
                for key, value in input.items()}
    elif isinstance(input, (list,np.ndarray)):
        return [serialize(element) for element in input]
    elif inspect.isclass(input):
        return input.__name__
    else:
        return input
    
def deserialize(input:Union[list,np.ndarray,dict,type]):
    """
    Converts a dict or list with non-serializable objects to strings

    Parameters
    ----------
    input : dict, list
        input to serialize

    Returns
    -------
    dict, list
        string formatted
    """
    if isinstance(input, dict):
        return {deserialize(key): deserialize(value)
                for key, value in input.items()}
    elif isinstance(input, (list,np.ndarray)):
        return [deserialize(element) for element in input]
    elif isinstance(input, str):
        return input
    else:
        return input
   
    '''
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    ''' 
def is_pareto_efficient(costs:np.ndarray) -> np.ndarray:
    """
    Find the pareto-efficient points
    https://stackoverflow.com/a/40239615

    Parameters
    ----------
    costs : np.ndarray
        An (n_points, n_costs) array

    Returns
    -------
    np.ndarray
        A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def get_by_attribute(G:Union[Graph,DiGraph],attr:str,value:Any) -> List[Any]:
    """
    Find the pareto-efficient points
    https://stackoverflow.com/a/40239615

    Parameters
    ----------
    attr : str
        attribute to lookup
    value : Any
        attribute value to lookup

    Returns
    -------
    List[Any]
        A list of matching nodes
    """
    return [n for n,d in G.nodes(data=True) if d[attr]==value]
