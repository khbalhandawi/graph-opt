import numpy as np
import collections
from typing import Tuple
from matplotlib import patches
import matplotlib.pyplot as plt
import joblib
import pickle

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

def get_legend_data(cluster_dict,exclude_keys=[]):
  '''
  Create legend elements and artists from dictionary
  '''
  legend_elements = []
  for key,value in cluster_dict.items():
    if value['n_vertices'] > 0 and key not in exclude_keys:
      legend_elements += [patches.Patch(facecolor=value['color'], 
                                        edgecolor=value['color_frame'], 
                                        label='%s:%i' %(key,value['n_vertices']))]
  return legend_elements

def make_colormap(values,classes=None,colors=None,colors_frame=None):
  '''
  Creates a legend out a list of class predictions
  '''
  if classes is None:
    classes = list(np.sort(np.unique(values)))
  n_classes = len(classes)
  
  if colors is None:
    colors = [rgba_to_hex(plt.cm.rainbow(x)) for x in np.linspace(1,0,n_classes)]
  else:
    colors = [colors[c] for c in classes]

  if colors_frame is None:
    colors_frame = ["#000000"] * (n_classes)

  class_properties = [dict(zip(['color','color_frame'],[c,cf])) for c,cf in zip(colors,colors_frame)]
  class_dict = dict(zip(classes, class_properties))

  for key in class_dict.keys():
    n_vertices = sum([1 if x == key else 0 for x in values])
    class_dict[key]['n_vertices'] = n_vertices

  # create color map for classes
  class_colors = [class_dict[x]['color'] for x in values]
  class_frames = [class_dict[x]['color_frame'] for x in values]
  
  return class_colors, class_frames, class_dict

def rgba_to_hex(rgba: Tuple[float]) -> str:
  rgb = tuple([int(c*255) for c in rgba[:-1]])
  return '#%02x%02x%02x' % rgb

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