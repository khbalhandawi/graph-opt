# -*- coding: utf-8 -*-
#
# Adopted from stellargraph

"""
``stellargraph.random`` contains functions to control the randomness behaviour in StellarGraph.

"""
# `random_state` is not user-facing
__all__ = ["set_seed"]

from .utilities import require_integer_in_range


import random
import numpy.random
import os

from typing import Tuple

import threading
from collections import namedtuple

RandomState = namedtuple("RandomState", "random, numpy")

def _global_state():
    return RandomState(random.Random(), numpy.random.RandomState())


def _seeded_state(s):
    require_integer_in_range(s, "seed", min_val=0)
    return RandomState(random.Random(s), numpy.random.RandomState(s))


_rs = _global_state()


def random_state(seed:int) -> Tuple[random.Random, numpy.random.RandomState]:
    """
    Create a RandomState using the provided seed. If seed is None, return the global RandomState.

    Args:
        seed (int, optional): random seed

    Returns:
        RandomState object
    """
    if seed is None:
        return _rs
    else:
        return _seeded_state(seed)


def set_seed(seed):
    """
    Create a new global RandomState using the provided seed. If seed is None, StellarGraph's global
    RandomState object simply wraps the global random state for each external module.

    When trying to create a reproducible workflow using this function, please note that this seed
    only controls the randomness of the non-TensorFlow part of the library. Randomness within
    TensorFlow layers is controlled via TensorFlow's own global random seed, which can be set using
    ``tensorflow.random.set_seed``.

    Args:
        seed (int, optional): random seed

    """
    global _rs
    if isinstance(seed,Tuple):
        assert len(seed) == 2 and \
            isinstance(seed[0],random.Random) and \
            isinstance(seed[1],numpy.random.RandomState)
        _rs = RandomState(seed[0], seed[1])
    elif isinstance(seed,int):
        _rs = _seeded_state(seed)
    elif seed is None:
        raise ValueError("a seed must be provided")


class SeededPerBatch:
    """
    Internal utility class for managing a random state per batch number in a multi-threaded
    environment.

    """

    def __init__(self, create_with_seed, seed):
        self._create_with_seed = create_with_seed
        self._walkers = []
        self._lock = threading.Lock()
        self._rs, _ = random_state(seed)

    def __getitem__(self, batch_num):
        self._lock.acquire()
        try:
            return self._walkers[batch_num]
        except IndexError:
            # always create a new seeded sampler in ascending order of batch number
            # this ensures seeds are deterministic even when batches are run in parallel
            self._walkers.extend(
                self._create_with_seed(self._rs.randrange(2 ** 32))
                for _ in range(len(self._walkers), batch_num + 1)
            )
            return self._walkers[batch_num]
        finally:
            self._lock.release()