
# -*- coding: utf-8 -*-
#
# Adopted from stellargraph

import os
from numba import cuda, errors, config, float64, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64

if os.environ.get('NUMBA_ENABLE_CUDASIM','0') == '1':
    from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray as DeviceNDArray
else:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

config.warnings.filterwarnings("ignore", category=errors.NumbaPerformanceWarning)

@cuda.jit(device=True)
def find_value(a, v):
    for i in range(a.shape[0]):
        if a[i] == v:
            return i
    return -1  # Return -1 if the value is not found

@cuda.jit(device=True)
def replace_values(a, old_value, new_value):
    for i in range(a.shape[0]):
        if a[i] == old_value:
            a[i] = new_value

@cuda.jit(device=True)
def searchsorted_d(a, v):
    out = 0
    total = a[-1]
    for j in range(a.shape[0]-1):
        if a[j] <= v*total:
            out = j+1

    return out

@cuda.jit(device=True)
def cumsum_d(arr, out):
    val = 0
    for j in range(arr.shape[0]):
        val += arr[j]
        out[j] = val

@cuda.jit(device=True)
def searchsorted_global_2d(a, v, idx, idy):
    out = 0
    total = a[idx,idy,-1]
    for j in range(a.shape[-1]-1):
        if a[idx,idy,j] <= v*total:
            out = j+1

    return out

@cuda.jit(device=True)
def cumsum_global_2d(arr, out, idx, idy):
    val = 0
    for j in range(arr.shape[-1]):
        val += arr[idx,idy,j]
        out[idx,idy,j] = val

@cuda.jit(device=True)
def searchsorted_global_3d(a, v, idx, idy, idz):
    out = 0
    total = a[idx,idy,idz,-1]
    for j in range(a.shape[-1]-1):
        if a[idx,idy,idz,j] <= v*total:
            out = j+1

    return out

@cuda.jit(device=True)
def cumsum_global_3d(arr, out, idx, idy, idz):
    val = 0
    for j in range(arr.shape[-1]):
        val += arr[idx,idy,idz,j]
        out[idx,idy,idz,j] = val

@cuda.jit()
def elementwise_divide(A, B):

    # Compute flattened index inside the array  
    x, y = cuda.grid(2)  

    # Compute the grid stride 
    dx, dy = cuda.gridsize(2)  

    for i in range(x, A.shape[0], dx):
        for j in range(y, A.shape[1], dy):
            if B[i,j] > 0:
                A[i,j] /= B[i,j]