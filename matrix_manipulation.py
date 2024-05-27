import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import cupy as cp

def device_to_np_matrix(d_mat, rows, cols):
    h_mat = np.empty((rows, cols), dtype=np.float64)
    cuda.memcpy_dtoh(h_mat, d_mat)
    return h_mat

def is_diagonally_dominant(matrix, strict=False):
    n = matrix.shape[0]

    for i in range(n):
        diag_abs = abs(matrix[i, i])        
        off_diag_sum = cp.sum(cp.abs(matrix[i, :])) - diag_abs
        
        if strict:
            if diag_abs <= off_diag_sum:
                return False
        else:
            if diag_abs < off_diag_sum:
                return False

    return True

def gpu_condition_number(d_A, len_x):
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_A = cp.asarray(d_A)

    U, S, VT = cp.linalg.svd(d_A)

    max_S = cp.amax(S)
    min_S = cp.amin(S)

    cond_num = max_S / min_S

    return cond_num

import cupy as cp

def pre_thomas(A):
    n = A.shape[0]
    
    a = cp.zeros(n, dtype=A.dtype)
    b = cp.zeros(n, dtype=A.dtype)
    c = cp.zeros(n, dtype=A.dtype)
    
    for i in range(n):
        b[i] = A[i, i]
        if i < n-1:
            c[i] = A[i, i+1]
        if i > 0:
            a[i-1] = A[i, i-1]
    
    return a, b, c