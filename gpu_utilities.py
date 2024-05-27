import numpy as np
import pycuda.driver as cuda
import kernels_utilities as kutil
import pycuda.gpuarray as gpuarray
import cupy as cp

def gpu_initial_conditions(d_x, d_L, size):
    d_u0 = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_Du0 = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    
    block_size = 1024  
    grid_size = (size + block_size - 1) // block_size
    
    kutil.initial_conditions_kernel(d_x, d_L, d_u0, d_Du0, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))
    
    return d_u0, d_Du0





def gpu_compute_phi(d_varphi_1, d_varphi_2, d_Dvarphi_1, d_Dvarphi_2, d_tao_T, d, size):    
    d_phi_1 = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_phi_2 = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)

    block_size = 1024  
    grid_size = (size + block_size - 1) // block_size
    kutil.compute_phi_kernel(d_varphi_1, d_varphi_2, d_Dvarphi_1, d_Dvarphi_2, d_tao_T, d_phi_1, d_phi_2, np.int32(size), np.int32(d), block=(block_size, 1, 1), grid=(grid_size, 1))
    
    return d_phi_1, d_phi_2





def gpu_boundary_conditions(d_t, size):
    d_varphi_1 = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_varphi_2 = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_Dvarphi_1 = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_Dvarphi_2 = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    
    block_size = 1024
    grid_size = (size + block_size - 1) // block_size
    kutil.boundary_conditions_kernel(d_t, d_varphi_1, d_varphi_2, d_Dvarphi_1, d_Dvarphi_2, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))
    
    return d_varphi_1, d_varphi_2, d_Dvarphi_1, d_Dvarphi_2





def gpu_physical_parameters(d_x, d_C, d_tao_q, d_tao_T, d_k, d_L, size):    
    d_Cx = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_kx = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_tauqx = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_tauTx = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    
    block_size = 1024
    grid_size = (size + block_size - 1) // block_size
    kutil.physical_parameters_kernel(d_x, d_Cx, d_kx, d_L, d_C, d_k, d_tauqx, d_tauTx, d_tao_q, d_tao_T, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return d_Cx, d_kx, d_tauqx, d_tauTx






def gpu_compute_efe(d_x, d_t, d_L, len_t, len_x):
    d_Efe = cuda.mem_alloc(len_t * len_x * np.dtype(np.float64).itemsize)
    cuda.memset_d8(d_Efe, 0, len_t * len_x * np.dtype(np.float64).itemsize)

    block_size = (32, 32, 1)  
    grid_size = ((len_t + block_size[0] - 1) // block_size[0], (len_x + block_size[1] - 1) // block_size[1])

    kutil.compute_efe_kernel(d_x, d_t, d_L, d_Efe, np.int32(len_t), np.int32(len_x), block=block_size, grid=grid_size)

    return d_Efe





def gpu_compute_ele(d_Cx,Dt, d_tauqx, size):
    d_EleL = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_EleC = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    
    block_size = 1024 
    grid_size = (size + block_size - 1) // block_size
    kutil.compute_ele_kernel(d_Cx, np.float64(Dt), d_tauqx, d_EleL, d_EleC, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))
    
    return d_EleL, d_EleC





def gpu_compute_mu(d_kx, d_Dx, size):
    d_mu = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)

    block_size = 1024
    grid_size = (size + block_size - 1) // block_size

    kutil.compute_mu_kernel(d_kx, d_Dx, d_mu, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return d_mu





def gpu_compute_PSI(d_tauTx, Dt, size):
    d_PSI_pos = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_PSI_neg = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_PSI_R = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_PSI_L = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)

    block_size = 1024
    grid_size = (size + block_size - 1) // block_size

    kutil.compute_PSI_kernel(d_tauTx, np.float64(Dt), d_PSI_pos, d_PSI_neg, d_PSI_R, d_PSI_L, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return d_PSI_pos, d_PSI_neg, d_PSI_R, d_PSI_L





def gpu_compute_Zeta(d_Cx, Dt, d_tauqx, size):
    d_ZetaL = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_ZetaU = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)
    d_ZetaC = cuda.mem_alloc(size * np.dtype(np.float64).itemsize)

    block_size = 1024
    grid_size = (size + block_size - 1) // block_size

    kutil.compute_Zeta_kernel(d_Cx, np.float64(Dt), d_tauqx, d_ZetaL, d_ZetaU, d_ZetaC, np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return d_ZetaL, d_ZetaU, d_ZetaC





def gpu_compute_hMatrices(d_M, d_EleC, d_mu, d_Dx, d_PSI_R, d_PSI_L, factor_BC0, factor_BCL, d, size):
    d_hA = cuda.mem_alloc(size * size * np.dtype(np.float64).itemsize)
    d_hB = cuda.mem_alloc(size * size * np.dtype(np.float64).itemsize)
    cuda.memset_d8(d_hA, 0, size * size * np.dtype(np.float64).itemsize)
    cuda.memset_d8(d_hB, 0, size * size * np.dtype(np.float64).itemsize)


    block_size = 1024
    grid_size = (size * size + block_size - 1) // block_size

    kutil.compute_hMatrices_kernel(d_EleC, d_mu, d_Dx, d_PSI_R, d_PSI_L, np.float64(factor_BC0), np.float64(factor_BCL), d_M, np.int32(d), np.int32(size), d_hA, d_hB, block=(block_size, 1, 1), grid=(grid_size, 1))

    return d_hA, d_hB





def gpu_compute_Matrices(d_ZetaU, d_ZetaC, d_ZetaL, d_mu, d_PSI_pos, d_PSI_neg, d_Dx, factor_BC0, factor_BCL, d_M, d, size):
    d_A = cuda.mem_alloc(size * size * np.dtype(np.float64).itemsize)
    d_B = cuda.mem_alloc(size * size * np.dtype(np.float64).itemsize)
    d_C = cuda.mem_alloc(size * size * np.dtype(np.float64).itemsize)
    cuda.memset_d8(d_A, 0, size * size * np.dtype(np.float64).itemsize)
    cuda.memset_d8(d_B, 0, size * size * np.dtype(np.float64).itemsize)
    cuda.memset_d8(d_C, 0, size * size * np.dtype(np.float64).itemsize)

    block_size = 1024
    grid_size = (size * size + block_size - 1) // block_size

    kutil.compute_Matrices_kernel(d_A, d_B, d_C, d_ZetaU, d_ZetaC, d_ZetaL, d_mu, d_PSI_pos, d_PSI_neg, d_Dx, np.float64(factor_BC0), np.float64(factor_BCL), d_M, np.int32(d), np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return d_A, d_B, d_C





def gpu_compute_Nsource(Dt, d_Dx, d_M, d_mu, d_alpha, d_Knd, d_EleL, d_Efe, d_phi_1, d_phi_2, d_Du0, d, N, size):
    d_Nsource = cuda.mem_alloc(N * size * np.dtype(np.float64).itemsize)
    cuda.memset_d8(d_Nsource, 0, N * size * np.dtype(np.float64).itemsize)

    block_size = 1024
    grid_size = (size * size + block_size - 1) // block_size

    kutil.compute_Nsource_kernel(d_Nsource, np.float64(Dt), d_Dx, d_M, d_mu, d_alpha, d_Knd,
                                       d_EleL, d_Efe, d_phi_1, d_phi_2, d_Du0, np.int32(d), np.int32(N), np.int32(size), block=(block_size, 1, 1), grid=(grid_size, 1))

    return d_Nsource





