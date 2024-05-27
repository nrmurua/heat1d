import numpy as np
import pycuda.gpuarray as gpuarray
import torch
import cupy as cp
import time
from cupyx.scipy.linalg import lu_factor, lu_solve, solve_triangular
import matrix_manipulation as mtx

def thomas_algorithm(a, b, c, d):
    n = len(d)
    cp_c = cp.copy(c)
    cp_d = cp.copy(d)
    
    cp_c[0] /= b[0]
    cp_d[0] /= b[0]
    
    for i in range(1, n):
        if i == n - 1:
            temp = b[i]
        else:
            temp = b[i] - a[i] * cp_c[i-1]
        cp_c[i] /= temp
        cp_d[i] = (cp_d[i] - a[i] * cp_d[i-1]) / temp
        
    x = cp.zeros_like(d)
    x[-1] = cp_d[-1]
    
    for i in range(n-2, -1, -1):
        x[i] = cp_d[i] - cp_c[i] * x[i+1]
        
    return x

def evolution_Thomas(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)
    
    d_hA = cp.asarray(d_hA, dtype=cp.float64)
    d_hB = cp.asarray(d_hB, dtype=cp.float64)
    d_A = cp.asarray(d_A, dtype=cp.float64)
    d_B = cp.asarray(d_B, dtype=cp.float64)
    d_C = cp.asarray(d_C, dtype=cp.float64)
    d_u0 = cp.asarray(d_u0, dtype=cp.float64)
    d_Nsource = cp.asarray(d_Nsource, dtype=cp.float64)

    u = cp.zeros((N, len_x), dtype=cp.float64)
    residuals = cp.zeros(N, dtype=cp.float64)

    #print(mtx.is_diagonally_dominant(d_hA))
    #print(mtx.is_diagonally_dominant(d_A))

    lu_hA, piv_hA = lu_factor(d_hA)

    u[0, :] = d_u0
    rhs = cp.matmul(d_hB, u[0, :]) + d_Nsource[0, :]
    unew = lu_solve((lu_hA, piv_hA), rhs)
    u[1, :] = unew
    residuals[0] = cp.linalg.norm(cp.matmul(d_hA, unew) - rhs)

    del lu_hA, piv_hA

    a, b, c = mtx.pre_thomas(d_A)

    #print(d_A)
    #print(a, b, c)

    for n in range(1, N-1):
        rhs = cp.matmul(d_B, u[n, :]) + cp.matmul(d_C, u[n-1, :]) + d_Nsource[n, :]
        unew = thomas_algorithm(a, b, c, rhs)
        u[n+1, :] = unew
        residuals[n] = cp.linalg.norm(cp.matmul(d_A, unew) - rhs)


    return u, residuals

def evolution_cp_lu(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)
    
    d_hA = cp.asarray(d_hA, dtype=cp.float64)
    d_hB = cp.asarray(d_hB, dtype=cp.float64)
    d_A = cp.asarray(d_A, dtype=cp.float64)
    d_B = cp.asarray(d_B, dtype=cp.float64)
    d_C = cp.asarray(d_C, dtype=cp.float64)
    d_u0 = cp.asarray(d_u0, dtype=cp.float64)
    d_Nsource = cp.asarray(d_Nsource, dtype=cp.float64)

    u = cp.zeros((N, len_x), dtype=cp.float64)
    residuals = cp.zeros(N, dtype=cp.float64)

    lu_hA, piv_hA = lu_factor(d_hA)

    u[0, :] = d_u0
    rhs = cp.matmul(d_hB, u[0, :]) + d_Nsource[0, :]
    u[1, :] = lu_solve((lu_hA, piv_hA), rhs)
    residuals[0] = cp.linalg.norm(cp.matmul(d_hA, u[1,:]) - rhs)

    del lu_hA, piv_hA

    lu_A, piv_A = lu_factor(d_A)

    for n in range(1, N-1):
        rhs = cp.matmul(d_B, u[n, :]) + cp.matmul(d_C, u[n-1, :]) + d_Nsource[n, :]
        u[n+1, :] = lu_solve((lu_A, piv_A), rhs)
        residuals[n] = cp.linalg.norm(cp.matmul(d_A, u[n+1, :]) - rhs)

    return u, residuals

def evolution_cp_qr(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)
    
    d_hA = cp.asarray(d_hA, dtype=cp.float64)
    d_hB = cp.asarray(d_hB, dtype=cp.float64)
    d_A = cp.asarray(d_A, dtype=cp.float64)
    d_B = cp.asarray(d_B, dtype=cp.float64)
    d_C = cp.asarray(d_C, dtype=cp.float64)
    d_u0 = cp.asarray(d_u0, dtype=cp.float64)
    d_Nsource = cp.asarray(d_Nsource, dtype=cp.float64)

    u = cp.zeros((N, len_x), dtype=cp.float64)
    residuals = cp.zeros(N, dtype=cp.float64)

    Q_hA, R_hA = cp.linalg.qr(d_hA)

    u[0, :] = d_u0
    rhs = cp.matmul(d_hB, u[0, :]) + d_Nsource[0, :]
    u[1, :] = solve_triangular(R_hA, cp.dot(Q_hA.T, rhs))
    residuals[0] = cp.linalg.norm(cp.matmul(d_hA, u[1,:]) - rhs)

    del Q_hA, R_hA

    Q_A, R_A = cp.linalg.qr(d_A)

    for n in range(1, N-1):
        rhs = cp.matmul(d_B, u[n, :]) + cp.matmul(d_C, u[n-1, :]) + d_Nsource[n, :]
        u[n+1, :] = solve_triangular(R_A, cp.dot(Q_A.T, rhs))
        residuals[n] = cp.linalg.norm(cp.matmul(d_A, u[n+1, :]) - rhs)

    return u, residuals

def evolution_cp(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)

    d_hA = cp.asarray(d_hA, dtype=cp.float64)
    d_hB = cp.asarray(d_hB, dtype=cp.float64)
    d_A = cp.asarray(d_A, dtype=cp.float64)
    d_B = cp.asarray(d_B, dtype=cp.float64)
    d_C = cp.asarray(d_C, dtype=cp.float64)
    d_u0 = cp.asarray(d_u0, dtype=cp.float64)
    d_Nsource = cp.asarray(d_Nsource, dtype=cp.float64)

    u = cp.zeros((N, len_x), dtype=cp.float64)
    residuals = cp.zeros(N, dtype=cp.float64)

    u[0, :] = d_u0
    rhs = cp.matmul(d_hB, u[0, :]) + d_Nsource[0, :]
    u[1, :] = cp.linalg.solve(d_hA, rhs)
    residuals[0] = cp.linalg.norm(cp.matmul(d_hA, u[1,:]) - rhs)

    for n in range(1, N-1):
        rhs = cp.matmul(d_B, u[n, :]) + cp.matmul(d_C, u[n-1, :]) + d_Nsource[n, :]
        u[n+1, :] = cp.linalg.solve(d_A, rhs)
        residuals[n] = cp.linalg.norm(cp.matmul(d_A, u[n+1,:]) - rhs)


    
    return u, residuals


def evolution_torch(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)

    d_u0 = torch.tensor(d_u0.get(), dtype=torch.float64).to('cuda')
    d_Nsource = torch.tensor(d_Nsource.get(), dtype=torch.float64).to('cuda')
    d_hA = torch.tensor(d_hA.get(), dtype=torch.float64).to('cuda')
    d_hB = torch.tensor(d_hB.get(), dtype=torch.float64).to('cuda')
    d_A = torch.tensor(d_A.get(), dtype=torch.float64).to('cuda')
    d_B = torch.tensor(d_B.get(), dtype=torch.float64).to('cuda')
    d_C = torch.tensor(d_C.get(), dtype=torch.float64).to('cuda')

    u = torch.zeros((N, len_x), dtype=torch.float64).to('cuda')
    residuals = torch.zeros(N, dtype=torch.float64)

    u[0, :] = d_u0
    rhs = torch.matmul(d_hB, d_u0) + d_Nsource[0, :]
    unew = torch.linalg.solve(d_hA, rhs)
    u[1, :] = unew
    residuals[0] = torch.linalg.norm(torch.matmul(d_hA, u[1,:]) - rhs)
    
    for n in range(1, N-1):
        rhs = torch.matmul(d_B, u[n, :]) + torch.matmul(d_C, u[n-1, :]) + d_Nsource[n, :]
        u[n+1, :] = torch.linalg.solve(d_A, rhs)
        residuals[n] = torch.linalg.norm(torch.matmul(d_A, u[n+1, :]) - rhs)

    return u, residuals



def conjugate_gradient_gpu(d_A, d_b, d_x0, tol=1e-6, max_iter=100000):
    d_x = d_x0.copy()  
    d_r = d_b - cp.matmul(d_A, d_x)
    d_p = d_r.copy()
    rsold = cp.matmul(d_r, d_r)

    for i in range(max_iter):
        Ap = cp.matmul(d_A, d_p)
        alpha = rsold / cp.matmul(d_p, Ap)
        d_x += alpha * d_p
        d_r -= alpha * Ap
        rsnew = cp.matmul(d_r, d_r)

        if cp.sqrt(cp.asarray(rsnew)) < tol:
            break

        beta = rsnew / rsold
        d_p = d_r + beta * d_p
        rsold = rsnew

    return d_x

def evolution_conjugate_gradient(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    c_hA = mtx.gpu_condition_number(d_hA, len_x)
    print("c_hA:\n\n", c_hA, "\n\n")
    c_A = mtx.gpu_condition_number(d_A, len_x)
    print("c_A:\n\n", c_A, "\n\n")


    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)

    u = cp.zeros((N, len_x), dtype=cp.float64)
    residuals = cp.zeros(N, dtype=cp.float64)

    d_hA = cp.asarray(d_hA, dtype=cp.float64)
    d_hB = cp.asarray(d_hB, dtype=cp.float64)
    d_A = cp.asarray(d_A, dtype=cp.float64)
    d_B = cp.asarray(d_B, dtype=cp.float64)
    d_C = cp.asarray(d_C, dtype=cp.float64)
    d_u0 = cp.asarray(d_u0, dtype=cp.float64)
    d_Nsource = cp.asarray(d_Nsource, dtype=cp.float64)

    u[0, :] = d_u0
    rhs = cp.dot(d_hB, d_u0) + d_Nsource[0, :]
    u[1, :] = conjugate_gradient_gpu(d_hA, rhs, d_u0)

    residuals[0] = cp.linalg.norm(cp.matmul(d_hA, u[1, :]) - rhs)
    
    for n in range(1, N-1):
        rhs = cp.matmul(d_B, u[n, :]) + cp.matmul(d_C, u[n-1, :]) + d_Nsource[n, :]
        u[n+1, :] = conjugate_gradient_gpu(d_A, rhs, u[n,:])
        residuals[n] = cp.linalg.norm(cp.matmul(d_A, u[n+1, :]) - rhs)

    return u, residuals

def evolution_torch_qr(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)

    device_index = torch.cuda.current_device()
    device = torch.device("cuda:" + str(device_index))

    d_hA = torch.tensor(d_hA.get()).to(device=device, dtype=torch.float64)
    d_hB = torch.tensor(d_hB.get()).to(device=device, dtype=torch.float64)
    d_A = torch.tensor(d_A.get()).to(device=device, dtype=torch.float64)
    d_B = torch.tensor(d_B.get()).to(device=device, dtype=torch.float64)
    d_C = torch.tensor(d_C.get()).to(device=device, dtype=torch.float64)
    d_u0 = torch.tensor(d_u0.get()).to(device=device, dtype=torch.float64)
    d_Nsource = torch.tensor(d_Nsource.get()).to(device=device, dtype=torch.float64)

    u = torch.zeros((N, len_x), dtype=torch.float64, device=device)
    residuals = torch.zeros(N, dtype=torch.float64, device=device)

    u[0, :] = d_u0

    rhs = torch.matmul(d_hB, u[0, :]) + d_Nsource[0, :]
    Q_hA, R_hA = torch.linalg.qr(d_hA)
    u[1, :] = torch.linalg.solve_triangular(R_hA, torch.matmul(Q_hA.T, rhs.unsqueeze(1)), upper=True).squeeze(1)

    residuals[0] = torch.linalg.norm(torch.matmul(d_hA, u[1, :]) - rhs)

    del Q_hA, R_hA

    Q_A, R_A = torch.linalg.qr(d_A)

    for n in range(1, N - 1):
        rhs = torch.matmul(d_B, u[n, :]) + torch.matmul(d_C, u[n - 1, :]) + d_Nsource[n, :]
        u[n + 1, :] = torch.linalg.solve_triangular(R_A, torch.matmul(Q_A.T, rhs.unsqueeze(1)), upper=True).squeeze(1)
        residuals[n] = torch.linalg.norm(torch.matmul(d_A, u[n + 1, :]) - rhs)

    del Q_A, R_A

    return u, residuals


def evolution_torch_lu(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)

    device_index = torch.cuda.current_device()
    device = torch.device("cuda:" + str(device_index))

    d_hA = torch.tensor(d_hA.get()).to(device=device, dtype=torch.float64)    
    d_hB = torch.tensor(d_hB.get()).to(device=device, dtype=torch.float64)
    d_A = torch.tensor(d_A.get()).to(device=device, dtype=torch.float64)
    d_B = torch.tensor(d_B.get()).to(device=device, dtype=torch.float64)
    d_C = torch.tensor(d_C.get()).to(device=device, dtype=torch.float64)
    d_u0 = torch.tensor(d_u0.get()).to(device=device, dtype=torch.float64)
    d_Nsource = torch.tensor(d_Nsource.get()).to(device=device, dtype=torch.float64)

    u = torch.zeros((N, len_x), dtype=torch.float64, device=device)
    residuals = torch.zeros(N, dtype=torch.float64, device=device)

    lu_hA, piv_hA = torch.linalg.lu_factor(d_hA)
    u[0] = d_u0
    rhs = torch.matmul(d_hB, u[0]) + d_Nsource[0]
    rhs = rhs.unsqueeze(1)  
    u[1] = torch.linalg.lu_solve(lu_hA, piv_hA, rhs).squeeze(1)  
    residuals[0] = torch.linalg.norm(torch.matmul(d_hA, u[1]) - rhs.squeeze(1))

    lu_A, piv_A = torch.linalg.lu_factor(d_A)

    for n in range(1, N - 1):
        rhs = torch.matmul(d_B, u[n]) + torch.matmul(d_C, u[n - 1]) + d_Nsource[n]
        rhs = rhs.unsqueeze(1)  
        u[n + 1] = torch.linalg.lu_solve(lu_A, piv_A, rhs).squeeze(1)  
        residuals[n] = torch.linalg.norm(torch.matmul(d_A, u[n + 1]) - rhs.squeeze(1))

    return u, residuals

def jacobi_preconditioner(A):
    diag_A = cp.diagonal(A)
    M = cp.diag(1.0 / diag_A)
    return M

def ic_preconditioner(A):
    L = cp.linalg.cholesky(A)
    M = cp.linalg.inv(L)
    return M

def diagonal_scaling_preconditioner(A):
    diag_A = cp.diagonal(A)
    M = cp.diag(diag_A)
    return M

def ssor_preconditioner(A, omega=1.0):
    D = cp.diag(cp.diagonal(A))
    L = cp.tril(A, k=-1)
    U = cp.triu(A, k=1)
    M = cp.linalg.inv(D + omega * L) @ ((1 - omega) * D - omega * U)
    return M

def preconditioned_conjugate_gradient_gpu(A, b, x0, M, tol=1e-6, max_iter=1000000):
    x = x0.copy()
    r = b - cp.matmul(A, x)
    z = cp.matmul(M, r) 
    p = z.copy()
    rsold = cp.matmul(r, z)

    for i in range(max_iter):
        Ap = cp.matmul(A, p)
        alpha = rsold / cp.matmul(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        z = cp.matmul(M, r)
        rsnew = cp.matmul(r, z)

        if cp.sqrt(rsnew) < tol:
            break

        beta = rsnew / rsold
        p = z + beta * p
        rsold = rsnew

    return x

def evolution_preconditioned_conjugate_gradient(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x, preconditioner):
    c_hA = mtx.gpu_condition_number(d_hA, len_x)
    print("c_hA:\n\n", c_hA, "\n\n")
    c_A = mtx.gpu_condition_number(d_A, len_x)
    print("c_A:\n\n", c_A, "\n\n")

    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)

    d_u0 = cp.asarray(d_u0, dtype=cp.float64)
    d_Nsource = cp.asarray(d_Nsource, dtype=cp.float64)
    d_hA = cp.asarray(d_hA, dtype=cp.float64)
    d_hB = cp.asarray(d_hB, dtype=cp.float64)
    d_A = cp.asarray(d_A, dtype=cp.float64)
    d_B = cp.asarray(d_B, dtype=cp.float64)
    d_C = cp.asarray(d_C, dtype=cp.float64)

    def s_jacobi_preconditioner(matrix):
        return jacobi_preconditioner(matrix)

    def s_ic_preconditioner(matrix):
        return ic_preconditioner(matrix)

    def s_diagonal_scaling_preconditioner(matrix):
        return diagonal_scaling_preconditioner(matrix)

    def s_ssor_preconditioner(matrix):
        return ssor_preconditioner(matrix)

    def select_preconditioner(key, matrix):
        preconditioner_switcher = {
            0: s_jacobi_preconditioner,
            1: s_ic_preconditioner,
            2: s_diagonal_scaling_preconditioner,
            3: s_ssor_preconditioner
        }

        preconditioner_func = preconditioner_switcher.get(key)
        if preconditioner_func is None:
            raise ValueError("Invalid preconditioner key")
        
        return preconditioner_func(matrix)

    d_hM = select_preconditioner(preconditioner, d_hA)
    
    u = cp.zeros((N, len_x), dtype=cp.float64)
    residuals = cp.zeros(N, dtype=cp.float64)
    u[0, :] = d_u0

    rhs = cp.matmul(d_hB, u[0, :]) + d_Nsource[0, :]
    unew = preconditioned_conjugate_gradient_gpu(d_hA, rhs, d_u0, d_hM)
    u[1, :] = unew

    residuals[0] = cp.linalg.norm(cp.matmul(d_hA, u[1, :]) - rhs)
    
    del d_hM

    d_M = select_preconditioner(preconditioner, d_A)
    
    for n in range(1, N-1):
        rhs = cp.dot(d_B, u[n, :]) + cp.dot(d_C, u[n-1, :]) + d_Nsource[n, :]
        u[n+1, :] = preconditioned_conjugate_gradient_gpu(d_A, rhs, u[n,:], d_M)
        residuals[n] = cp.linalg.norm(cp.matmul(d_A, u[n+1, :]) - rhs)

    del d_M

    return u, residuals

def jacobi_solver(A, b, x0, iterations=1000000, tol=1e-6):
    D = cp.diag(cp.diag(A))
    L_U = A - D
    x = x0.copy()

    for _ in range(iterations):
        x_new = cp.linalg.solve(D, b - cp.matmul(L_U, x))
        
        if cp.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new

    return x

def evolution_cp_Jacobi(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
    c_hA = mtx.gpu_condition_number(d_hA, len_x)
    print("c_hA:\n\n", c_hA, "\n\n")
    c_A = mtx.gpu_condition_number(d_A, len_x)
    print("c_A:\n\n", c_A, "\n\n")
    
    d_hA = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hA)
    d_hB = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_hB)
    d_A = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_A)
    d_B = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_B)
    d_C = gpuarray.GPUArray((len_x, len_x), np.float64, gpudata=d_C)
    d_u0 = gpuarray.GPUArray(len_x, np.float64, gpudata=d_u0)
    d_Nsource = gpuarray.GPUArray((N, len_x), np.float64, gpudata=d_Nsource)

    d_hA = cp.asarray(d_hA, dtype=cp.float64)
    d_hB = cp.asarray(d_hB, dtype=cp.float64)
    d_A = cp.asarray(d_A, dtype=cp.float64)
    d_B = cp.asarray(d_B, dtype=cp.float64)
    d_C = cp.asarray(d_C, dtype=cp.float64)
    d_u0 = cp.asarray(d_u0, dtype=cp.float64)
    d_Nsource = cp.asarray(d_Nsource, dtype=cp.float64)

    u = cp.zeros((N, len_x), dtype=cp.float64)
    residuals = cp.zeros(N, dtype=cp.float64)

    u[0, :] = d_u0
    rhs = cp.matmul(d_hB, u[0, :]) + d_Nsource[0, :]
    u[1, :] = jacobi_solver(d_hA, rhs, d_u0)

    residuals[0] = cp.linalg.norm(cp.matmul(d_hA, u[1, :]) - rhs)

    for n in range(1, N-1):
        rhs = cp.matmul(d_B, u[n, :]) + cp.matmul(d_C, u[n-1, :]) + d_Nsource[n, :]
        u[n+1, :] = jacobi_solver(d_A, rhs, u[n])
        residuals[n] = cp.linalg.norm(cp.matmul(d_A, u[n+1, :]) - rhs)



    return u, residuals

