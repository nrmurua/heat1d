from matplotlib import pyplot as plt
import numpy as np
import utility as util
import gpu_utilities as gpu
import gpu_solver as gpusolver
import pycuda.driver as cuda
import matrix_manipulation as mtx
import time
import argparse
import cupy as cp


np.set_printoptions(suppress=True, precision=12, linewidth=np.inf)

def main():
    parser = argparse.ArgumentParser(description='Process arguments for heat transfer simulation.')
    parser.add_argument('-m', type=int, nargs='+', help='Layer discretization steps (list of integers)')
    parser.add_argument('-N', type=int, help='Temporal discretization steps')
    parser.add_argument('-s', type=int, help='Solver')
    parser.add_argument('-p', type=int, help='Preconditioner')

    args = parser.parse_args()

    if args.m is None or args.N is None:
        print("Please provide both -m and -N arguments.")
        return

    # Step 1: Input Data

    d = 3                                     # Number of layers
    W = [1/3, 1/3, 1/3]                       
    L = np.zeros(d+1)                         # Boundaries and interfaces
    for i in range(1, d+1):
        L[i] = L[i-1] + W[i-1]

    T = 1                                      # End time

    Cap = [1, 1, 1]                       
    tao_q = [1, 1, 1]                     
    tao_T = [1.0, 4.0, 4/3]                 
    
    k = [4, 1, 6]                               

    alpha = [0.5, 0.5]                          # Proportionality constants
    Knd = [1, 1]                            # Knudsen numbers

    m = args.m                            # Layer discretization steps
    N = args.N                             # Temporal discretization steps
    s = args.s
    p = args.p


    # Step 2: Discretization of Domain

    M = np.zeros(d+1, dtype=int)                # Index for interfaces and boundaries
    M[0] = 0
    for i in range(d):
        M[i+1] = m[i] + M[i]
    
    DW = np.zeros(d)                            # Size step discretization of each layer
    for i in range(d):
        DW[i] = (L[i+1] - L[i]) / m[i]
    
    x = np.zeros(M[d]+1)                          # Discretization of the spatial domain [0, L]
    for i in range(d):
        for j in range(M[i]+1, M[i+1]+1):
            x[j] = L[i] + (j - M[i]) * DW[i]
    
    Dx = np.zeros(M[d]+1)                         # Size of space step domain [0, L]
    for i in range(M[d]):
        Dx[i] = x[i+1] - x[i]
    Dx[M[d]] = Dx[M[d]-1]

    Dt = T / N                                 # Size step of time discretization
    t = np.arange(0, T+Dt, Dt)                  # Time interval discretization




    # Step 3: Alloc GPU memory

    # Transform arrays to np.float64 for gpu usage
    x = x.astype(np.float64)
    L = L.astype(np.float64)
    t = t.astype(np.float64)
    tao_T = np.array(tao_T, dtype=np.float64)
    Cap = np.array(Cap, dtype=np.float64)
    tao_q = np.array(tao_q, dtype=np.float64)
    k = np.array(k, dtype=np.float64)
    Dx = np.array(Dx, dtype=np.float64)

    # Alloc memory on GPU

    d_x = cuda.mem_alloc(x.nbytes)
    d_L = cuda.mem_alloc(L.nbytes)
    d_t = cuda.mem_alloc(t.nbytes)
    d_tao_T = cuda.mem_alloc(tao_T.nbytes)
    d_tao_q = cuda.mem_alloc(tao_q.nbytes)
    d_Cap = cuda.mem_alloc(Cap.nbytes)
    d_k = cuda.mem_alloc(k.nbytes)
    d_Dx = cuda.mem_alloc(Dx.nbytes)

    # Copy data from host (CPU) to device (GPU)

    cuda.memcpy_htod(d_x, x)
    cuda.memcpy_htod(d_L, L)
    cuda.memcpy_htod(d_t, t)
    cuda.memcpy_htod(d_tao_T, tao_T)
    cuda.memcpy_htod(d_tao_q, tao_q)
    cuda.memcpy_htod(d_Cap, Cap)
    cuda.memcpy_htod(d_k, k)
    cuda.memcpy_htod(d_Dx, Dx)

    #print("x:\n\n", x, "\n\n")
    #print("L:\n\n", L, "\n\n")
    #print("t:\n\n", t, "\n\n")

    

    start_time = time.time()


    # Step 4: Evaluation of IC-BC-SOURCE on Mesh and Notation Preliminary

    len_x = len(x)
    len_t = len(t)

    # Step 4.1: GPU functions

    d_u0, d_Du0 = gpu.gpu_initial_conditions(d_x, d_L, len_x)
    
    #u0 = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(u0.data, d_u0)
    #np.savetxt('values/u0.txt', u0, fmt='%0.4f', delimiter=' ')
#
    #Du0 = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(Du0.data, d_Du0)
    #np.savetxt('values/Du0.txt', Du0, fmt='%0.4f', delimiter=' ')
    
    d_varphi_1, d_varphi_2, d_Dvarphi_1, d_Dvarphi_2 = gpu.gpu_boundary_conditions(d_t, len_t)

    #varphi_1 = np.zeros(len_t, dtype=np.float64)
    #cuda.memcpy_dtoh(varphi_1.data, d_varphi_1)
    #np.savetxt('values/varphi_1.txt', varphi_1, fmt='%0.4f', delimiter=' ')
#
    #varphi_2 = np.zeros(len_t, dtype=np.float64)
    #cuda.memcpy_dtoh(varphi_2.data, d_varphi_2)
    #np.savetxt('values/varphi_2.txt', varphi_2, fmt='%0.4f', delimiter=' ')
#
    #Dvarphi_1 = np.zeros(len_t, dtype=np.float64)
    #cuda.memcpy_dtoh(Dvarphi_1.data, d_Dvarphi_1)
    #np.savetxt('values/Dvarphi_1.txt', Dvarphi_1, fmt='%0.4f', delimiter=' ')
#
    #Dvarphi_2 = np.zeros(len_t, dtype=np.float64)
    #cuda.memcpy_dtoh(Dvarphi_2.data, d_Dvarphi_2)
    #np.savetxt('values/Dvarphi_2.txt', Dvarphi_2, fmt='%0.4f', delimiter=' ')

    d_phi_1, d_phi_2 = gpu.gpu_compute_phi(d_varphi_1, d_varphi_2, d_Dvarphi_1, d_Dvarphi_2, d_tao_T, d, len_t)
    
    #phi_1 = np.zeros(len_t, dtype=np.float64)
    #cuda.memcpy_dtoh(phi_1.data, d_phi_1)
    #np.savetxt('values/phi_1.txt', phi_1, fmt='%0.4f', delimiter=' ')
#
    #phi_2 = np.zeros(len_t, dtype=np.float64)
    #cuda.memcpy_dtoh(phi_2.data, d_phi_2)
    #np.savetxt('values/phi_2.txt', phi_2, fmt='%0.4f', delimiter=' ')

    d_Cx, d_kx, d_tauqx, d_tauTx = gpu.gpu_physical_parameters(d_x, d_Cap, d_tao_q, d_tao_T, d_k, d_L, len_x)
    
    #Cx = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(Cx.data, d_Cx)
    #np.savetxt('values/Cx.txt', Cx, fmt='%0.4f', delimiter=' ')
    #
    #kx = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(kx.data, d_kx)
    #np.savetxt('values/kx.txt', kx, fmt='%0.4f', delimiter=' ')
    #
    #tauqx = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(tauqx.data, d_tauqx)
    #np.savetxt('values/tauqx.txt', tauqx, fmt='%0.4f', delimiter=' ')
    #
    #tauTx = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(tauTx.data, d_tauTx)
    #np.savetxt('values/tauTx.txt', tauTx, fmt='%0.4f', delimiter=' ')

    d_Efe = gpu.gpu_compute_efe(d_x, d_t, d_L, len_t, len_x)

    #Efe = mtx.device_to_np_matrix(d_Efe, len_t, len_x)
    #np.savetxt('values/Efe.txt', Efe, fmt='%0.4f', delimiter=' ')

    d_EleL, d_EleC = gpu.gpu_compute_ele(d_Cx, Dt, d_tauqx, len_x)

    #EleL = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(EleL.data, d_EleL)
    #np.savetxt('values/EleL.txt', EleL, fmt='%0.4f', delimiter=' ')
    #
    #EleC = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(EleC.data, d_EleC)
    #np.savetxt('values/EleC.txt', EleC, fmt='%0.4f', delimiter=' ')

    d_mu = gpu.gpu_compute_mu(d_kx, d_Dx, len_x)
    
    #mu = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(mu.data, d_mu)
    #np.savetxt('values/mu.txt', mu, fmt='%0.4f', delimiter=' ')
    
    d_PSI_pos, d_PSI_neg, d_PSI_R, d_PSI_L = gpu.gpu_compute_PSI(d_tauTx, Dt, len_x)
    
    #PSI_pos = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(PSI_pos.data, d_PSI_pos)
    #np.savetxt('values/PSI_pos.txt', PSI_pos, fmt='%0.4f', delimiter=' ')
    #
    #PSI_neg = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(PSI_neg.data, d_PSI_neg)
    #np.savetxt('values/PSI_neg.txt', PSI_neg, fmt='%0.4f', delimiter=' ')
    #
    #PSI_R = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(PSI_R.data, d_PSI_R)
    #np.savetxt('values/PSI_R.txt', PSI_R, fmt='%0.4f', delimiter=' ')
    #
    #PSI_L = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(PSI_L.data, d_PSI_L)
    #np.savetxt('values/PSI_L.txt', PSI_L, fmt='%0.4f', delimiter=' ')

    d_ZetaL, d_ZetaU, d_ZetaC = gpu.gpu_compute_Zeta(d_Cx, Dt, d_tauqx, len_x)

    #ZetaL = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(ZetaL.data, d_ZetaL)
    #np.savetxt('values/ZetaL.txt', ZetaL, fmt='%0.4f', delimiter=' ')
    #
    #ZetaU = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(ZetaU.data, d_ZetaU)
    #np.savetxt('values/ZetaU.txt', ZetaU, fmt='%0.4f', delimiter=' ')
    #
    #ZetaC = np.zeros(len_x, dtype=np.float64)
    #cuda.memcpy_dtoh(ZetaC.data, d_ZetaC)
    #np.savetxt('values/ZetaC.txt', ZetaC, fmt='%0.4f', delimiter=' ')
    
    # Step 4.2: CPU functions

    factor_BC0 = 1 + (Dx[M[0]] / (alpha[0] * Knd[0]))
    factor_BCL = 1 + (Dx[M[d] - 1] / (alpha[1] * Knd[1]))

    #print(factor_BC0)
    #print(factor_BCL)


    end_time = time.time()
    duration = end_time - start_time
    print("Tiempo de ejecución (Step 3):", duration, "segundos")


    # Step 5: Dealloc Unnecesary values

    d_x.free()
    d_L.free()
    d_t.free()
    d_tao_T.free()
    d_Cap.free()
    d_tao_q.free()
    d_k.free()
    d_varphi_1.free()
    d_varphi_2.free()
    d_Dvarphi_1.free()
    d_Dvarphi_2.free()
    d_Cx.free()
    d_kx.free()
    d_tauqx.free()
    d_tauTx.free()




    # Step 6: Alloc hMatrices, Matrices, Nsource and additional variables

    # Variables

    M = M.astype(np.int32)
    alpha = np.array(alpha, dtype=np.float64)
    Knd = np.array(Knd, dtype=np.float64)

    d_M = cuda.mem_alloc(M.nbytes)
    d_alpha = cuda.mem_alloc(alpha.nbytes)
    d_Knd = cuda.mem_alloc(Knd.nbytes)

    cuda.memcpy_htod(d_M, M)
    cuda.memcpy_htod(d_alpha, alpha)
    cuda.memcpy_htod(d_Knd, Knd)


    #np.savetxt('values/M.txt', M, fmt='%0.4f', delimiter=' ')
    #np.savetxt('values/alpha.txt', alpha, fmt='%0.4f', delimiter=' ')
    #np.savetxt('values/Knd.txt', Knd, fmt='%0.4f', delimiter=' ')


    start_time = time.time()


    # Step 7: Compute hMatrices, Matrices and Nsource

    d_hA, d_hB = gpu.gpu_compute_hMatrices(d_M, d_EleC, d_mu, d_Dx, d_PSI_R, d_PSI_L, factor_BC0, factor_BCL, d, len_x)

    #hA = mtx.device_to_np_matrix(d_hA, len_x, len_x)
    #print(mtx.is_diagonally_dominant(hA,True))

    #hB = mtx.device_to_np_matrix(d_hB, len_x, len_x)

    #np.savetxt('values/hA.txt', hA, fmt='%0.4f', delimiter=' ')
    #np.savetxt('values/hB.txt', hB, fmt='%0.4f', delimiter=' ')

    d_A, d_B, d_C = gpu.gpu_compute_Matrices(d_ZetaU, d_ZetaC, d_ZetaL, d_mu, d_PSI_pos, d_PSI_neg, d_Dx, factor_BC0, factor_BCL, d_M, d, len_x)

    #A = mtx.device_to_np_matrix(d_A, len_x, len_x)
    #B = mtx.device_to_np_matrix(d_B, len_x, len_x)
    #C = mtx.device_to_np_matrix(d_C, len_x, len_x)

    #np.savetxt('values/A.txt', A, fmt='%0.4f', delimiter=' ')
    #np.savetxt('values/B.txt', B, fmt='%0.4f', delimiter=' ')
    #np.savetxt('values/C.txt', C, fmt='%0.4f', delimiter=' ')

    d_Nsource = gpu.gpu_compute_Nsource(Dt, d_Dx, d_M, d_mu, d_alpha, d_Knd, d_EleL, d_Efe, d_phi_1, d_phi_2, d_Du0, d, N, len_x)

    #Nsource = mtx.device_to_np_matrix(d_Nsource, N, len_x)
    #np.savetxt('values/Nsource.txt', Nsource, fmt='%0.4f', delimiter=' ')
    

    end_time = time.time()
    duration = end_time - start_time
    print("Tiempo de ejecución (Step 4):", duration, "segundos")


    # Step 8: Dealloc Unnecesary values

    d_Du0.free()
    d_Dx.free()
    d_phi_1.free()
    d_phi_2.free()
    d_Efe.free()
    d_EleC.free()
    d_EleL.free()
    d_mu.free()
    d_PSI_L.free()
    d_PSI_R.free()
    d_PSI_pos.free()
    d_PSI_neg.free()
    d_ZetaC.free()
    d_ZetaL.free()
    d_ZetaU.free()
    d_M.free()
    d_alpha.free()
    d_Knd.free()


    # Step 9: Compute evolution

    def cupy_lu_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_cp_lu(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.get(), residuals
    
    def cupy_qr_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_cp_qr(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.get(), residuals

    def cupy_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_cp(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.get(), residuals

    def torch_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_torch(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.cpu().numpy(), residuals

    def conjugate_gradient_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_conjugate_gradient(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.get(), residuals

    def preconditioned_conjugate_gradient_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_preconditioned_conjugate_gradient(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x, p)
        return d_u.get(), residuals

    def cupy_jacobi_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_cp_Jacobi(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.get(), residuals

    def torch_qr_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_torch_qr(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.cpu().numpy(), residuals.cpu().numpy()

    def evolution_Thomas(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_Thomas(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.get(), residuals

    def torch_lu_solver(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x):
        d_u, residuals = gpusolver.evolution_torch_lu(d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)
        return d_u.cpu().numpy(), residuals.cpu().numpy()

    def switch_solver(method, *args):
        switcher = {
            0: cupy_lu_solver,
            1: cupy_solver,
            2: torch_solver,
            3: conjugate_gradient_solver,  
            4: preconditioned_conjugate_gradient_solver,
            5: cupy_jacobi_solver,
            6: torch_qr_solver,
            7: evolution_Thomas,
            8: torch_lu_solver,
            9: cupy_qr_solver,
        }

        func = switcher.get(method, lambda *args: (None, None))
        
        return func(*args)

    start_time = time.time()

    u, residuals = switch_solver(s, d_u0, d_Nsource, d_hA, d_hB, d_A, d_B, d_C, N, len_x)

    end_time = time.time()
    duration = end_time - start_time
    print("Tiempo de ejecución (Step 5):", duration, "segundos")

    #np.savetxt('values/u.txt', u, fmt='%0.4f', delimiter=' ')

    print("\n\n", u, "\n\n")
    print("\n\n", residuals, "\n\n")
    print("\n\nResiduals: ", cp.average(residuals), "\n\n")


    # Step 10: Copy data and dealloc unnecesessary data

    d_u0.free()
    d_Nsource.free()
    d_hA.free()
    d_hB.free()
    d_A.free()
    d_B.free()
    d_C.free()

    # PLOTTING RESULTS

    # Plotting initial condition
    uexacta = util.analytical_solution(x, t, L)
    plt.plot(x, u[0, :], '-', label='numerical')
    plt.plot(x, uexacta[0, :], '.-', label='analytic', alpha=0.5)
    plt.title('Initial condition')
    plt.xlabel('space domain')
    plt.ylabel('temperature at t=0')
    plt.legend()
    plt.savefig('Initial_condition_plot.png')  
    plt.close()  

    # Plotting first iteration
    plt.plot(x, u[1, :], '-', label='numerical')
    plt.plot(x, uexacta[1, :], '.-', label='analytic', alpha=0.5)
    plt.title('First iteration')
    plt.xlabel('space domain')
    plt.ylabel('temperature at t=Dt')
    plt.legend()
    plt.savefig('First_iteration.png')  
    plt.close()  

    # Plotting solution
    plt.plot(x, u[-1, :], '-', label='numerical')
    plt.plot(x, uexacta[-1, :], '.-', label='analytic', alpha=0.5)
    plt.title('Solution')
    plt.xlabel('space domain')
    plt.ylabel('temperature at t=T')
    plt.legend()
    plt.savefig('Solution.png') 
    plt.close()  



if __name__ == "__main__":
    main()