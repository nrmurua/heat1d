from math import pi
from matplotlib import pyplot as plt
import numpy as np
import utility
import time
import argparse

np.set_printoptions(suppress=True, precision=4, linewidth=np.inf)

def main():
    parser = argparse.ArgumentParser(description='Process arguments for heat transfer simulation.')
    parser.add_argument('-m', type=int, nargs='+', help='Layer discretization steps (list of integers)')
    parser.add_argument('-N', type=int, help='Temporal discretization steps')
    parser.add_argument('-s', type=int, help='Solver')

    args = parser.parse_args()

    if args.m is None or args.N is None:
        print("Please provide both -m and -N arguments.")
        return
    
    # Step 1: Input Data
    d = 3                                     # Number of layers
    W = [1/3, 1/3, 1/3]                       # Width of the layers
    L = np.zeros(d+1)                         # Boundaries and interfaces
    for i in range(1, d+1):
        L[i] = L[i-1] + W[i-1]
    
    T = 1                                      # End time

    Cap = [1.0, 1.0, 1.0]                       # Heat capacitance
    tao_q = [1.0, 1.0, 1.0]                     # Heat flux phase lag
    tao_T = [1.0, 4.0, 4/3]                     # Temperature gradient phase lags
    k = [4, 1, 6]                               # Thermal Conductivity
    alpha = [0.5, 0.5]                          # Proportionality constants
    Knd = [1.0, 1.0]                            # Knudsen numbers

    m = args.m                            # Layer discretization steps
    N = args.N                             # Temporal discretization steps
    s = args.s

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
    t = np.linspace(0, T, N+1)                  # Time interval discretization
    
    # Step 3: Evaluation of IC-BC-SOURCE on Mesh and Notation Preliminary
    
    start_time = time.time()
    
    

    u0, Du0 = utility.initial_conditions(x, L)

    varphi_1, varphi_2, Dvarphi_1, Dvarphi_2 = utility.boundary_conditions(t)

    #print("varphi_1: \n\n", varphi_1, "\n\n")
    #print("varphi_2: \n\n", varphi_2, "\n\n")
    #print("Dvarphi_1: \n\n", Dvarphi_1, "\n\n")
    #print("Dvarphi_2: \n\n", Dvarphi_2, "\n\n")

    phi_1, phi_2 = utility.compute_phi(varphi_1, varphi_2, Dvarphi_1, Dvarphi_2, tao_T, d)
    
    #print("phi_1: \n\n", phi_1, "\n\n")
    #print("phi_2: \n\n", phi_2, "\n\n")
    
    Cx, kx, tauqx, tauTx = utility.physical_parameters(x, Cap, tao_q, tao_T, k, L)

    #print("Cx: \n\n", Cx, "\n\n")
    #print("kx: \n\n", kx, "\n\n")
    #print("tauqx: \n\n", tauqx, "\n\n")
    #print("tauTx: \n\n", tauTx, "\n\n")
    
    Efe = utility.compute_efe(x, t, L)

    #print("Efe: \n\n", Efe, "\n\n")

    EleL, EleC = utility.compute_ele(Cx, Dt, tauqx)
    
    #print("EleL: \n\n", EleL, "\n\n")
    #print("EleC: \n\n", EleC, "\n\n")

    mu = utility.compute_mu(kx, Dx)

    #print("mu: \n\n", mu, "\n\n")

    PSI_pos, PSI_neg, PSI_R, PSI_L = utility.compute_PSI(tauTx, Dt)
    
    #print("PSI_pos: \n\n", PSI_pos, "\n\n")
    #print("PSI_neg: \n\n", PSI_neg, "\n\n")
    #print("PSI_R: \n\n", PSI_R, "\n\n")
    #print("PSI_L: \n\n", PSI_L, "\n\n")
    
    ZetaL, ZetaU, ZetaC = utility.compute_Zeta(Cx, Dt, tauqx)
    
    #print("ZetaL: \n\n", ZetaL, "\n\n")
    #print("ZetaU: \n\n", ZetaU, "\n\n")
    #print("ZetaC: \n\n", ZetaC, "\n\n")
    
    factor_BC0 = 1 + (Dx[M[0]] / (alpha[0] * Knd[0]))
    factor_BCL = 1 + (Dx[M[d] - 1] / (alpha[1] * Knd[1]))

    #print("factor_BC0: \n\n", factor_BC0, "\n\n")
    #print("factor_BCL: \n\n", factor_BCL, "\n\n")

    end_time = time.time()
    duration = end_time - start_time
    print("Tiempo de ejecución (Step 3):", duration, "segundos")




    start_time = time.time()


    # Step 4: Matrix Evaluation
    hA, hB = utility.compute_hMatrices(M, EleC, mu, Dx, PSI_R, PSI_L, factor_BC0, factor_BCL, d)
    
    #print("hA: \n\n", hA, "\n\n")
    #print("hB: \n\n", hB, "\n\n")
    
    A, B, C = utility.compute_matrices(M, mu, Dx, PSI_pos, PSI_neg, factor_BC0, factor_BCL, ZetaU, ZetaC, ZetaL, d)
    
    #print("A: \n\n", A, "\n\n")
    #print("B: \n\n", B, "\n\n")
    #print("C: \n\n", C, "\n\n")

    Nsource = utility.compute_Nsource(Dt, Dx, M, mu, alpha, Knd, EleL, Efe, phi_1, phi_2, Du0, d, N)
    
    #print("Nsource: \n\n", Nsource, "\n\n")


    end_time = time.time()
    duration = end_time - start_time
    print("Tiempo de ejecución (Step 4):", duration, "segundos")

#
#
    start_time = time.time()



    def solve_system(N, x, Nsource, hA, hB, u0, A, B, C, s):
        switch = {
            0: utility.LU_solver,
            1: utility.QR_solver
        }

        solver = switch.get(s, lambda: print("Invalid solver selection"))

        if callable(solver):
            u, residuals = solver(N, x, Nsource, hA, hB, u0, A, B, C)
            return u, residuals
        else:
            return None, None

    u, residuals = solve_system(N, x, Nsource, hA, hB, u0, A, B, C, s)
    #print("\n\n", u, "\n\n")
    #print("\n\n", residuals, "\n\n")


    end_time = time.time()
    duration = end_time - start_time
    print("Tiempo de ejecución (Step 5):", duration, "segundos")



    # PLOTTING RESULTS

    uexacta = utility.analytical_solution(x, t, L)
    plt.plot(x, u[0, :], '-', label='numerical')
    plt.plot(x, uexacta[0, :], '.-', label='analytic', alpha=0.5)
    plt.title('Initial condition')
    plt.xlabel('space domain')
    plt.ylabel('temperature at t=0')
    plt.legend()
    plt.savefig('Initial_condition.png')  # Save the plot to a file
    plt.close()  # Close the plot to release memory


    # Plotting first iteration
    plt.plot(x, u[1, :], '-', label='numerical')
    plt.plot(x, uexacta[1, :], '.-', label='analytic', alpha=0.5)
    plt.title('First iteration')
    plt.xlabel('space domain')
    plt.ylabel('temperature at t=Dt')
    plt.legend()
    plt.savefig('First_iteration_plot.png')  # Save the plot to a file
    plt.close()  # Close the plot to release memory

    # Plotting solution
    plt.plot(x, u[-1, :], '-', label='numerical')
    plt.plot(x, uexacta[-1, :], '.-', label='analytic', alpha=0.5)
    plt.title('Solution')
    plt.xlabel('space domain')
    plt.ylabel('temperature at t=T')
    plt.legend()
    plt.savefig('Solution.png')  # Save the plot to a file
    plt.close()  # Close the plot to release memory

if __name__ == "__main__":
    main()