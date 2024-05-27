import numpy as np
from scipy.linalg import lu_factor, lu_solve
import scipy.linalg as spl
import time

pi = np.pi

def initial_conditions(x, L):
    u0 = np.zeros_like(x)
    Du0 = np.zeros_like(x)
    
    for i in range(len(x)):
        if x[i] < L[1]:
            u0[i] = np.sin(3 * pi * x[i] / 4)
            Du0[i] = -0.5 * u0[i]
        elif x[i] < L[2]:
            u0[i] = -np.cos(3 * pi * (2 * x[i] - 1) / 4)+ np.sqrt(2)
            Du0[i] = -0.5 * u0[i]
        else:
            u0[i] = np.cos(pi *(3 * x[i] - 1)  / 4)
            Du0[i] = -0.5 * u0[i]
    
    return u0, Du0

def boundary_conditions(t):
    varphi_1  = np.zeros_like(t)
    varphi_2  = np.zeros_like(t)
    Dvarphi_1 = np.zeros_like(t)
    Dvarphi_2 = np.zeros_like(t)

    for i in range(len(t)):
        varphi_1[i] = -(3*pi/8) * np.exp(-t[i] / 2)
        varphi_2[i] = -(3*pi/8) * np.exp(-t[i] / 2)
        Dvarphi_1[i] = (3*pi/16) * np.exp(-t[i] / 2)
        Dvarphi_2[i] = (3*pi/16) * np.exp(-t[i] / 2)

    return varphi_1, varphi_2, Dvarphi_1, Dvarphi_2

def physical_parameters(x, C, tao_q, tao_T, k, L):
    Cx = np.zeros_like(x)
    kx = np.zeros_like(x)
    tauqx = np.zeros_like(x)
    tauTx = np.zeros_like(x)

    for i in range(len(x)):
        if x[i] < L[1]:
            Cx[i] = C[0]
            kx[i] = k[0]
            tauqx[i] = tao_q[0]
            tauTx[i] = tao_T[0]
        elif x[i] < L[2]:
            Cx[i] = C[1]
            kx[i] = k[1]
            tauqx[i] = tao_q[1]
            tauTx[i] = tao_T[1]
        else:
            Cx[i] = C[2]
            kx[i] = k[2]
            tauqx[i] = tao_q[2]
            tauTx[i] = tao_T[2]

    return Cx, kx, tauqx, tauTx

def compute_efe(x, t, L):
    Efe = np.zeros((len(t),len(x)))

    for i in range(len(t)):
        for j in range(len(x)):
            if x[j] < L[1]:
                Efe[i][j] = ((-2 + 9 * (pi ** 2)) / 8) * np.exp(-t[i] / 2) * np.sin(3 * pi * x[j] / 4)
            elif x[j] < L[2]:
                Efe[i][j] = np.exp(-t[i] / 2) * (((1 + 9 * (pi ** 2)) / 4) * np.cos(3 * pi * (2 * x[j] - 1) / 4) - np.sqrt(2) / 4)
            else:
                Efe[i][j] = ((-2 + 9 * (pi ** 2)) / 8) * np.exp(-t[i] / 2) * np.cos(pi * (3 * x[j] - 1) / 4)

    return Efe

def compute_phi(varphi_1, varphi_2, Dvarphi_1, Dvarphi_2, tao_T, d):
    phi_1 = np.zeros_like(varphi_1)
    phi_2 = np.zeros_like(varphi_2)

    for i in range(len(phi_1)):
        phi_1[i] = varphi_1[i] + tao_T[0] * Dvarphi_1[i]
    for i in range(len(phi_2)):
        phi_2[i] = varphi_2[i] + tao_T[d-1] * Dvarphi_2[i]

    return phi_1, phi_2

def compute_ele(Cx, Dt, tauqx):
    EleL = np.zeros_like(Cx)
    EleC = np.zeros_like(Cx)

    for i in range(len(Cx)):
        EleL[i] = 2 * Cx[i] * tauqx[i] / (Dt ** 2)
        EleC[i] = Cx[i] * (Dt + 2 * tauqx[i]) / (Dt ** 2)

    return EleL, EleC

def compute_mu(kx, Dx):
    mu = np.zeros_like(kx)

    for i in range(len(mu)):
        mu[i] = kx[i] / (Dx[i] ** 2)

    return mu

def compute_PSI(tauTx, Dt):
    PSI_pos = np.zeros_like(tauTx)
    PSI_neg = np.zeros_like(tauTx)
    PSI_R = np.zeros_like(tauTx)
    PSI_L = np.zeros_like(tauTx)

    for i in range(len(tauTx)):
        PSI_pos[i] = 0.5 + (tauTx[i] / Dt)
        PSI_neg[i] = 0.5 - (tauTx[i] / Dt)
        PSI_R[i] = 1 + (tauTx[i]  / Dt)
        PSI_L[i] = -tauTx[i] / Dt


    return PSI_pos, PSI_neg, PSI_R, PSI_L

def compute_Zeta(Cx, Dt, tauqx):
    ZetaL = np.zeros_like(Cx)
    ZetaU = np.zeros_like(Cx)
    ZetaC = np.zeros_like(Cx)

    for i in range(len(ZetaL)):
        ZetaL[i] = Cx[i] * (-Dt + 2 * tauqx[i]) / (2 * (Dt ** 2))
        ZetaU[i] = Cx[i] * (Dt + 2 * tauqx[i]) / (2 * (Dt ** 2))
        ZetaC[i] = ZetaL[i] + ZetaU[i]

    return ZetaL, ZetaU, ZetaC

def compute_hMatrices(M, EleC, mu, Dx, PSI_R, PSI_L, factor_BC0, factor_BCL, d):
    hA = np.zeros((len(Dx),len(Dx)))
    hB = np.zeros((len(Dx),len(Dx)))

    # Diagonal #
    hA[M[0]][M[0]] = EleC[M[0]] + 2 * mu[M[0]] * factor_BC0 * PSI_R[M[0]]
    hB[M[0]][M[0]] = EleC[M[0]] - 2 * mu[M[0]] * factor_BC0 * PSI_L[M[0]]
    # Upper Diagonal #
    hA[M[0]][M[0] + 1] = -2 * mu[M[0]] * PSI_R[M[0]]
    hB[M[0]][M[0] + 1] = 2 * mu[M[0]] * PSI_L[M[0]]

    ## I^lay ##
    for i in range(M[0] + 1, M[d]):
        # Under Diagonal #
        hA[i][i - 1] = -mu[i] * PSI_R[i]
        hB[i][i - 1] = mu[i] * PSI_L[i]
        # Diagonal #
        hA[i][i] = EleC[i] + 2 * mu[i] * PSI_R[i]
        hB[i][i] = EleC[i] - 2 * mu[i] * PSI_L[i]
        # Upper Diagonal #
        hA[i][i + 1] = -mu[i] * PSI_R[i]
        hB[i][i + 1] = mu[i] * PSI_L[i]

    ## I^int ##
    for i in range(M[1], M[d-1]+1):
        # Under Diagonal #
        hA[i][i - 1] = -2 * Dx[i - 1] * mu[i - 1] * PSI_R[i - 1]
        hB[i][i - 1] = 2 * Dx[i - 1] * mu[i - 1] * PSI_L[i - 1]
        # Diagonal #
        hA[i][i] = Dx[i - 1] * (EleC[i] + 2 * mu[i - 1] * PSI_R[i - 1]) + Dx[i - 1] * (EleC[i] + 2 * mu[i] * PSI_R[i])
        hB[i][i] = Dx[i - 1] * (EleC[i] - 2 * mu[i - 1] * PSI_L[i - 1]) + Dx[i - 1] * (EleC[i] - 2 * mu[i] * PSI_L[i])
        # Upper Diagonal #
        hA[i][i + 1] = -2 * Dx[i] * mu[i] * PSI_R[i]
        hB[i][i + 1] = 2 * Dx[i] * mu[i] * PSI_L[i]

    ## M3 ##
    # Under Diagonal #
    hA[M[d]][M[d] - 1] = -2 * mu[M[d] - 1] * PSI_R[M[d] - 1]
    hB[M[d]][M[d] - 1] = 2 * mu[M[d] - 1] * PSI_L[M[d] - 1]
    # Diagonal #
    hA[M[d]][M[d]] = EleC[M[d]] + 2 * mu[M[d] - 1] * factor_BCL * PSI_R[M[d]]
    hB[M[d]][M[d]] = EleC[M[d]] - 2 * mu[M[d] - 1] * factor_BCL * PSI_L[M[d]]

    return hA, hB

def compute_matrices(M, mu, Dx, PSI_pos, PSI_neg, factor_BC0, factor_BCL, ZetaU, ZetaC, ZetaL, d):
    A = np.zeros((len(Dx),len(Dx)))
    B = np.zeros((len(Dx),len(Dx)))
    C = np.zeros((len(Dx),len(Dx)))
    
    # LEFT BOUNDARY j=M_0

    # Diagonal
    A[M[0]][M[0]] = ZetaU[M[0]] + mu[M[0]] * factor_BC0 * PSI_pos[M[0]]
    B[M[0]][M[0]] = ZetaC[M[0]] - mu[M[0]] * factor_BC0 
    C[M[0]][M[0]] = -(ZetaL[M[0]] + mu[M[0]] * factor_BC0 * PSI_neg[M[0]])

    # Upper Diagonal
    A[M[0]][M[0] + 1] = -mu[M[0]] * PSI_pos[M[0]]
    B[M[0]][M[0] + 1] = mu[M[0]]
    C[M[0]][M[0] + 1] = mu[M[0]] * PSI_neg[M[0]]

    # INTERIOR j IN I^lay
    for i in range(M[0] + 1, M[d]):
        # Under Diagonal
        A[i][i - 1] = -0.5 * mu[i] * PSI_pos[i]
        B[i][i - 1] = 0.5 * mu[i]
        C[i][i - 1] = 0.5 * mu[i] * PSI_neg[i]

        # Diagonal
        A[i][i] = ZetaU[i] + mu[i] * PSI_pos[i]
        B[i][i] = ZetaC[i] - mu[i]
        C[i][i] = -(ZetaL[i] + mu[i] * PSI_neg[i])

        # Upper Diagonal
        A[i][i + 1] = -0.5 * mu[i] * PSI_pos[i]
        B[i][i + 1] = 0.5 * mu[i]
        C[i][i + 1] = 0.5 * mu[i] * PSI_neg[i]

    # INTERIOR j IN I^int
    for i in range(M[1], M[d-1]+1):
        # Under Diagonal
        A[i][i - 1] = -Dx[i - 1] * mu[i - 1] * PSI_pos[i - 1]
        B[i][i - 1] = Dx[i - 1] * mu[i - 1]
        C[i][i - 1] = Dx[i - 1] * mu[i - 1] * PSI_neg[i - 1]

        # Diagonal
        A[i][i] = Dx[i - 1] * (ZetaU[i] + mu[i - 1] * PSI_pos[i - 1]) + Dx[i] * (ZetaU[i] + mu[i] * PSI_pos[i])
        B[i][i] = Dx[i - 1] * (ZetaC[i] - mu[i - 1]) + Dx[i] * (ZetaC[i] - mu[i])
        C[i][i] = -(Dx[i - 1] * (ZetaL[i] + mu[i - 1] * PSI_neg[i - 1]) + Dx[i] * (ZetaL[i] + mu[i] * PSI_neg[i]))

        # Upper Diagonal
        A[i][i + 1] = -Dx[i] * mu[i] * PSI_pos[i]
        B[i][i + 1] = Dx[i] * mu[i]
        C[i][i + 1] = Dx[i] * mu[i] * PSI_neg[i]

    # RIGHT BOUNDARY j=M_3
    # Under Diagonal
    A[M[d]][M[d] - 1] = -mu[M[d] - 1] * PSI_pos[M[d] - 1]
    B[M[d]][M[d] - 1] = mu[M[d] - 1]
    C[M[d]][M[d] - 1] = mu[M[d] - 1] * PSI_neg[M[d] - 1]

    # Diagonal
    A[M[d]][M[d]] = ZetaU[M[d]] + mu[M[d] - 1] * factor_BCL * PSI_pos[M[d] - 1]
    B[M[d]][M[d]] = ZetaC[M[d]] - mu[M[d] - 1] * factor_BCL
    C[M[d]][M[d]] = -(ZetaL[M[d]] + mu[M[d] - 1] * factor_BCL * PSI_neg[M[d] - 1])

    return A, B, C

def compute_Nsource(Dt, Dx, M, mu, alpha, Knd, EleL, Efe, phi_1, phi_2, Du0, d, N):
    Nsource = np.zeros((N, len(Dx)))
    
    # Nsource[0][:]
    # LEFT BOUNDARY j=M_0
    Nsource[0][M[0]] = Dt * EleL[M[0]] * Du0[M[0]] + 2 * mu[M[0]] * (Dx[M[0]] / (alpha[0] * Knd[0])) * phi_1[1] + Efe[1][M[0]]

    # INTERIOR j IN I^int
    for i in range(M[0] + 1, M[d]):
        Nsource[0][i] = Dt * EleL[i] * Du0[i] + Efe[1][i]

    # INTERIOR j IN I^int
    for i in range(M[1], M[d-1]+1):
        Nsource[0][i] = Dt * (Dx[i - 1] + Dx[i]) * EleL[i] * Du0[i] + Dx[i - 1] * Efe[1][i - 1] + Dx[i] * Efe[1][i]
  
    # RIGHT BOUNDARY j=M_3
    Nsource[0][M[d]] = Dt * EleL[M[d]] * Du0[M[d]] + 2 * mu[M[d] - 1] * (Dx[M[d] - 1] / (alpha[1] * Knd[1])) * phi_2[1] + Efe[1][M[d]]

    # Nsource[n][:]
    for n in range(1, N):
        # LEFT BOUNDARY j=M_0
        Nsource[n][M[0]] = mu[M[0]] * (Dx[M[0]] / (2 * alpha[0] * Knd[0])) * (
                    phi_1[n - 1] + 2 * phi_1[n] + phi_1[n + 1]) + 0.25 * (
                                        Efe[n - 1][M[0]] + 2 * Efe[n][M[0]] + Efe[n + 1][M[0]])

        # INTERIOR j IN I^int
        for i in range(M[0] + 1, M[d]):
            Nsource[n][i] = 0.25 * (Efe[n - 1][i] + 2 * Efe[n][i] + Efe[n + 1][i])

        # INTERIOR j IN I^int
        for i in range(M[1], M[d-1]+1):
            FbarL = 0.25 * (Efe[n - 1][i - 1] + 2 * Efe[n][i - 1] + Efe[n+1][i-1])
            FbarR = 0.25 * (Efe[n - 1][i] + 2 * Efe[n][i] + Efe[n+1][i])
            Nsource[n][i] = Dx[i - 1] * FbarL + Dx[i] * FbarR

        # RIGHT BOUNDARY j=M_3
        Nsource[n][M[d]] = mu[M[d] - 1] * (Dx[M[d] - 1] / (2 * alpha[1] * Knd[1])) * (
                    phi_2[n - 1] + 2 * phi_2[n] + phi_2[n + 1]) + 0.25 * (
                                                   Efe[n - 1][M[d] - 1] + Efe[n][M[d] - 1] + Efe[n + 1][M[d]] - 1)

    return Nsource

def analytical_solution(x, t, Ele):
    usol = np.zeros((len(t), len(x)))
    for n in range(len(t)):
        for j in range(len(x)):
            if x[j] < Ele[1]:
                usol[n][j] = np.exp(-t[n] / 2) * np.sin(3 * pi * x[j] / 4)
            elif x[j] < Ele[2]:
                usol[n][j] = np.exp(-t[n] / 2) * (-np.cos(3 * pi * (2 * x[j] - 1) / 4) + np.sqrt(2))
            else:
                usol[n][j] = np.exp(-t[n] / 2) * np.cos(pi * (3 * x[j] - 1) / 4)
    return usol

def linalg_solve(N, x, Nsource, hA, hB, u0, A, B, C):

    u = np.zeros((N, len(x)))
    residuals = np.zeros(N, dtype=np.float64)
    u[0, :] = u0

    rhs = hB.dot(u[0,:]) + Nsource[0,:]
    unew = np.linalg.solve(hA, rhs)
    u[1, :] = unew
    residuals[0] = np.linalg.norm(np.matmul(hA, u[1,:]) - rhs)

    # EVOLUTION
    for n in range(1, N-1):
        rhs = B.dot(u[n,:]) + C.dot(u[n-1,:]) + Nsource[n+1]
        u[n+1, :] = np.linalg.solve(A, rhs)
        residuals[n] = np.linalg.norm(np.matmul(A, u[n+1,:]) - rhs)

    return u, residuals

def LU_solver(N, x, Nsource, hA, hB, u0, A, B, C):
    #print("Data type of the array:", hA.dtype)

    u = np.zeros((N, len(x)))
    residuals = np.zeros(N, dtype=np.float64)
    u[0,:] = u0

    lu_hA, piv_hA = lu_factor(hA)

    #print("LU_hA: \n\n", lu_hA, "\n\n")

    rhs = np.matmul(hB, u[0, :]) + Nsource[0, :]
    unew = lu_solve((lu_hA, piv_hA), rhs)
    u[1, :] = unew
    residuals[0] = np.linalg.norm(np.matmul(hA, u[1,:]) - rhs)

    del lu_hA 
    del piv_hA

    lu_A, piv_A = lu_factor(A)

    #print("LU_A: \n\n", lu_A, "\n\n")

    for n in range(1, N-1):
        rhs = np.matmul(B, u[n,:]) + np.matmul(C, u[n-1,:]) + Nsource[n+1,:]
        u[n+1, :] = lu_solve((lu_A, piv_A), rhs)
        residuals[n] = np.linalg.norm(np.matmul(A, u[n+1,:]) - rhs)

    del lu_A 
    del piv_A
    
    return u, residuals

def QR_solver(N, x, d_Nsource, d_hA, d_hB, d_u0, d_A, d_B, d_C):
    u = np.zeros((N, len(x)), dtype=np.float64)
    residuals = np.zeros(N, dtype=np.float64)

    u[0, :] = d_u0

    rhs = np.matmul(d_hB, u[0, :]) + d_Nsource[0, :]
    Q_hA, R_hA = spl.qr(d_hA, mode='economic')
    u[1, :] = spl.solve_triangular(R_hA, np.matmul(Q_hA.T, rhs))

    residuals[0] = np.linalg.norm(np.matmul(d_hA, u[1, :]) - rhs)

    Q_A, R_A = spl.qr(d_A, mode='economic')

    for n in range(1, N-1):
        rhs = np.matmul(d_B, u[n, :]) + np.matmul(d_C, u[n-1, :]) + d_Nsource[n, :]
        u[n+1, :] = spl.solve_triangular(R_A, np.matmul(Q_A.T, rhs))
        residuals[n] = np.linalg.norm(np.matmul(d_A, u[n+1, :]) - rhs)

    return u, residuals