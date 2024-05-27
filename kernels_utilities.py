import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

cuda_initial_conditions = """
#include <math.h>

__global__ void initial_conditions_kernel(double *x, double *L, double *u0, double *Du0, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size) {
        double pi = 3.14159265358979323846f;  // Ensure pi is a double
        double x_val = x[idx];
        if (x_val < L[1]) {
            double sin_val = sinf(3 * pi * x_val / 4);
            u0[idx] = sin_val;
            Du0[idx] = -0.5 * sin_val;
        } else if (x_val < L[2]) {
            double cos_val = -cosf(3 * pi * (2 * x_val - 1) / 4) + sqrt(2.0);
            u0[idx] = cos_val;
            Du0[idx] = -0.5 * cos_val;
        } else {
            double cos_val = cosf(pi * (3 * x_val - 1) / 4);
            u0[idx] = cos_val;
            Du0[idx] = -0.5 * cos_val;
        }
    }
}
"""
mod = SourceModule(cuda_initial_conditions)
initial_conditions_kernel = mod.get_function("initial_conditions_kernel")





cuda_compute_phi = """
__global__ void compute_phi_kernel(double *varphi_1, double *varphi_2, double *Dvarphi_1, double *Dvarphi_2, double *tao_T, double *phi_1, double *phi_2, int size, int d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        phi_1[idx] = varphi_1[idx] + tao_T[0] * Dvarphi_1[idx];
        phi_2[idx] = varphi_2[idx] + tao_T[d-1] * Dvarphi_2[idx];
    }
}
"""

mod = SourceModule(cuda_compute_phi)
compute_phi_kernel = mod.get_function("compute_phi_kernel")





cuda_boundary_conditions = """
#include <math.h>

__global__ void boundary_conditions_kernel(double *t, double *varphi_1, double *varphi_2, double *Dvarphi_1, double *Dvarphi_2, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size) {
        double pi = 3.14159265358979323846f;  // Ensure pi is a double
        double exp_val = expf(-t[idx] / 2);
        varphi_1[idx] = -(3*pi/8) * exp_val;
        varphi_2[idx] = -(3*pi/8) * exp_val;
        Dvarphi_1[idx] = (3*pi/16) * exp_val;
        Dvarphi_2[idx] = (3*pi/16) * exp_val;
    }
}
"""

mod = SourceModule(cuda_boundary_conditions)
boundary_conditions_kernel = mod.get_function("boundary_conditions_kernel")





cuda_physical_parameters = """
__global__ void physical_parameters_kernel(double *x, double *Cx, double *kx, double *L, double *C, double *k, double *tauqx, double *tauTx, double *tao_q, double *tao_T, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size) {
        if (x[idx] < L[1]) {
            Cx[idx] = C[0];
            kx[idx] = k[0];
            tauqx[idx] = tao_q[0];
            tauTx[idx] = tao_T[0];
        } else if (x[idx] < L[2]) {
            Cx[idx] = C[1];
            kx[idx] = k[1];
            tauqx[idx] = tao_q[1];
            tauTx[idx] = tao_T[1];
        } else {
            Cx[idx] = C[2];
            kx[idx] = k[2];
            tauqx[idx] = tao_q[2];
            tauTx[idx] = tao_T[2];
        }
    }
    
}
"""

mod = SourceModule(cuda_physical_parameters)
physical_parameters_kernel = mod.get_function("physical_parameters_kernel")





cuda_compute_efe = """
__global__ void compute_efe_kernel(double *x, double *t, double *L, double *Efe, int t_len, int x_len) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (n < t_len && j < x_len) {
        if (x[j] < L[1]) {
            Efe[n * x_len + j] = ((-2 + 9 * 3.14159265359 * 3.14159265359) / 8) * expf(-t[n] / 2) * sinf(3 * 3.14159265359 * x[j] / 4);
        } else if (x[j] < L[2]) {
            Efe[n * x_len + j] = expf(-t[n] / 2) * (((1 + 9 * 3.14159265359 * 3.14159265359) / 4) * cosf(3 * 3.14159265359 * (2 * x[j] - 1) / 4) - sqrtf(2) / 4);
        } else {
            Efe[n * x_len + j] = ((-2 + 9 * 3.14159265359 * 3.14159265359) / 8) * expf(-t[n] / 2) * cosf(3.14159265359 * (3 * x[j] - 1) / 4);
        }
    }
}
"""

mod = SourceModule(cuda_compute_efe)
compute_efe_kernel = mod.get_function("compute_efe_kernel")





cuda_compute_ele = """
__global__ void compute_ele_kernel(double *Cx, double Dt, double *tauqx, double *EleL, double *EleC, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size) {
        EleL[idx] = 2 * Cx[idx] * tauqx[idx] / (Dt * Dt);
        EleC[idx] = Cx[idx] * (Dt + 2 * tauqx[idx]) / (Dt * Dt);
    }
}
"""

mod = SourceModule(cuda_compute_ele)
compute_ele_kernel = mod.get_function("compute_ele_kernel")





cuda_compute_eme = """
__global__ void compute_mu_kernel(double *kx, double *Dx, double *mu, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) {
        mu[idx] = kx[idx] / (Dx[idx] * Dx[idx]);
    }
}
"""

mod = SourceModule(cuda_compute_eme)
compute_mu_kernel = mod.get_function("compute_mu_kernel")





cuda_compute_PSI = """
__global__ void compute_PSI_kernel(double *tauTx, double Dt, double *PSI_pos, double *PSI_neg, double *PSI_R, double *PSI_L, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) {
        double tauTx_val = tauTx[idx];
        
        PSI_pos[idx] = 0.5 + (tauTx_val / Dt);
        PSI_neg[idx] = 0.5 - (tauTx_val / Dt);
        PSI_R[idx] = 1 + (tauTx_val / Dt);
        PSI_L[idx] = -tauTx_val / Dt;
    }
}
"""

mod = SourceModule(cuda_compute_PSI)
compute_PSI_kernel = mod.get_function("compute_PSI_kernel")

cuda_compute_Zeta = """
__global__ void compute_Zeta_kernel(double *Cx, double Dt, double *tauqx, double *ZetaL, double *ZetaU, double *ZetaC, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) {
        double Cx_val = Cx[idx];
        double tauqx_val = tauqx[idx];
        
        ZetaL[idx] = Cx_val * (-Dt + 2 * tauqx_val) / (2 * (Dt * Dt));
        ZetaU[idx] = Cx_val * (Dt + 2 * tauqx_val) / (2 * (Dt * Dt));
        ZetaC[idx] = ZetaL[idx] + ZetaU[idx];
    }
}
"""

mod = SourceModule(cuda_compute_Zeta)
compute_Zeta_kernel = mod.get_function("compute_Zeta_kernel")






cuda_compute_hMatrices = """
__global__ void compute_hMatrices_kernel(double *EleC, double *mu, double *Dx, double *PSI_R, double *PSI_L, double factor_BC0,
                                         double factor_BCL, int *M, int d, int size, double *hA, double *hB) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx >= M[0] && idx <= M[d]) {
        int i = idx;
        
        if (i == M[0]) {
            // Diagonal #
            hA[0] = EleC[i] + 2 * mu[i] * factor_BC0 * PSI_R[i];
            hB[0] = EleC[i] - 2 * mu[i] * factor_BC0 * PSI_L[i];
            // Upper Diagonal #
            hA[1] = -2 * mu[i] * PSI_R[i];
            hB[1] = 2 * mu[i] * PSI_L[i];
        } else if (i == M[d]) {
            // Under Diagonal #
            hA[i * size + i - 1] = -2 * mu[i - 1] * PSI_R[i - 1];
            hB[i * size + i - 1] = 2 * mu[i - 1] * PSI_L[i - 1];
            // Diagonal #
            hA[i * size + i] = EleC[i] + 2 * mu[i - 1] * factor_BCL * PSI_R[i];
            hB[i * size + i] = EleC[i] - 2 * mu[i - 1] * factor_BCL * PSI_L[i];
        } else if (i < (M[1]) || i > M[d - 1]) {
            // Under Diagonal #
            hA[i * size + i - 1] = -mu[i] * PSI_R[i];
            hB[i * size + i - 1] = mu[i] * PSI_L[i];
            // Diagonal #
            hA[i * size + i] = EleC[i] + 2 * mu[i] * PSI_R[i];
            hB[i * size + i] = EleC[i] - 2 * mu[i] * PSI_L[i];
            // Upper Diagonal #
            hA[i * size + i + 1] = -mu[i] * PSI_R[i+1];
            hB[i * size + i + 1] = mu[i] * PSI_L[i+1];
        } else {
            // Under Diagonal #
            hA[i * size + i - 1] = -2 * Dx[i - 1] * mu[i - 1] * PSI_R[i - 1];
            hB[i * size + i - 1] = 2 * Dx[i - 1] * mu[i - 1] * PSI_L[i - 1];
            // Diagonal #
            hA[i * size + i] = Dx[i - 1] * (EleC[i] + 2 * mu[i - 1] * PSI_R[i - 1]) + Dx[i - 1] * (EleC[i] + 2 * mu[i] * PSI_R[i]);
            hB[i * size + i] = Dx[i - 1] * (EleC[i] - 2 * mu[i - 1] * PSI_L[i - 1]) + Dx[i - 1] * (EleC[i] - 2 * mu[i] * PSI_L[i]);
            // Upper Diagonal #
            hA[i * size + i + 1] = -2 * Dx[i] * mu[i] * PSI_R[i];
            hB[i * size + i + 1] = 2 * Dx[i] * mu[i] * PSI_L[i];
        }
    }
}
"""

mod = SourceModule(cuda_compute_hMatrices)
compute_hMatrices_kernel = mod.get_function("compute_hMatrices_kernel")





cuda_compute_Matrices = """
__global__ void compute_Matrices_kernel(double *A, double *B, double *C, double *ZetaU, double *ZetaC, double *ZetaL, 
                                        double *mu, double *PSI_pos, double *PSI_neg, double *Dx, double factor_BC0, double factor_BCL, 
                                        int *M, int d, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= M[0] && idx <= M[d]) {
        // LEFT BOUNDARY j=M[0]
        if (idx == M[0]) {
            // Diagonal
            A[idx * size + idx] = ZetaU[idx] + mu[idx] * factor_BC0 * PSI_pos[idx];
            B[idx * size + idx] = ZetaC[idx] - mu[idx] * factor_BC0;
            C[idx * size + idx] = -(ZetaL[idx] + mu[idx] * factor_BC0 * PSI_neg[idx]);
            // Upper Diagonal
            A[idx * size + idx + 1] = -mu[idx] * PSI_pos[idx];
            B[idx * size + idx + 1] = mu[idx];
            C[idx * size + idx + 1] = mu[idx] * PSI_neg[idx];
        } else if (idx == M[d]) {
            // Under Diagonal
            A[idx * size + idx - 1] = -mu[idx - 1] * PSI_pos[idx - 1];
            B[idx * size + idx - 1] = mu[idx - 1];
            C[idx * size + idx - 1] = mu[idx - 1] * PSI_neg[idx - 1];
            // Diagonal
            A[idx * size + idx] = ZetaU[idx] + mu[idx - 1] * factor_BCL * PSI_pos[idx - 1];
            B[idx * size + idx] = ZetaC[idx] - mu[idx - 1] * factor_BCL;
            C[idx * size + idx] = -(ZetaL[idx] + mu[idx - 1] * factor_BCL * PSI_neg[idx - 1]);
        } else if (idx < (M[1]) || idx > M[d - 1]) {
            // Under Diagonal
            A[idx * size + idx - 1] = -0.5 * mu[idx] * PSI_pos[idx];
            B[idx * size + idx - 1] = 0.5 * mu[idx];
            C[idx * size + idx - 1] = 0.5 * mu[idx] * PSI_neg[idx];
            // Diagonal
            A[idx * size + idx] = ZetaU[idx] + mu[idx] * PSI_pos[idx];
            B[idx * size + idx] = ZetaC[idx] - mu[idx];
            C[idx * size + idx] = -(ZetaL[idx] + mu[idx] * PSI_neg[idx]);
            // Upper Diagonal
            A[idx * size + idx + 1] = -0.5 * mu[idx] * PSI_pos[idx];
            B[idx * size + idx + 1] = 0.5 * mu[idx];
            C[idx * size + idx + 1] = 0.5 * mu[idx] * PSI_neg[idx];
        } else {
            // Under Diagonal
            A[idx * size + idx - 1] = -Dx[idx - 1] * mu[idx - 1] * PSI_pos[idx - 1];
            B[idx * size + idx - 1] = Dx[idx - 1] * mu[idx - 1];
            C[idx * size + idx - 1] = Dx[idx - 1] * mu[idx - 1] * PSI_neg[idx - 1];
            // Diagonal
            A[idx * size + idx] = Dx[idx - 1] * (ZetaU[idx] + mu[idx - 1] * PSI_pos[idx - 1]) + Dx[idx] * (ZetaU[idx] + mu[idx] * PSI_pos[idx]);
            B[idx * size + idx] = Dx[idx - 1] * (ZetaC[idx] - mu[idx - 1]) + Dx[idx] * (ZetaC[idx] - mu[idx]);
            C[idx * size + idx] = -(Dx[idx - 1] * (ZetaL[idx] + mu[idx - 1] * PSI_neg[idx - 1]) + Dx[idx] * (ZetaL[idx] + mu[idx] * PSI_neg[idx]));
            // Upper Diagonal
            A[idx * size + idx + 1] = -Dx[idx] * mu[idx] * PSI_pos[idx];
            B[idx * size + idx + 1] = Dx[idx] * mu[idx];
            C[idx * size + idx + 1] = Dx[idx] * mu[idx] * PSI_neg[idx];
        }
    }
}
"""

mod = SourceModule(cuda_compute_Matrices)
compute_Matrices_kernel = mod.get_function("compute_Matrices_kernel")




cuda_compute_Nsource = """
__global__ void compute_Nsource_kernel(double *Nsource, double Dt, double *Dx, int *M, double *mu, double *alpha, double *Knd,
                                       double *EleL, double *Efe, double *phi_1, double *phi_2, double *Du0, int d, int N, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        if (idx == M[0]) {
            Nsource[idx] = Dt * EleL[idx] * Du0[idx] + 2 * mu[idx] * (Dx[idx] / (alpha[0] * Knd[0])) * phi_1[1] + Efe[idx];
            for (int n = 1; n < N; n++) {
                Nsource[n * size + idx] = mu[idx] * (Dx[idx] / (2 * alpha[0] * Knd[0])) * (
                            phi_1[n - 1] + 2 * phi_1[n] + phi_1[n + 1]) + 
                                           0.25 * (Efe[(n - 1) * size + idx] + 2 * Efe[n * size + idx] + Efe[(n + 1) * size + idx]);
            }
        } else if (idx == M[d]) {
            Nsource[idx] = Dt * EleL[idx] * Du0[idx] + 2 * mu[idx - 1] * (Dx[idx - 1] / (alpha[1] * Knd[1])) * phi_2[1] + Efe[size + idx];
            for (int n = 1; n < N; n++) {
                Nsource[n * size + idx] = mu[idx - 1] * (Dx[idx - 1] / (2 * alpha[1] * Knd[1])) * 
                         (phi_2[n - 1] + 2 * phi_2[n] + phi_2[n + 1]) + 
                         0.25 * (Efe[(n - 1) * size + idx - 1] + 
                                 2 * Efe[n * size + idx - 1] + 
                                 Efe[(n + 1) * size + idx - 1]);
                }
        } else if (idx < (M[1]) || idx > M[d - 1]) {
            Nsource[idx] = Dt * EleL[idx] * Du0[idx] + Efe[size + idx];
            for (int n = 1; n < N; n++) {
                Nsource[n * size + idx] = 0.25 * (Efe[(n - 1) * size + idx] + 2 * Efe[n * size + idx] + Efe[(n + 1) * size + idx]);
            }
        } else {
            Nsource[idx] = Dt * (Dx[idx - 1] + Dx[idx]) * EleL[idx] * Du0[idx] + Dx[idx - 1] * Efe[size + idx - 1] + Dx[idx] * Efe[size + idx];    
            for (int n = 1; n < N; n++) {
                double FbarL = 0.25 * (Efe[(n - 1) * size + idx - 1] + 2 * Efe[n * size + idx - 1] + Efe[(n + 1) * size + idx - 1]);
                double FbarR = 0.25 * (Efe[(n - 1) * size + idx] + 2 * Efe[n * size + idx] + Efe[(n + 1) * size + idx]);
                Nsource[n * size + idx] = Dx[idx - 1] * FbarL + Dx[idx] * FbarR;
            }
        }
    }
}
"""

mod = SourceModule(cuda_compute_Nsource)
compute_Nsource_kernel = mod.get_function("compute_Nsource_kernel")




