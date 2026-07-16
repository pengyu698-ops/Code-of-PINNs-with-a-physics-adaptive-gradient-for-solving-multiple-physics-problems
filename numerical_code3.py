#Numerical Solution Code for Hydrogen Molecule
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import pandas as pd
R_nuc = 1.4                       
a_soft = 1.0                      
L_domain = 4.0                      
N = 100                        
x = np.linspace(-L_domain, L_domain, N)
dx = x[1] - x[0]
X1, X2 = np.meshgrid(x, x, indexing='ij')
V_e1_n = -1.0 / np.sqrt((X1 - R_nuc/2)**2 + a_soft**2) - 1.0 / np.sqrt((X1 + R_nuc/2)**2 + a_soft**2)
V_e2_n = -1.0 / np.sqrt((X2 - R_nuc/2)**2 + a_soft**2) - 1.0 / np.sqrt((X2 + R_nuc/2)**2 + a_soft**2)
V_ee = 1.0 / np.sqrt((X1 - X2)**2 + a_soft**2)
V_nn = 1.0 / R_nuc
V_total = V_e1_n + V_e2_n + V_ee + V_nn
V_flat = V_total.flatten()
V_sp = sp.diags(V_flat) 
diag = np.ones(N)
T_1D = sp.spdiags([-diag, 2*diag, -diag], [-1, 0, 1], N, N) / (2 * dx**2)
I_1D = sp.eye(N)
T_2D = sp.kron(T_1D, I_1D) + sp.kron(I_1D, T_1D)
H = T_2D + V_sp
eigenvalues, eigenvectors = eigsh(H, k=10, which='SA')
found_bonding = False
found_antibonding = False
for i in range(len(eigenvalues)):
    E = eigenvalues[i]
    psi_flat = eigenvectors[:, i]
    psi_2d = psi_flat.reshape((N, N))
    norm_factor = np.sqrt(np.sum(psi_2d**2) * (dx**2))
    psi_2d = psi_2d / norm_factor
    diff_sym = np.max(np.abs(psi_2d - psi_2d.T))
    diff_anti = np.max(np.abs(psi_2d + psi_2d.T))
    state_type = None
    if diff_sym < 1e-5 and not found_bonding:
        state_type = "bonding"
        found_bonding = True
        print(f"Found {state_type} state (Ground State)! Energy = {E:.6f} a.u.")
    elif diff_anti < 1e-5 and not found_antibonding:
        state_type = "antibonding"
        found_antibonding = True
        print(f"Found {state_type} state (Excited State)! Energy = {E:.6f} a.u.")
    if state_type:
        P_num = np.abs(psi_2d)**2
        Psi_real_num = psi_2d
        Psi_imag_num = np.zeros_like(psi_2d)
        if np.max(Psi_real_num) < np.abs(np.min(Psi_real_num)):
            Psi_real_num = -Psi_real_num
        pd.DataFrame(P_num).to_excel(f'P_num_{state_type}.xlsx', index=False, header=False)
        pd.DataFrame(Psi_real_num).to_excel(f'Psi_real_num_{state_type}.xlsx', index=False, header=False)
        pd.DataFrame(Psi_imag_num).to_excel(f'Psi_imag_num_{state_type}.xlsx', index=False, header=False)
    if found_bonding and found_antibonding:
        break
