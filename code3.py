#Physical Adaptive Gradient Clipping Sin-Tanh Hybrid PINN Architecture for Solving Hydrogen Molecule Code
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
R_nuc = 1.4                        
a_soft = 1.0                      
L_domain = 4.0                      
x_domain = [-L_domain, L_domain]    
y_domain = [-L_domain, L_domain]    
N_phy = 15000                  
N_bc = 4000                      
N_norm_int = 80                     
Loss_Weight = {'phy': 1.0, 'bc': 10.0, 'norm': 5.0, 'ortho': 50.0} 
Learning_Rate = 1e-3   
torch.manual_seed(42)
Epoch = 30000                     
def normalize(data, domain):
    return 2.0 * (data - domain[0]) / (domain[1] - domain[0]) - 1.0
def denormalize(data, domain):
    return 0.5 * (data + 1.0) * (domain[1] - domain[0]) + domain[0]
class SinActivation(nn.Module):
    def __init__(self, omega):
        super().__init__()
        self.omega = omega
    def forward(self, x):
        return torch.sin(self.omega * x)
class SoftConstraintPINN(nn.Module):
    def __init__(self, omega=30.0, out_dim=2, domain1=x_domain, domain2=y_domain): 
        super(SoftConstraintPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), SinActivation(omega),
            nn.Linear(128, 256), SinActivation(omega),
            nn.Linear(256, 256), SinActivation(omega),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, out_dim)
        )
        linear_idx = 0
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    if linear_idx == 0:  
                        freq_1 = (domain1[1] - domain1[0]) / omega
                        freq_2 = (domain2[1] - domain2[0]) / omega
                        m.weight[:, 0].uniform_(-freq_1, freq_1) 
                        m.weight[:, 1].uniform_(-freq_2, freq_2) 
                    else:             
                        m.weight.uniform_(-np.sqrt(6 / m.in_features) / omega, 
                                           np.sqrt(6 / m.in_features) / omega)
                    linear_idx += 1
    def forward(self, x1, x2):
        net_input = torch.cat([x1, x2], dim=1)
        net_output = self.net(net_input)
        return net_output
def get_complex_psi(model, x1_norm, x2_norm):
    net_output = model(x1_norm, x2_norm)
    real_part = net_output[:, [0]]
    imag_part = net_output[:, [1]]
    return torch.complex(real_part, imag_part)
def get_symmetrized_psi(model, x1_norm, x2_norm, sym_type='bonding'):
    psi_12 = get_complex_psi(model, x1_norm, x2_norm)
    psi_21 = get_complex_psi(model, x2_norm, x1_norm) 
    if sym_type == 'bonding':
        return (psi_12 + psi_21) / np.sqrt(2.0)
    elif sym_type == 'antibonding':
        return (psi_12 - psi_21) / np.sqrt(2.0)
    return psi_12
def generate_training_data():
    x1_phy = torch.rand(N_phy, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    x2_phy = torch.rand(N_phy, 1, device=device) * (y_domain[1] - y_domain[0]) + y_domain[0]
    rand_bc = torch.rand(N_bc, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    bc_neg_L = torch.full((N_bc, 1), x_domain[0], device=device)
    bc_pos_L = torch.full((N_bc, 1), x_domain[1], device=device)
    x1_bc = torch.cat([bc_neg_L, bc_pos_L, rand_bc, rand_bc], dim=0)
    x2_bc = torch.cat([rand_bc, rand_bc, bc_neg_L, bc_pos_L], dim=0)
    x1_phy_norm = normalize(x1_phy, x_domain)
    x2_phy_norm = normalize(x2_phy, y_domain)
    x1_bc_norm = normalize(x1_bc, x_domain)
    x2_bc_norm = normalize(x2_bc, y_domain)
    return {
        'phy': (x1_phy_norm, x2_phy_norm),
        'bc': (x1_bc_norm, x2_bc_norm)
    }
def Loss(pinn_model, energy_param, data, sym_type, previous_models):
    x1_phy_norm, x2_phy_norm = data['phy']
    x1_bc_norm, x2_bc_norm = data['bc']
    x1_phy_norm.requires_grad_(True)
    x2_phy_norm.requires_grad_(True)
    def Potential_V(x1_true, x2_true):
        v_e1_n = -1.0/torch.sqrt((x1_true - R_nuc/2)**2 + a_soft**2) - 1.0/torch.sqrt((x1_true + R_nuc/2)**2 + a_soft**2)
        v_e2_n = -1.0/torch.sqrt((x2_true - R_nuc/2)**2 + a_soft**2) - 1.0/torch.sqrt((x2_true + R_nuc/2)**2 + a_soft**2)
        v_ee = 1.0/torch.sqrt((x1_true - x2_true)**2 + a_soft**2)
        v_nn = 1.0/R_nuc
        return v_e1_n + v_e2_n + v_ee + v_nn
    psi_phy = get_symmetrized_psi(pinn_model, x1_phy_norm, x2_phy_norm, sym_type)
    psi_real, psi_imag = psi_phy.real, psi_phy.imag
    d1_norm_d1 = 2.0 / (x_domain[1] - x_domain[0])
    grads_real_x1 = torch.autograd.grad(psi_real.sum(), x1_phy_norm, create_graph=True)[0]
    grads2_real_x1 = torch.autograd.grad(grads_real_x1.sum(), x1_phy_norm, create_graph=True)[0]
    grads_real_x2 = torch.autograd.grad(psi_real.sum(), x2_phy_norm, create_graph=True)[0]
    grads2_real_x2 = torch.autograd.grad(grads_real_x2.sum(), x2_phy_norm, create_graph=True)[0]
    grads_imag_x1 = torch.autograd.grad(psi_imag.sum(), x1_phy_norm, create_graph=True)[0]
    grads2_imag_x1 = torch.autograd.grad(grads_imag_x1.sum(), x1_phy_norm, create_graph=True)[0]
    grads_imag_x2 = torch.autograd.grad(psi_imag.sum(), x2_phy_norm, create_graph=True)[0]
    grads2_imag_x2 = torch.autograd.grad(grads_imag_x2.sum(), x2_phy_norm, create_graph=True)[0]
    psi_real_xx = (grads2_real_x1 + grads2_real_x2) * (d1_norm_d1**2)
    psi_imag_xx = (grads2_imag_x1 + grads2_imag_x2) * (d1_norm_d1**2)
    V = Potential_V(denormalize(x1_phy_norm, x_domain), denormalize(x2_phy_norm, y_domain))
    residual_real = -0.5 * psi_real_xx + V * psi_real - energy_param * psi_real
    residual_imag = -0.5 * psi_imag_xx + V * psi_imag - energy_param * psi_imag
    loss_phy = torch.mean(residual_real**2) + torch.mean(residual_imag**2)
    psi_bc = get_symmetrized_psi(pinn_model, x1_bc_norm, x2_bc_norm, sym_type)
    loss_bc = torch.mean(psi_bc.real**2) + torch.mean(psi_bc.imag**2) 
    x_int = torch.linspace(x_domain[0], x_domain[1], N_norm_int)
    X1_int, X2_int = torch.meshgrid(x_int, x_int, indexing='ij')
    X1_flat = X1_int.reshape(-1, 1).to(device)
    X2_flat = X2_int.reshape(-1, 1).to(device)
    X1_norm = normalize(X1_flat, x_domain)
    X2_norm = normalize(X2_flat, y_domain)
    dx = (x_domain[1] - x_domain[0]) / (N_norm_int - 1)
    area_element = dx * dx
    psi_int = get_symmetrized_psi(pinn_model, X1_norm, X2_norm, sym_type)
    prob = psi_int.real**2 + psi_int.imag**2
    integral = torch.sum(prob) * area_element
    loss_norm = (integral - 1.0)**2
    loss_ortho = torch.tensor(0.0, device=device)
    if previous_models:
        same_sym_models = [m for m, s in previous_models if s == sym_type]
        if same_sym_models:
            for prev_model in same_sym_models:
                with torch.no_grad():
                    psi_prev_int = get_symmetrized_psi(prev_model, X1_norm, X2_norm, sym_type)
                overlap_real = torch.sum(psi_prev_int.real * psi_int.real + psi_prev_int.imag * psi_int.imag) * area_element
                overlap_imag = torch.sum(psi_prev_int.real * psi_int.imag - psi_prev_int.imag * psi_int.real) * area_element
                loss_ortho += (overlap_real**2 + overlap_imag**2)     
    loss_total = (Loss_Weight['phy'] * loss_phy +
                  Loss_Weight['bc'] * loss_bc +
                  Loss_Weight['norm'] * loss_norm +
                  Loss_Weight['ortho'] * loss_ortho)             
    return loss_total, loss_phy, loss_bc, loss_norm, loss_ortho
def train_molecule_state(state_idx, previous_models, LOAD_MODEL=False):
    sym_type = 'bonding' if state_idx % 2 == 0 else 'antibonding'
    pinn_model = SoftConstraintPINN(omega=30.0, out_dim=2, domain1=x_domain, domain2=y_domain).to(device)
    initial_E = -1.0 + 0.1 * state_idx 
    E = nn.Parameter(torch.tensor([initial_E], device=device)) 
    all_params = list(pinn_model.parameters()) + [E]
    optimizer = torch.optim.Adam(all_params, lr=Learning_Rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6)
    MODEL_PATH = f"PINN_H2_State_{state_idx}_{sym_type}.pth"
    ALPHA = 0.75
    loss_total_history, loss_phy_history, loss_bc_history = [], [], []
    loss_norm_history, loss_ortho_history = [], []
    LR_history, C_history, E_history = [], [], []
    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        pinn_model.load_state_dict(checkpoint['model_state_dict'])
        E.data = checkpoint['E']
    else:
        for epoch in range(Epoch + 1):
            data = generate_training_data()
            optimizer.zero_grad()       
            loss_total, loss_phy, loss_bc, loss_norm, loss_ortho = Loss(pinn_model, E, data, sym_type, previous_models)
            loss_total.backward()           
            current_lr = optimizer.param_groups[0]['lr']
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) 
                             for p in all_params if p.grad is not None]), 
                2
            )
            dynamic_C = (ALPHA / (current_lr + 1e-8)) * (torch.sqrt(loss_total.detach()) / (total_norm + 1e-8))
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=dynamic_C)      
            optimizer.step()
            scheduler.step(loss_total)
            loss_total_history.append(loss_total.item())
            loss_phy_history.append(loss_phy.item())
            loss_bc_history.append(loss_bc.item())
            loss_norm_history.append(loss_norm.item())
            loss_ortho_history.append(loss_ortho.item())
            LR_history.append(current_lr)
            C_history.append(dynamic_C.item() if isinstance(dynamic_C, torch.Tensor) else dynamic_C)
            E_history.append(E.item())
            if epoch % 500 == 0:
                print(f"Epoch[{epoch}/{Epoch}], LR: {current_lr:.2e}, E_calc: {E.item():.4f} a.u., "
                      f"Total: {loss_total.item():.4e}, Phy: {loss_phy.item():.4e}, "
                      f"Norm: {loss_norm.item():.4e}, Ortho: {loss_ortho.item():.4e}")      
        torch.save({
            'model_state_dict': pinn_model.state_dict(),
            'E': E.data
        }, MODEL_PATH)
    pinn_model.eval()
    pts_1 = np.linspace(x_domain[0], x_domain[1], 100)
    pts_2 = np.linspace(y_domain[0], y_domain[1], 100)
    Grid_1, Grid_2 = np.meshgrid(pts_1, pts_2, indexing='ij')
    x1_torch = torch.tensor(Grid_1, dtype=torch.float32).view(-1, 1).to(device)
    x2_torch = torch.tensor(Grid_2, dtype=torch.float32).view(-1, 1).to(device)
    x1_norm = normalize(x1_torch, x_domain)
    x2_norm = normalize(x2_torch, y_domain)
    with torch.no_grad():
        psi_pred = get_symmetrized_psi(pinn_model, x1_norm, x2_norm, sym_type)
    Psi = psi_pred.reshape(Grid_1.shape).cpu()
    P = (torch.abs(Psi)**2).numpy()
    Psi_real = Psi.real.numpy()
    Psi_imag = Psi.imag.numpy()
    df0 = pd.DataFrame({
        'Total Loss': loss_total_history,
        'Physics Loss': loss_phy_history,
        'Ortho Loss': loss_ortho_history, 
        'BC Loss': loss_bc_history,
        'Norm Loss': loss_norm_history,
        'Learning Rate' : LR_history,
        'Dynamic Clip C': C_history,
        'Energy (a.u.)': E_history 
    })
    prefix = f"State_{state_idx}_{sym_type}"
    df0.to_excel(f"LOSS_{prefix}.xlsx", index=False)
    pd.DataFrame(P).to_excel(f'P_{prefix}.xlsx', index=False, header=False)
    pd.DataFrame(Psi_real).to_excel(f'Psi_real_{prefix}.xlsx', index=False, header=False)
    pd.DataFrame(Psi_imag).to_excel(f'Psi_imag_{prefix}.xlsx', index=False, header=False)
    print(f"Data saved for State {state_idx}. Final Energy: {E.item():.4f} a.u.")
    return pinn_model, sym_type
if __name__ == "__main__":
    previous_models = []
    num_total_states = 2
    for state_idx in range(num_total_states):
        trained_model, sym_type = train_molecule_state(state_idx, previous_models, LOAD_MODEL=False)
        frozen_model = SoftConstraintPINN(omega=30.0, out_dim=2, domain1=x_domain, domain2=y_domain).to(device)
        frozen_model.load_state_dict(trained_model.state_dict())
        frozen_model.eval()
        for p in frozen_model.parameters():
            p.requires_grad = False
        previous_models.append((frozen_model, sym_type))