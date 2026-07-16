#Physical Adaptive Gradient Clipping Sin-Tanh Hybrid PINN Architecture for Solving One-Dimensional Quantum Tunneling Code
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
V_max = 2.10                    
w = 3.0                          
x_c = 41.5               
k0 = 2.0                       
sigma = 2.0                      
x0 = 30.0                           
sigma_v = w / np.sqrt(2 * np.pi)    
x_domain = [0.0, 100.0]           
t_domain =[0.0, 10.0]               
N_phy = 10000                       
N_bc = 5000                         
N_ic = 5000                         
N_norm_time = 20                   
Loss_Weight = {'phy': 35.0, 'bc': 2.0, 'ic': 30.0, 'norm': 10.0}
Learning_Rate = 1e-3   
torch.manual_seed(3)
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
    def __init__(self, omega=30.0, out_dim=2, domain1=t_domain, domain2=x_domain): 
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
def get_complex_psi(model, t_norm, x_norm):
    net_output = model(t_norm, x_norm)
    real_part = net_output[:, [0]]
    imag_part = net_output[:, [1]]
    return torch.complex(real_part, imag_part)
def generate_training_data():
    t_phy = torch.rand(N_phy, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    N_phy_focus = N_phy // 2
    N_phy_unif = N_phy - N_phy_focus
    x_phy_unif = torch.rand(N_phy_unif, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    x_center = x0 + k0 * t_phy[N_phy_unif:]
    x_phy_focus = x_center + 3.0 * sigma * torch.randn(N_phy_focus, 1, device=device)
    x_phy_focus = torch.clamp(x_phy_focus, x_domain[0], x_domain[1])
    x_phy = torch.cat([x_phy_unif, x_phy_focus], dim=0)
    t_bc = torch.rand(N_bc, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    bc_left_val = torch.full((N_bc, 1), x_domain[0], device=device)
    bc_right_val = torch.full((N_bc, 1), x_domain[1], device=device)
    x_bc_all = torch.cat([bc_left_val, bc_right_val], dim=0)
    t_bc_all = torch.cat([t_bc] * 2, dim=0)
    N_ic_focus = N_ic // 2
    N_ic_unif = N_ic - N_ic_focus
    t_ic = torch.zeros(N_ic, 1, device=device)
    x_ic_unif = torch.rand(N_ic_unif, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    x_ic_focus = x0 + 2 * sigma * torch.randn(N_ic_focus, 1, device=device)
    x_ic_focus = torch.clamp(x_ic_focus, x_domain[0], x_domain[1])
    x_ic = torch.cat([x_ic_unif, x_ic_focus], dim=0)
    norm_factor = 1.0 / (np.pi * sigma**2)**0.25
    gaussian = norm_factor * torch.exp(-(x_ic - x0)**2 / (2 * sigma**2))
    initial_real = gaussian * torch.cos(k0 * x_ic)
    initial_imag = gaussian * torch.sin(k0 * x_ic)
    psi_ic_target = torch.complex(initial_real, initial_imag)
    t_phy_norm = normalize(t_phy, t_domain)
    x_phy_norm = normalize(x_phy, x_domain)
    t_bc_norm = normalize(t_bc_all, t_domain)
    x_bc_norm = normalize(x_bc_all, x_domain)
    t_ic_norm = normalize(t_ic, t_domain)
    x_ic_norm = normalize(x_ic, x_domain)
    data = {
        'phy': (t_phy_norm, x_phy_norm),
        'bc': (t_bc_norm, x_bc_norm),
        'ic': (t_ic_norm, x_ic_norm, psi_ic_target)
    }
    return data
def Loss(pinn_model, data):
    t_phy_norm, x_phy_norm = data['phy']
    t_bc_norm, x_bc_norm = data['bc']
    t_ic_norm, x_ic_norm, psi_ic_target = data['ic']
    t_phy_norm.requires_grad_(True)
    x_phy_norm.requires_grad_(True)
    def Potential_V(x_norm):
        x_true = denormalize(x_norm, x_domain)
        V = V_max * torch.exp(- (x_true - x_c)**2 / (2 * sigma_v**2))
        return V
    psi_phy = get_complex_psi(pinn_model, t_phy_norm, x_phy_norm)
    psi_real, psi_imag = psi_phy.real, psi_phy.imag
    grads_real_t = torch.autograd.grad(psi_real.sum(), t_phy_norm, create_graph=True)[0]
    grads_real_x = torch.autograd.grad(psi_real.sum(), x_phy_norm, create_graph=True)[0]
    grads_imag_t = torch.autograd.grad(psi_imag.sum(), t_phy_norm, create_graph=True)[0]
    grads_imag_x = torch.autograd.grad(psi_imag.sum(), x_phy_norm, create_graph=True)[0]
    grads2_real_x = torch.autograd.grad(grads_real_x.sum(), x_phy_norm, create_graph=True)[0]
    grads2_imag_x = torch.autograd.grad(grads_imag_x.sum(), x_phy_norm, create_graph=True)[0]
    d1_norm_d1 = 2.0 / (t_domain[1] - t_domain[0])
    d2_norm_d2 = 2.0 / (x_domain[1] - x_domain[0])
    psi_real_t = grads_real_t * d1_norm_d1
    psi_imag_t = grads_imag_t * d1_norm_d1
    psi_real_xx = grads2_real_x * (d2_norm_d2**2)
    psi_imag_xx = grads2_imag_x * (d2_norm_d2**2)
    V = Potential_V(x_phy_norm)
    residual_real = -psi_imag_t + 0.5 * psi_real_xx - V * psi_real
    residual_imag =  psi_real_t + 0.5 * psi_imag_xx - V * psi_imag
    loss_phy = torch.mean(residual_real**2) + torch.mean(residual_imag**2)
    psi_bc = get_complex_psi(pinn_model, t_bc_norm, x_bc_norm)
    loss_bc = torch.mean(psi_bc.real**2) + torch.mean(psi_bc.imag**2) 
    psi_ic = get_complex_psi(pinn_model, t_ic_norm, x_ic_norm)
    loss_ic = torch.mean((psi_ic.real - psi_ic_target.real)**2) + torch.mean((psi_ic.imag - psi_ic_target.imag)**2)
    N_int = 501  
    x_int = torch.linspace(x_domain[0], x_domain[1], N_int).reshape(-1, 1).to(device)
    x_int_norm = normalize(x_int, x_domain)
    t_samples = torch.rand(N_norm_time, 1).to(device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    t_samples_norm = normalize(t_samples, t_domain)
    dx = (x_domain[1] - x_domain[0]) / (N_int - 1)
    weights = torch.ones(N_int, device=device)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    weights = weights * dx / 3
    weights = weights.reshape(-1, 1)
    loss_norm = 0.0
    for i in range(N_norm_time):
        t_cur = t_samples_norm[i].repeat(N_int, 1)
        psi_cur = get_complex_psi(pinn_model, t_cur, x_int_norm)
        prob = torch.abs(psi_cur)**2
        integral = torch.sum(weights * prob)
        loss_norm += (integral - 1.0)**2
    loss_norm /= N_norm_time
    loss_total = (Loss_Weight['phy'] * loss_phy +
                  Loss_Weight['bc'] * loss_bc +
                  Loss_Weight['ic'] * loss_ic +
                  Loss_Weight['norm'] * loss_norm)        
    return loss_total, loss_phy, loss_bc, loss_ic, loss_norm
pinn_model = SoftConstraintPINN(omega=30.0, out_dim=2, domain1=t_domain, domain2=x_domain).to(device)
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=Learning_Rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6)
LOAD_MODEL = False
MODEL_PATH = "PINN_steady.pth"
ALPHA = 0.75
if LOAD_MODEL and os.path.exists(MODEL_PATH):
    pinn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    loss_total_history, loss_phy_history, loss_bc_history, loss_ic_history, loss_norm_history, LR, C_history = [], [], [], [], [], [], []
    for epoch in range(Epoch + 1):
        data = generate_training_data()
        optimizer.zero_grad() 
        loss_total, loss_phy, loss_bc, loss_ic, loss_norm = Loss(pinn_model, data)
        loss_total.backward()
        current_lr = optimizer.param_groups[0]['lr']
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) 
                         for p in pinn_model.parameters() if p.grad is not None]), 
            2
        )
        dynamic_C = (ALPHA / (current_lr + 1e-8)) * (torch.sqrt(loss_total.detach()) / (total_norm + 1e-8))
        torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), max_norm=dynamic_C) 
        optimizer.step()
        scheduler.step(loss_total)
        loss_total_history.append(loss_total.item())
        loss_phy_history.append(loss_phy.item())
        loss_bc_history.append(loss_bc.item())
        loss_ic_history.append(loss_ic.item())
        loss_norm_history.append(loss_norm.item())
        LR.append(current_lr)
        C_history.append(dynamic_C.item() if isinstance(dynamic_C, torch.Tensor) else dynamic_C)  
        if epoch % 500 == 0:
            print(f"Epoch[{epoch}/{Epoch}], LR: {current_lr:.2e}, "
                  f"Total: {loss_total.item():.4e}, Phy: {loss_phy.item():.4e}, "
                  f"BC: {loss_bc.item():.4e}, IC: {loss_ic.item():.4e}, Norm: {loss_norm.item():.4e},"
                  f"Clip_C: {dynamic_C:.4f}")      
    torch.save(pinn_model.state_dict(), MODEL_PATH)
pinn_model.eval()
pts_1 = np.linspace(t_domain[0], t_domain[1], 100)
pts_2 = np.linspace(x_domain[0], x_domain[1], 100)
Grid_1, Grid_2 = np.meshgrid(pts_1, pts_2)
t_torch = torch.tensor(Grid_1, dtype=torch.float32).view(-1, 1).to(device)
x_torch = torch.tensor(Grid_2, dtype=torch.float32).view(-1, 1).to(device)
t_norm = normalize(t_torch, t_domain)
x_norm = normalize(x_torch, x_domain)
with torch.no_grad():
    psi_pred = get_complex_psi(pinn_model, t_norm, x_norm)
Psi = psi_pred.reshape(Grid_1.shape).cpu()
P = (torch.abs(Psi)**2).numpy()
df0 = pd.DataFrame({
    'Total Loss': loss_total_history,
    'Physics Loss': loss_phy_history,
    'IC Loss': loss_ic_history,
    'BC Loss': loss_bc_history,
    'Norm Loss': loss_norm_history,
    'Learning Rate' : LR,
    'Dynamic Clip C': C_history
})
Psi_real = Psi.real.numpy()
Psi_imag = Psi.imag.numpy()
df0.to_excel("LOSS.xlsx", index=False)
pd.DataFrame(P).to_excel('P_soft_constraint.xlsx', index=False, header=False)
pd.DataFrame(Psi_real).to_excel('Psi_real_soft_constraint.xlsx', index=False, header=False)
pd.DataFrame(Psi_imag).to_excel('Psi_imag_soft_constraint.xlsx', index=False, header=False)