#Physical Adaptive Gradient Clipping Sin-Tanh Hybrid PINN Architecture for Solving One-Dimensional Heat Conduction Code
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
alpha = 0.1                           
sigma_ic = np.sqrt(0.02)       
N_phy = 10000       
N_bc = 5000          
N_ic = 5000           
Loss_Weight = {'phy': 0.4, 'bc': 0.3, 'ic': 0.3}
x_domain = [-1.0, 1.0]
t_domain =[0.0, 1.0]  
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
    def __init__(self, omega=30.0, out_dim=1, domain1=t_domain, domain2=x_domain): 
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
def generate_training_data():
    t_phy = torch.rand(N_phy, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    N_phy_focus = N_phy // 2
    N_phy_unif = N_phy - N_phy_focus
    x_phy_unif = torch.rand(N_phy_unif, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    x_center = 0.0
    x_phy_focus = x_center + 3.0 * sigma_ic * torch.randn(N_phy_focus, 1, device=device)
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
    x_ic_focus = 0.0 + 2 * sigma_ic * torch.randn(N_ic_focus, 1, device=device)
    x_ic_focus = torch.clamp(x_ic_focus, x_domain[0], x_domain[1])
    x_ic = torch.cat([x_ic_unif, x_ic_focus], dim=0)
    u_ic_target = torch.exp(-x_ic**2 / 0.04)
    t_phy_norm = normalize(t_phy, t_domain)
    x_phy_norm = normalize(x_phy, x_domain)
    t_bc_norm = normalize(t_bc_all, t_domain)
    x_bc_norm = normalize(x_bc_all, x_domain)
    t_ic_norm = normalize(t_ic, t_domain)
    x_ic_norm = normalize(x_ic, x_domain)
    data = {
        'phy': (t_phy_norm, x_phy_norm),
        'bc': (t_bc_norm, x_bc_norm),
        'ic': (t_ic_norm, x_ic_norm, u_ic_target)
    }
    return data
def Loss(pinn_model, data):
    t_phy_norm, x_phy_norm = data['phy']
    t_bc_norm, x_bc_norm = data['bc']
    t_ic_norm, x_ic_norm, u_ic_target = data['ic']
    t_phy_norm.requires_grad_(True)
    x_phy_norm.requires_grad_(True)
    u_phy = pinn_model(t_phy_norm, x_phy_norm)
    grads_u_t = torch.autograd.grad(u_phy.sum(), t_phy_norm, create_graph=True)[0]
    grads_u_x = torch.autograd.grad(u_phy.sum(), x_phy_norm, create_graph=True)[0]
    grads2_u_x = torch.autograd.grad(grads_u_x.sum(), x_phy_norm, create_graph=True)[0]
    d1_norm_d1 = 2.0 / (t_domain[1] - t_domain[0])
    d2_norm_d2 = 2.0 / (x_domain[1] - x_domain[0])
    u_t = grads_u_t * d1_norm_d1
    u_xx = grads2_u_x * (d2_norm_d2**2)
    residual = u_t - alpha * u_xx
    loss_phy = torch.mean(residual**2)
    u_bc = pinn_model(t_bc_norm, x_bc_norm)
    loss_bc = torch.mean(u_bc**2)  
    u_ic = pinn_model(t_ic_norm, x_ic_norm)
    loss_ic = torch.mean((u_ic - u_ic_target)**2)
    loss_total = (Loss_Weight['phy'] * loss_phy +
                  Loss_Weight['bc'] * loss_bc +
                  Loss_Weight['ic'] * loss_ic)          
    return loss_total, loss_phy, loss_bc, loss_ic
pinn_model = SoftConstraintPINN(omega=30.0, out_dim=1, domain1=t_domain, domain2=x_domain).to(device)
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=Learning_Rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6)
LOAD_MODEL = False                                        
MODEL_PATH = "PINN_steady.pth"
ALPHA = 0.75                                                    
loss_total_history, loss_phy_history, loss_bc_history, loss_ic_history, LR, C_history = [], [], [], [], [], []
if LOAD_MODEL and os.path.exists(MODEL_PATH):
    pinn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    for epoch in range(Epoch + 1):
        data = generate_training_data()
        optimizer.zero_grad()  
        loss_total, loss_phy, loss_bc, loss_ic = Loss(pinn_model, data)
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
        LR.append(current_lr)
        C_history.append(dynamic_C.item() if isinstance(dynamic_C, torch.Tensor) else dynamic_C) 
        if epoch % 500 == 0:
            print(f"Epoch[{epoch}/{Epoch}], LR: {current_lr:.2e}, "
                  f"Total: {loss_total.item():.4e}, Phy: {loss_phy.item():.4e}, "
                  f"BC: {loss_bc.item():.4e}, IC: {loss_ic.item():.4e}, "
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
    u_pred = pinn_model(t_norm, x_norm) 
U = u_pred.reshape(Grid_1.shape).cpu().numpy()
df0 = pd.DataFrame({
    'Total Loss': loss_total_history,
    'Physics Loss': loss_phy_history,
    'BC Loss': loss_bc_history,
    'IC Loss': loss_ic_history,
    'Learning Rate' : LR,
    'Dynamic Clip C': C_history
})
df0.to_excel("LOSS.xlsx", index=False)
pd.DataFrame(U).to_excel('U_soft_constraint.xlsx', index=False, header=False)