#Standard PINN Architecture for Solving One-Dimensional Quantum Tunneling Code
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
torch.manual_seed(3)
v_max_val = 2.10                       
w_param = 3.0                            
xc = 41.5                        
k_wave = 2.0                          
sigma_w = 2.0                        
x_start = 30.0                          
sigma_pot = w_param / np.sqrt(2 * np.pi)    
domain_x = [0.0, 100.0]            
domain_t = [0.0, 10.0]            
n_phy = 10000                    
n_bc = 5000                      
n_ic = 5000                       
n_norm = 20                    
weight_dict = {'phy': 35.0, 'bc': 2.0, 'ic': 30.0, 'norm': 10.0}
lr_rate = 1e-3   
total_epochs = 30000         
def scale_data(data, bounds):
    return 2.0 * (data - bounds[0]) / (bounds[1] - bounds[0]) - 1.0
def unscale_data(data, bounds):
    return 0.5 * (data + 1.0) * (bounds[1] - bounds[0]) + bounds[0]
class SchrodingerNet(nn.Module):
    def __init__(self): 
        super(SchrodingerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )
    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        out = self.network(inputs)
        out_real = out[:, [0]]
        out_imag = out[:, [1]]
        return torch.complex(out_real, out_imag)
def build_dataset():
    t_phy = torch.rand(n_phy, 1, device=device) * (domain_t[1] - domain_t[0]) + domain_t[0]
    n_focus_phy = n_phy // 2
    n_unif_phy = n_phy - n_focus_phy
    x_unif_phy = torch.rand(n_unif_phy, 1, device=device) * (domain_x[1] - domain_x[0]) + domain_x[0]
    wave_center = x_start + k_wave * t_phy[n_unif_phy:]
    x_focus_phy = wave_center + 3.0 * sigma_w * torch.randn(n_focus_phy, 1, device=device)
    x_focus_phy = torch.clamp(x_focus_phy, domain_x[0], domain_x[1])
    x_phy = torch.cat([x_unif_phy, x_focus_phy], dim=0)
    t_bc = torch.rand(n_bc, 1, device=device) * (domain_t[1] - domain_t[0]) + domain_t[0]
    left_bnd = torch.full((n_bc, 1), domain_x[0], device=device)
    right_bnd = torch.full((n_bc, 1), domain_x[1], device=device)
    x_bnd = torch.cat([left_bnd, right_bnd], dim=0)
    t_bnd = torch.cat([t_bc] * 2, dim=0)
    n_focus_ic = n_ic // 2
    n_unif_ic = n_ic - n_focus_ic
    t_ic = torch.zeros(n_ic, 1, device=device)
    x_unif_ic = torch.rand(n_unif_ic, 1, device=device) * (domain_x[1] - domain_x[0]) + domain_x[0]
    x_focus_ic = x_start + 2 * sigma_w * torch.randn(n_focus_ic, 1, device=device)
    x_focus_ic = torch.clamp(x_focus_ic, domain_x[0], domain_x[1])
    x_ic = torch.cat([x_unif_ic, x_focus_ic], dim=0)
    constant = 1.0 / (np.pi * sigma_w**2)**0.25
    gauss_env = constant * torch.exp(-(x_ic - x_start)**2 / (2 * sigma_w**2))
    ini_real = gauss_env * torch.cos(k_wave * x_ic)
    ini_imag = gauss_env * torch.sin(k_wave * x_ic)
    target_psi = torch.complex(ini_real, ini_imag)
    t_phy_s = scale_data(t_phy, domain_t)
    x_phy_s = scale_data(x_phy, domain_x)
    t_bc_s = scale_data(t_bnd, domain_t)
    x_bc_s = scale_data(x_bnd, domain_x)
    t_ic_s = scale_data(t_ic, domain_t)
    x_ic_s = scale_data(x_ic, domain_x)
    dataset = {
        'phy': (t_phy_s, x_phy_s),
        'bc': (t_bc_s, x_bc_s),
        'ic': (t_ic_s, x_ic_s, target_psi)
    }
    return dataset
def calc_loss(model, dataset):
    t_p, x_p = dataset['phy']
    t_b, x_b = dataset['bc']
    t_i, x_i, psi_target = dataset['ic']
    t_p.requires_grad_(True)
    x_p.requires_grad_(True)
    def get_v(x_norm_in):
        x_real = unscale_data(x_norm_in, domain_x)
        pot = v_max_val * torch.exp(- (x_real - xc)**2 / (2 * sigma_pot**2))
        return pot
    psi_p = model(t_p, x_p)
    pr, pi = psi_p.real, psi_p.imag
    pr_t = torch.autograd.grad(pr.sum(), t_p, create_graph=True)[0]
    pr_x = torch.autograd.grad(pr.sum(), x_p, create_graph=True)[0]
    pi_t = torch.autograd.grad(pi.sum(), t_p, create_graph=True)[0]
    pi_x = torch.autograd.grad(pi.sum(), x_p, create_graph=True)[0]
    pr_xx = torch.autograd.grad(pr_x.sum(), x_p, create_graph=True)[0]
    pi_xx = torch.autograd.grad(pi_x.sum(), x_p, create_graph=True)[0]
    dt_scale = 2.0 / (domain_t[1] - domain_t[0])
    dx_scale = 2.0 / (domain_x[1] - domain_x[0])
    psi_r_t = pr_t * dt_scale
    psi_i_t = pi_t * dt_scale
    psi_r_xx = pr_xx * (dx_scale**2)
    psi_i_xx = pi_xx * (dx_scale**2)
    V_val = get_v(x_p)
    res_r = -psi_i_t + 0.5 * psi_r_xx - V_val * pr
    res_i =  psi_r_t + 0.5 * psi_i_xx - V_val * pi
    loss_p = torch.mean(res_r**2) + torch.mean(res_i**2)
    psi_b = model(t_b, x_b)
    loss_b = torch.mean(psi_b.real**2) + torch.mean(psi_b.imag**2) 
    psi_i_pred = model(t_i, x_i)
    loss_i = torch.mean((psi_i_pred.real - psi_target.real)**2) + torch.mean((psi_i_pred.imag - psi_target.imag)**2)
    n_points = 501  
    x_int = torch.linspace(domain_x[0], domain_x[1], n_points).reshape(-1, 1).to(device)
    x_int_s = scale_data(x_int, domain_x)
    t_samp = torch.rand(n_norm, 1).to(device) * (domain_t[1] - domain_t[0]) + domain_t[0]
    t_samp_s = scale_data(t_samp, domain_t)
    step = (domain_x[1] - domain_x[0]) / (n_points - 1)
    wt = torch.ones(n_points, device=device)
    wt[1:-1:2] = 4
    wt[2:-2:2] = 2
    wt = (wt * step / 3).reshape(-1, 1)
    loss_n = 0.0
    for i in range(n_norm):
        t_cur = t_samp_s[i].repeat(n_points, 1)
        psi_cur = model(t_cur, x_int_s)
        prob_density = torch.abs(psi_cur)**2
        integ = torch.sum(wt * prob_density)
        loss_n += (integ - 1.0)**2
    loss_n /= n_norm
    tot_loss = (weight_dict['phy'] * loss_p +
                weight_dict['bc'] * loss_b +
                weight_dict['ic'] * loss_i +
                weight_dict['norm'] * loss_n)
    return tot_loss, loss_p, loss_b, loss_i, loss_n
q_model = SchrodingerNet().to(device)
opt = torch.optim.Adam(q_model.parameters(), lr=lr_rate)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=500, min_lr=1e-6)
load_flag = False                            
path_model = "PINN_steady.pth"
if load_flag and os.path.exists(path_model):
    q_model.load_state_dict(torch.load(path_model, map_location=device))
else:
    hist_tot, hist_p, hist_b, hist_i, hist_n, hist_lr = [], [], [], [], [], []
    for ep in range(total_epochs + 1):
        batch = build_dataset()
        opt.zero_grad()
        l_tot, l_p, l_b, l_i, l_n = calc_loss(q_model, batch)
        l_tot.backward()
        opt.step()
        sched.step(l_tot)
        cur_lr = opt.param_groups[0]['lr']
        hist_tot.append(l_tot.item())
        hist_p.append(l_p.item())
        hist_b.append(l_b.item())
        hist_i.append(l_i.item())
        hist_n.append(l_n.item())
        hist_lr.append(cur_lr)
        if ep % 500 == 0:
            print(f"Epoch[{ep}/{total_epochs}], LR: {cur_lr:.2e}, "
                f"Total: {l_tot.item():.4e}, Phy: {l_p.item():.4e}, "
                f"BC: {l_b.item():.4e}, IC: {l_i.item():.4e}, Norm: {l_n.item():.4e}")  
    torch.save(q_model.state_dict(), path_model)
q_model.eval()
t_arr = np.linspace(domain_t[0], domain_t[1], 100)
x_arr = np.linspace(domain_x[0], domain_x[1], 100)
T_grid, X_grid = np.meshgrid(t_arr, x_arr)
x_tsr = torch.tensor(X_grid, dtype=torch.float32).view(-1, 1).to(device)
t_tsr = torch.tensor(T_grid, dtype=torch.float32).view(-1, 1).to(device)
t_scaled = scale_data(t_tsr, domain_t)
x_scaled = scale_data(x_tsr, domain_x)
with torch.no_grad():
    psi_eval = q_model(t_scaled, x_scaled)
Psi_mat = psi_eval.reshape(X_grid.shape).cpu()
P_mat = (torch.abs(Psi_mat)**2).numpy()
df_loss = pd.DataFrame({
    'Total Loss': hist_tot,
    'Physics Loss': hist_p,
    'IC Loss': hist_i,
    'BC Loss': hist_b,
    'Norm Loss': hist_n,
    'Learning Rate': hist_lr
})
Psi_r_mat = Psi_mat.real.numpy()
Psi_i_mat = Psi_mat.imag.numpy()
df_loss.to_excel("LOSS.xlsx", index=False)
pd.DataFrame(P_mat).to_excel('P_soft_constraint.xlsx', index=False, header=False)
pd.DataFrame(Psi_r_mat).to_excel('Psi_real_soft_constraint.xlsx', index=False, header=False)
pd.DataFrame(Psi_i_mat).to_excel('Psi_imag_soft_constraint.xlsx', index=False, header=False)