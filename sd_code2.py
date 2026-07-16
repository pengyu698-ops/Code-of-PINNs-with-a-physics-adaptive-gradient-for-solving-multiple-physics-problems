#Standard PINN Architecture for Solving Flow Around a Cylinder Code
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
U_inf = 1.0                         
nu = 0.025                          
R = 0.5                           
x_c, y_c = 0.0, 0.0              
x_domain = [-5.0, 10.0]         
y_domain = [-5.0, 5.0]            
N_phy = 12000              
N_bc = 4000                       
N_cyl = 3000                      
Loss_Weight = {'phy': 1.0, 'bc': 2.0, 'cyl': 10.0} 
Learning_Rate = 1e-3   
torch.manual_seed(3)
Epoch = 30000         
def normalize(data, domain):
    return 2.0 * (data - domain[0]) / (domain[1] - domain[0]) - 1.0
def denormalize(data, domain):
    return 0.5 * (data + 1.0) * (domain[1] - domain[0]) + domain[0]
class SoftConstraintPINN(nn.Module):
    def __init__(self): 
        super(SoftConstraintPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 2)
        )
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x, y):
        net_input = torch.cat([x, y], dim=1)
        net_output = self.net(net_input)
        psi = net_output[:, [0]]
        p = net_output[:, [1]]
        return psi, p
def generate_training_data():
    x_phy_raw = torch.rand(N_phy * 2, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    y_phy_raw = torch.rand(N_phy * 2, 1, device=device) * (y_domain[1] - y_domain[0]) + y_domain[0]
    mask = (x_phy_raw - x_c)**2 + (y_phy_raw - y_c)**2 > R**2
    x_phy = x_phy_raw[mask][:N_phy].view(-1, 1)
    y_phy = y_phy_raw[mask][:N_phy].view(-1, 1)
    N_each_bc = N_bc // 4
    x_in = torch.full((N_each_bc, 1), x_domain[0], device=device)
    y_in = torch.rand(N_each_bc, 1, device=device) * (y_domain[1] - y_domain[0]) + y_domain[0]
    x_out = torch.full((N_each_bc, 1), x_domain[1], device=device)
    y_out = torch.rand(N_each_bc, 1, device=device) * (y_domain[1] - y_domain[0]) + y_domain[0]
    x_top = torch.rand(N_each_bc, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    y_top = torch.full((N_each_bc, 1), y_domain[1], device=device)
    x_bot = torch.rand(N_each_bc, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    y_bot = torch.full((N_each_bc, 1), y_domain[0], device=device)
    x_wall = torch.cat([x_top, x_bot], dim=0)
    y_wall = torch.cat([y_top, y_bot], dim=0)
    theta = torch.rand(N_cyl, 1, device=device) * 2 * np.pi
    x_cyl = x_c + R * torch.cos(theta)
    y_cyl = y_c + R * torch.sin(theta)
    data = {
        'phy': (normalize(x_phy, x_domain), normalize(y_phy, y_domain)),
        'in': (normalize(x_in, x_domain), normalize(y_in, y_domain)),
        'out': (normalize(x_out, x_domain), normalize(y_out, y_domain)),
        'wall': (normalize(x_wall, x_domain), normalize(y_wall, y_domain)),
        'cyl': (normalize(x_cyl, x_domain), normalize(y_cyl, y_domain))
    }
    return data
def Loss(pinn_model, data):
    x_phy_norm, y_phy_norm = data['phy']
    x_in_norm, y_in_norm = data['in']
    x_out_norm, y_out_norm = data['out']
    x_wall_norm, y_wall_norm = data['wall']
    x_cyl_norm, y_cyl_norm = data['cyl']
    dx_norm_dx = 2.0 / (x_domain[1] - x_domain[0])
    dy_norm_dy = 2.0 / (y_domain[1] - y_domain[0])
    def get_uv_p(x_n, y_n):
        x_n.requires_grad_(True)
        y_n.requires_grad_(True)
        psi, p = pinn_model(x_n, y_n)
        u =  torch.autograd.grad(psi.sum(), y_n, create_graph=True)[0] * dy_norm_dy
        v = -torch.autograd.grad(psi.sum(), x_n, create_graph=True)[0] * dx_norm_dx
        return u, v, p, x_n, y_n
    u_phy, v_phy, p_phy, x_n, y_n = get_uv_p(x_phy_norm, y_phy_norm)
    u_x = torch.autograd.grad(u_phy.sum(), x_n, create_graph=True)[0] * dx_norm_dx
    u_y = torch.autograd.grad(u_phy.sum(), y_n, create_graph=True)[0] * dy_norm_dy
    v_x = torch.autograd.grad(v_phy.sum(), x_n, create_graph=True)[0] * dx_norm_dx
    v_y = torch.autograd.grad(v_phy.sum(), y_n, create_graph=True)[0] * dy_norm_dy
    u_xx = torch.autograd.grad(u_x.sum(), x_n, create_graph=True)[0] * dx_norm_dx
    u_yy = torch.autograd.grad(u_y.sum(), y_n, create_graph=True)[0] * dy_norm_dy
    v_xx = torch.autograd.grad(v_x.sum(), x_n, create_graph=True)[0] * dx_norm_dx
    v_yy = torch.autograd.grad(v_y.sum(), y_n, create_graph=True)[0] * dy_norm_dy
    p_x = torch.autograd.grad(p_phy.sum(), x_n, create_graph=True)[0] * dx_norm_dx
    p_y = torch.autograd.grad(p_phy.sum(), y_n, create_graph=True)[0] * dy_norm_dy
    f_u = u_phy * u_x + v_phy * u_y + p_x - nu * (u_xx + u_yy)
    f_v = u_phy * v_x + v_phy * v_y + p_y - nu * (v_xx + v_yy)
    loss_phy = torch.mean(f_u**2) + torch.mean(f_v**2)
    u_in, v_in, _, _, _ = get_uv_p(x_in_norm, y_in_norm)
    loss_in = torch.mean((u_in - U_inf)**2) + torch.mean(v_in**2)
    _, _, p_out, _, _ = get_uv_p(x_out_norm, y_out_norm)
    loss_out = torch.mean(p_out**2)
    u_wall, v_wall, _, _, _ = get_uv_p(x_wall_norm, y_wall_norm)
    loss_wall = torch.mean((u_wall - U_inf)**2) + torch.mean(v_wall**2)
    loss_bc = loss_in + loss_out + loss_wall
    u_cyl, v_cyl, _, _, _ = get_uv_p(x_cyl_norm, y_cyl_norm)
    loss_cyl = torch.mean(u_cyl**2) + torch.mean(v_cyl**2)
    loss_total = (Loss_Weight['phy'] * loss_phy +
                  Loss_Weight['bc'] * loss_bc +
                  Loss_Weight['cyl'] * loss_cyl)         
    return loss_total, loss_phy, loss_bc, loss_cyl
pinn_model = SoftConstraintPINN().to(device)
optimizer = torch.optim.Adam(pinn_model.parameters(), lr=Learning_Rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6)
LOAD_MODEL = False
MODEL_PATH = "PINN_Cylinder_steady.pth"
loss_total_history, loss_phy_history, loss_bc_history, loss_cyl_history, LR, C_history = [],[],[],[],[],[]
if LOAD_MODEL and os.path.exists(MODEL_PATH):
    pinn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    for epoch in range(Epoch + 1):
        data = generate_training_data()
        optimizer.zero_grad()
        loss_total, loss_phy, loss_bc, loss_cyl = Loss(pinn_model, data)
        loss_total.backward()
        current_lr = optimizer.param_groups[0]['lr']
        clip_value = 10.0
        torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), max_norm=clip_value) 
        optimizer.step()
        scheduler.step(loss_total)
        loss_total_history.append(loss_total.item())
        loss_phy_history.append(loss_phy.item())
        loss_bc_history.append(loss_bc.item())
        loss_cyl_history.append(loss_cyl.item())
        LR.append(current_lr)
        C_history.append(clip_value)
        if epoch % 500 == 0:
            print(f"Epoch[{epoch}/{Epoch}], LR: {current_lr:.2e}, "
                  f"Total: {loss_total.item():.4e}, Phy: {loss_phy.item():.4e}, "
                  f"BC: {loss_bc.item():.4e}, Cyl: {loss_cyl.item():.4e},"
                  f"Clip_C: {clip_value:.1f}")
    torch.save(pinn_model.state_dict(), MODEL_PATH)
t_pts = np.linspace(x_domain[0], x_domain[1], 150)
y_pts = np.linspace(y_domain[0], y_domain[1], 100)
X_grid, Y_grid = np.meshgrid(t_pts, y_pts)
x_torch = torch.tensor(X_grid, dtype=torch.float32).view(-1, 1).requires_grad_(True).to(device)
y_torch = torch.tensor(Y_grid, dtype=torch.float32).view(-1, 1).requires_grad_(True).to(device)
x_norm = normalize(x_torch, x_domain)
y_norm = normalize(y_torch, y_domain)
dx_norm_dx = 2.0 / (x_domain[1] - x_domain[0])
dy_norm_dy = 2.0 / (y_domain[1] - y_domain[0])
psi_pred, p_pred = pinn_model(x_norm, y_norm)
u_pred =  torch.autograd.grad(psi_pred.sum(), y_norm, create_graph=False, retain_graph=True)[0] * dy_norm_dy
v_pred = -torch.autograd.grad(psi_pred.sum(), x_norm, create_graph=False)[0] * dx_norm_dx
U = u_pred.detach().reshape(X_grid.shape).cpu().numpy()
V = v_pred.detach().reshape(X_grid.shape).cpu().numpy()
P = p_pred.detach().reshape(X_grid.shape).cpu().numpy()
mask = (X_grid - x_c)**2 + (Y_grid - y_c)**2 <= R**2
U[mask] = np.nan
V[mask] = np.nan
P[mask] = np.nan
df_loss = pd.DataFrame({
    'Total Loss': loss_total_history,
    'Physics Loss': loss_phy_history,
    'BC Loss': loss_bc_history,
    'Cyl Loss': loss_cyl_history,
    'Learning Rate' : LR,
    'Dynamic Clip C': C_history
})
df_loss.to_excel("LOSS_Cylinder.xlsx", index=False)
pd.DataFrame(U).to_excel('Velocity_U.xlsx', index=False, header=False)
pd.DataFrame(V).to_excel('Velocity_V.xlsx', index=False, header=False)
pd.DataFrame(P).to_excel('Pressure_P.xlsx', index=False, header=False)