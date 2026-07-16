#CFD Code for Solving Flow Around a Cylinder
import torch
import numpy as np
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
U_inf = 1.0                     
nu = 0.025                          
R = 0.5                        
x_c, y_c = 0.0, 0.0               
x_domain = [-5.0, 10.0]            
y_domain = [-5.0, 5.0]         
nx, ny = 150, 100
x_pts = np.linspace(x_domain[0], x_domain[1], nx)
y_pts = np.linspace(y_domain[0], y_domain[1], ny)
X_grid, Y_grid = np.meshgrid(x_pts, y_pts)
dx = x_pts[1] - x_pts[0]
dy = y_pts[1] - y_pts[0]
dt = 0.005       
gamma = 2.0      
eps_p = 0.02    
max_steps = 15000  
X_torch = torch.tensor(X_grid, dtype=torch.float32, device=device)
Y_torch = torch.tensor(Y_grid, dtype=torch.float32, device=device)
mask = ((X_torch - x_c)**2 + (Y_torch - y_c)**2 <= R**2)
def ddx(f):
    return (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2 * dx)
def ddy(f):
    return (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2 * dy)
def d2dx2(f):
    return (torch.roll(f, -1, dims=1) - 2 * f + torch.roll(f, 1, dims=1)) / (dx**2)
def d2dy2(f):
    return (torch.roll(f, -1, dims=0) - 2 * f + torch.roll(f, 1, dims=0)) / (dy**2)
def advect(u, v, f):
    adv_x_cd = u * (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2 * dx)
    adv_y_cd = v * (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2 * dy)
    fx_fwd = (torch.roll(f, -1, dims=1) - f) / dx
    fx_bwd = (f - torch.roll(f, 1, dims=1)) / dx
    fy_fwd = (torch.roll(f, -1, dims=0) - f) / dy
    fy_bwd = (f - torch.roll(f, 1, dims=0)) / dy
    adv_x_up = torch.where(u > 0, u * fx_bwd, u * fx_fwd)
    adv_y_up = torch.where(v > 0, v * fy_bwd, v * fy_fwd)
    alpha = 0.2
    return (1 - alpha) * (adv_x_cd + adv_y_cd) + alpha * (adv_x_up + adv_y_up)
u = torch.ones((ny, nx), device=device) * U_inf
v = torch.zeros((ny, nx), device=device)
p = torch.zeros((ny, nx), device=device)
u[mask] = 0.0
v[mask] = 0.0
for step in range(max_steps + 1):
    u_old, v_old, p_old = u.clone(), v.clone(), p.clone()
    adv_u = advect(u_old, v_old, u_old)
    adv_v = advect(u_old, v_old, v_old)
    diff_u = nu * (d2dx2(u_old) + d2dy2(u_old))
    diff_v = nu * (d2dx2(v_old) + d2dy2(v_old))
    grad_p_x = ddx(p_old)
    grad_p_y = ddy(p_old)
    div_V = ddx(u_old) + ddy(v_old)
    diff_p = eps_p * (d2dx2(p_old) + d2dy2(p_old)) 
    u = u_old + dt * (diff_u - adv_u - grad_p_x)
    v = v_old + dt * (diff_v - adv_v - grad_p_y)
    p = p_old - dt * gamma * div_V + dt * diff_p
    u[mask] = 0.0
    v[mask] = 0.0
    u[:, 0] = U_inf
    v[:, 0] = 0.0
    p[:, 0] = p[:, 1]      
    u[:, -1] = u[:, -2]    
    v[:, -1] = v[:, -2]
    p[:, -1] = 0.0         
    u[-1, :] = U_inf       
    v[-1, :] = 0.0
    p[-1, :] = p[-2, :]
    u[0, :] = U_inf         
    v[0, :] = 0.0
    p[0, :] = p[1, :]
    if step % 1000 == 0:
        err_u = torch.max(torch.abs(u - u_old)).item()
U_np = u.cpu().numpy()
V_np = v.cpu().numpy()
P_np = p.cpu().numpy()
mask_np = mask.cpu().numpy()
U_np[mask_np] = np.nan
V_np[mask_np] = np.nan
P_np[mask_np] = np.nan
pd.DataFrame(U_np).to_excel('Velocity_U_CFD.xlsx', index=False, header=False)
pd.DataFrame(V_np).to_excel('Velocity_V_CFD.xlsx', index=False, header=False)
pd.DataFrame(P_np).to_excel('Pressure_P_CFD.xlsx', index=False, header=False)
