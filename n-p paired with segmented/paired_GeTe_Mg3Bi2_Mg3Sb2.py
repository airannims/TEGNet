#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import griddata
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# ==================== Data load and process ====================
def load_dataset_from_csv(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file {csv_file} not existed！")

    X_columns = ['a', 'b', 'c', 'Tc', 'Th', 'Ih']
    Y_columns = ['V', 'Q']

    df = pd.read_csv(csv_file)
    X_cached = df[X_columns].copy().values
    Y_cached = df[Y_columns].copy().values

    X_cached[:, 5] = np.log1p(X_cached[:, 5])
    Y_offset = np.zeros((1, Y_cached.shape[1]))
    for i in range(Y_cached.shape[1]):
        min_val = Y_cached[:, i].min()
        if min_val <= 0:
            Y_offset[0, i] = -min_val + 1e-3
            Y_cached[:, i] += Y_offset[0, i]
        Y_cached[:, i] = np.log1p(Y_cached[:, i])

    X_mean = X_cached.mean(axis=0, keepdims=True)
    X_std = X_cached.std(axis=0, keepdims=True)
    Y_mean = Y_cached.mean(axis=0, keepdims=True)
    Y_std = Y_cached.std(axis=0, keepdims=True)

    X_norm = (X_cached - X_mean) / X_std
    Y_norm = (Y_cached - Y_mean) / Y_std

    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)
    X_cached_tensor = torch.tensor(X_cached, dtype=torch.float32)

    stats = {
        'X_mean': torch.tensor(X_mean, dtype=torch.float32),
        'X_std': torch.tensor(X_std, dtype=torch.float32),
        'Y_mean': torch.tensor(Y_mean, dtype=torch.float32),
        'Y_std': torch.tensor(Y_std, dtype=torch.float32),
        'Y_offset': torch.tensor(Y_offset, dtype=torch.float32)
    }

    print(f"Load success：{X_tensor.shape[0]} sample size in total")
    return X_tensor, Y_tensor, stats, X_cached_tensor

# ==================== TEGNet ====================
class TEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

def load_model(path):
    model = TEGNet()
    state_dict = torch.load(path, weights_only=True)  # 明确只加载权重
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model is loaded from {path}")
    return model

# ==================== Simulation function ====================
def predict(a, b, c, Tc, Th, Ih, model, stats):
    X_ori = np.array([[a,b,c,Tc,Th,Ih]], dtype=np.float32)
    X_ori[:, 5] = np.log1p(X_ori[:, 5])
    X_norm = (X_ori - stats['X_mean'].numpy()) / stats['X_std'].numpy()
    X_input = torch.tensor(X_norm, dtype=torch.float32)
    Y_norm_pred = model(X_input)
    Y_pred_phys = Y_norm_pred * stats['Y_std'] + stats['Y_mean']
    Y_pred_phys = torch.expm1(Y_pred_phys) - stats['Y_offset']
    V_pred = Y_pred_phys[0,0]
    q_pred = Y_pred_phys[0,1]
    return V_pred, q_pred

def get_material_property_fns(fixed_material):
    T_vals = np.array([t for t,_ in fixed_material[0]])
    sigma_vals   = np.array([v for _,v in fixed_material[0]])
    kappa_vals   = np.array([v for _,v in fixed_material[1]])
    Seebeck_vals = np.array([v for _,v in fixed_material[2]])

    def linear_extrapolate(x, x0, y0, x1, y1):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def spline_with_linear_extrap(spline, x_data, y_data):
        def wrapped(x):
            x = np.asarray(x)
            y = spline(x)
            out = y.copy()
            left_mask = x < x_data[0]
            if np.any(left_mask):
                out[left_mask] = linear_extrapolate(x[left_mask], x_data[0], y_data[0], x_data[1], y_data[1])
            right_mask = x > x_data[-1]
            if np.any(right_mask):
                out[right_mask] = linear_extrapolate(x[right_mask], x_data[-2], y_data[-2], x_data[-1], y_data[-1])
            return out
        return wrapped

    sigma_fn   = spline_with_linear_extrap(CubicSpline(T_vals, sigma_vals, extrapolate=False), T_vals, sigma_vals)
    kappa_fn   = spline_with_linear_extrap(CubicSpline(T_vals, kappa_vals, extrapolate=False), T_vals, kappa_vals)
    Seebeck_fn = spline_with_linear_extrap(CubicSpline(T_vals, Seebeck_vals, extrapolate=False), T_vals, Seebeck_vals)

    return sigma_fn, kappa_fn, Seebeck_fn

def compute_ih_max(a, b, c, Tc, Th, sigma_fn, Seebeck_fn):

    U, _ = quad(Seebeck_fn, Tc, Th)

    def resistivity(T): return 1.0 / sigma_fn(T)
    rou_int, _ = quad(resistivity, Tc, Th)
    rou_avg = rou_int / (Th - Tc)
    L = c * 1e-3  # mm → m
    A = a * b * 1e-6  # mm² → m²
    R = rou_avg * (L / A)

    Ih_max = U / R if R > 0 else 0.0
    Ih_max = max(Ih_max, 0.0)

    return Ih_max, U, R  

def scan_current_sweep(model, a, b, c, Tc, Th, I_range, stats):
    V_list, q_list, P_list, efficiency_list, I_list = [], [], [], [], []
    
    for Ih in I_range:
        X_ori = np.array([[a,b,c,Tc,Th,Ih]], dtype=np.float32)
        X_ori[:, 5] = np.log1p(X_ori[:, 5])
        X_norm = (X_ori - stats['X_mean'].numpy()) / stats['X_std'].numpy()
        X_input = torch.tensor(X_norm, dtype=torch.float32)
        Y_norm_pred = model(X_input)
        Y_pred_phys = Y_norm_pred * stats['Y_std'] + stats['Y_mean']
        Y_pred_phys = torch.expm1(Y_pred_phys) - stats['Y_offset']
        V_pred = Y_pred_phys[0,0]
        q_pred = Y_pred_phys[0,1]
    
        V = V_pred.item()
        if V < 0:
            continue

        q = q_pred.item()
        P = V * Ih
        eff = P / (P + q) * 100

        V_list.append(V)
        q_list.append(q)
        P_list.append(P)
        efficiency_list.append(eff)
        I_list.append(Ih)

    if not P_list:
        return V_list, q_list, P_list, efficiency_list, I_list, None, None

    P_max = max(P_list)
    eff_max = max(efficiency_list)

    return V_list, q_list, P_list, efficiency_list, I_list, P_max, eff_max

# ==================== Temperature scan ====================
def temperature_param_scan(model, sigma_fn, Seebeck_fn, Tc, Th_list, a, b, c, stats):
    Th_valid, P_max_list, eff_max_list = [], [], []

    for Th in Th_list:
        if Th <= Tc:
            continue
        Ih_max = compute_ih_max(a, b, c, Tc, Th, sigma_fn, Seebeck_fn)
        I_range = np.linspace(0, Ih_max, 50)  
        V_list, q_list, P_list, eff_list, I_list, P_max, eff_max = scan_current_sweep(model, a, b, c, Tc, Th, I_range, stats)
        if P_max is None or eff_max is None:
            continue

        Th_valid.append(Th)
        P_max_list.append(P_max)
        eff_max_list.append(eff_max)

    return Th_valid, P_max_list, eff_max_list

# ==================== Length scan ====================
def find_interface_temperature(model1, model2, stats1, stats2, a, b, c, Tc, Th, Ih, c1):

    c2 = c - c1

    def residual(T):
        V1, Qc1 = predict(a, b, c1, T, Th, Ih, model1, stats1)
        V2, Qc2 = predict(a, b, c2, Tc, T, Ih, model2, stats2)

        V1 = V1.detach().cpu().numpy().item()
        Qc1 = Qc1.detach().cpu().numpy().item()
        V2 = V2.detach().cpu().numpy().item()
        Qc2 = Qc2.detach().cpu().numpy().item()

        Qh2 = Qc2 + V2 * Ih  

        return Qc1 - Qh2  

    T_low = Tc + 1e-2
    T_high = Th - 1e-2
    
    f_low = residual(T_low)
    f_high = residual(T_high)

    if f_low * f_high > 0:
        return None

    sol = root_scalar(residual, bracket=[T_low, T_high], method='brentq', xtol=1e-3, rtol=1e-3)

    if sol.converged:
        return sol.root
    else:
        raise ValueError("Root finding failed")

# ==================== n-p pair scan ====================
def evaluate_pair(a_p, b_p, a_n, b_n, c, Tc, Th,
                  model1, model2, model3,
                  sigma_fn1, Seebeck_fn1,
                  sigma_fn2, Seebeck_fn2,
                  sigma_fn3, Seebeck_fn3,
                  stats1, stats2, stats3,
                  Ti_limit=None,  
                  num_c1=15, num_Ih=20):

    P_max_list, eff_max_list = [], []
    c1_list_real = []

    c1_list = np.linspace(0, c, num_c1)[1:-1]  

    _, U1, R1 = compute_ih_max(a_n, b_n, c, Tc, Th, sigma_fn1, Seebeck_fn1)
    _, U2, R2 = compute_ih_max(a_n, b_n, c, Tc, Th, sigma_fn2, Seebeck_fn2)
    _, U3, R3 = compute_ih_max(a_p, b_p, c, Tc, Th, sigma_fn3, Seebeck_fn3)
    Ih_max_est = (max(U1, U2) + U3) / (min(R1, R2) + R3)

    for c1 in c1_list:
        c2 = c - c1
        I_range_coarse = np.linspace(0, Ih_max_est, num_Ih)
        T_interface = None

        for Ih in I_range_coarse:
            T_try = find_interface_temperature(model1, model2, stats1, stats2, a_n, b_n, c, Tc, Th, Ih, c1)
            if T_try is None:
                break
            T_interface = T_try

        if T_interface is None:
            continue

        _, U1, R1 = compute_ih_max(a_n, b_n, c1, T_interface, Th, sigma_fn1, Seebeck_fn1)
        _, U2, R2 = compute_ih_max(a_n, b_n, c2, Tc, T_interface, sigma_fn2, Seebeck_fn2)
        _, U3, R3 = compute_ih_max(a_p, b_p, c, Tc, Th, sigma_fn3, Seebeck_fn3)
        Ih_max = (U1 + U2 + U3) / (R1 + R2 + R3)

        P_scan, eff_scan = [], []
        invalid_c1 = False

        for Ih in np.linspace(0, Ih_max, num_Ih):
            Ti = find_interface_temperature(
                model1, model2, stats1, stats2,
                a_n, b_n, c, Tc, Th, Ih, c1
            )

            if Ti_limit is not None and Ti > Ti_limit:
                invalid_c1 = True
                P_scan.clear()
                eff_scan.clear()
                break

            V1, Qc1 = predict(a_n, b_n, c1, Ti, Th, Ih, model1, stats1)
            V2, Qc2 = predict(a_n, b_n, c2, Tc, Ti, Ih, model2, stats2)
            V3, Qc3 = predict(a_p, b_p, c, Tc, Th, Ih, model3, stats3)

            V = V1.item() + V2.item() + V3.item()
            if V < 0:
                continue

            Q = Qc2.item() + Qc3.item()
            P = V * Ih
            eff = P / (P + Q) * 100 if (P + Q) > 1e-6 else 0

            P_scan.append(P)
            eff_scan.append(eff)

        if invalid_c1:
            continue

        if P_scan:
            P_max_list.append(max(P_scan))
            eff_max_list.append(max(eff_scan))
            c1_list_real.append(c1/c)
        else:
            P_max_list.append(np.nan)
            eff_max_list.append(np.nan)
            c1_list_real.append(np.nan)

    return c1_list_real, P_max_list, eff_max_list

def evaluate_two(a_p, b_p, a_n, b_n, c, Tc, Th, model1, model2, sigma_fn1, Seebeck_fn1, sigma_fn2, Seebeck_fn2, stats1, stats2):
    _, U1, R1 = compute_ih_max(a_p, b_p, c, Tc, Th, sigma_fn1, Seebeck_fn1)
    _, U2, R2 = compute_ih_max(a_n, b_n, c, Tc, Th, sigma_fn2, Seebeck_fn2)
    Ih_max = max(U1, U2) / min(R1, R2)
    
    I_range = np.linspace(0, Ih_max, 20)
    P_list, eff_list = [], []
    
    for Ih in I_range:
        V1, Qc1 = predict(a_p, b_p, c, Tc, Th, Ih, model1, stats1)
        V2, Qc2 = predict(a_n, b_n, c, Tc, Th, Ih, model2, stats2)
        
        V = V1.item() + V2.item()  # 两对                
        if V < 0:
            continue            
        Q = Qc1.item() + Qc2.item()
        P = V * Ih
        eff = P / (P + Q) * 100 if (P + Q) != 0 else 0
        
        P_list.append(P)
        eff_list.append(eff)
        
    return max(P_list), max(eff_list)

def scan_HApn_ratios(model1, model2, model3,
                     sigma_fn1, Seebeck_fn1,
                     sigma_fn2, Seebeck_fn2,
                     sigma_fn3, Seebeck_fn3,
                     stats1, stats2, stats3,
                     Tc, Th, Apn_total, Apn_ratio, Ti,
                     H_ratios):

    start_time = time.time()
    H_Apn_list = []
    c1_all, P_all, eff_all = [], [], []

    for H_ratio in H_ratios:
        
        c = Apn_total * H_ratio

        Ap = Apn_total * Apn_ratio / (1 + Apn_ratio)
        An = Apn_total - Ap
        a_p = b_p = np.sqrt(Ap)
        a_n = b_n = np.sqrt(An)

        c1_list, P_list, eff_list = evaluate_pair(
            a_p, b_p, a_n, b_n, c, Tc, Th,
            model1, model2, model3,
            sigma_fn1, Seebeck_fn1,
            sigma_fn2, Seebeck_fn2,
            sigma_fn3, Seebeck_fn3,
            stats1, stats2, stats3,
            Ti_limit = Ti,
            num_c1=20,
            num_Ih=20)

        P2, eff2 = evaluate_two(a_n, b_n, a_p, b_p, c, Tc, Th, model1, model3, sigma_fn1, Seebeck_fn1, sigma_fn3, Seebeck_fn3, stats1, stats3)

        c1_list.append(1) 
        P_list.append(P2)
        eff_list.append(eff2)

        for c1_val, P_val, eff_val in zip(c1_list, P_list, eff_list):
            H_Apn_list.append(H_ratio)
            c1_all.append(c1_val)
            P_all.append(P_val)
            eff_all.append(eff_val)

    end_time = time.time()
    print(f"[scan_HApn_ratios] Th={Th:.2f}, Time used: {end_time - start_time:.4f} s")

    return H_Apn_list, c1_all, P_all, eff_all

def scan_Apn_ratios(model1, model2, model3,
                     sigma_fn1, Seebeck_fn1,
                     sigma_fn2, Seebeck_fn2,
                     sigma_fn3, Seebeck_fn3,
                     stats1, stats2, stats3,
                     Tc, Th, Apn_total, H_ratio, Ti,
                     Apn_ratios):

    start_time = time.time()
    Apn_list = []
    c1_all, P_all, eff_all = [], [], []

    for Apn_ratio in Apn_ratios:

        Ap = Apn_total * Apn_ratio / (1 + Apn_ratio)
        An = Apn_total - Ap

        a_p = b_p = np.sqrt(Ap)
        a_n = b_n = np.sqrt(An)
        
        c = Apn_total * H_ratio

        c1_list, P_list, eff_list = evaluate_pair(
            a_p, b_p, a_n, b_n, c, Tc, Th,
            model1, model2, model3,
            sigma_fn1, Seebeck_fn1,
            sigma_fn2, Seebeck_fn2,
            sigma_fn3, Seebeck_fn3,
            stats1, stats2, stats3,
            Ti_limit=Ti,
            num_c1=20,
            num_Ih=20,
        )

        P2, eff2 = evaluate_two(a_n, b_n, a_p, b_p, c, Tc, Th, model1, model3, sigma_fn1, Seebeck_fn1, sigma_fn3, Seebeck_fn3, stats1, stats3)

        c1_list.append(1) 
        P_list.append(P2)
        eff_list.append(eff2)

        for c1_val, P_val, eff_val in zip(c1_list, P_list, eff_list):
            Apn_list.append(Apn_ratio)  
            c1_all.append(c1_val)
            P_all.append(P_val)
            eff_all.append(eff_val)

    end_time = time.time()
    print(f"[scan_HApn_ratios] Th={Th:.2f}, Time used: {end_time - start_time:.4f} s")

    return Apn_list, c1_all, P_all, eff_all

def export_surface_csv(x_list, y_list, z_list, csv_path="exported_data.csv"):
    x = np.array(x_list)
    y = np.array(y_list)
    z = np.array(z_list)

    df = pd.DataFrame({"Ap_An": x, "H_Apn": y, "Z": z})
    df.to_csv(csv_path, index=False)
    print(f"Surface data is saved as CSV: {csv_path}")

# ==================== Main ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

model_dir = os.path.join(parent_dir, "model")
model1 = load_model(path=os.path.join(model_dir, 'Mg3Sb2.pth'))
model2 = load_model(path=os.path.join(model_dir, 'Mg3Bi2.pth'))
model3 = load_model(path=os.path.join(model_dir, 'GeTe.pth'))

data_dir = os.path.join(parent_dir, "data")
csv_file1 = os.path.join(data_dir, "Mg3Sb2.csv")
csv_file2 = os.path.join(data_dir, "Mg3Bi2.csv")
csv_file3 = os.path.join(data_dir, "GeTe.csv")

_, _, stats1, _ = load_dataset_from_csv(csv_file1)
_, _, stats2, _ = load_dataset_from_csv(csv_file2)
_, _, stats3, _ = load_dataset_from_csv(csv_file3)

property_dir = os.path.join(parent_dir, "property")
material_csv1 = os.path.join(property_dir, "Mg3Sb2_property.csv") 
material_csv2 = os.path.join(property_dir, "Mg3Bi2_property.csv") 
material_csv3 = os.path.join(property_dir, "GeTe_property.csv") 

df1 = pd.read_csv(material_csv1)
material_property1 = [
    list(zip(df1['T'], df1['sigma'])),
    list(zip(df1['T'], df1['kappa'])),
    list(zip(df1['T'], df1['Seebeck']))
]

df2 = pd.read_csv(material_csv2)
material_property2 = [
    list(zip(df2['T'], df2['sigma'])),
    list(zip(df2['T'], df2['kappa'])),
    list(zip(df2['T'], df2['Seebeck']))
]

df3 = pd.read_csv(material_csv3)
material_property3 = [
    list(zip(df3['T'], df3['sigma'])),
    list(zip(df3['T'], df3['kappa'])),
    list(zip(df3['T'], df3['Seebeck']))
]

sigma_fn1, kappa_fn1, Seebeck_fn1 = get_material_property_fns(material_property1)
sigma_fn2, kappa_fn2, Seebeck_fn2 = get_material_property_fns(material_property2)
sigma_fn3, kappa_fn3, Seebeck_fn3 = get_material_property_fns(material_property2)

a_max = 3.3
b_max = 3.3
Tc = 293.15
Th = 723.15
Ti = 573.15
Apn_total = a_max*b_max*2

H_ratios = np.linspace(0.2, 0.5, 20)
Apn_ratio = 1
H_Apn_list, c1_list, P_max_list, eff_max_list = scan_HApn_ratios(model1, model2, model3,
                     sigma_fn1, Seebeck_fn1,
                     sigma_fn2, Seebeck_fn2,
                     sigma_fn3, Seebeck_fn3,
                     stats1, stats2, stats3,
                     Tc, Th, Apn_total, Apn_ratio, Ti,
                     H_ratios)

export_surface_csv(H_Apn_list, c1_list, P_max_list, csv_path="Pmax_HApn_Mg3Bi2+Mg3Sb2_GeTe.csv")
export_surface_csv(H_Apn_list, c1_list, eff_max_list, csv_path="effmax_HApn_Mg3Bi2+Mg3Sb2_GeTe.csv")

a_max = 3.3
b_max = 3.3
Tc = 293.15
Th = 723.15
Ti = 573.15
Apn_total = a_max*b_max*2

Apn_ratios = np.geomspace(0.2, 5, 20)
H_ratio = 0.303
Apn_list, c1_list, P_max_list, eff_max_list = scan_Apn_ratios(model1, model2, model3,
                     sigma_fn1, Seebeck_fn1,
                     sigma_fn2, Seebeck_fn2,
                     sigma_fn3, Seebeck_fn3,
                     stats1, stats2, stats3,
                     Tc, Th, Apn_total, H_ratio, Ti,
                     Apn_ratios)

export_surface_csv(Apn_list, c1_list, P_max_list, csv_path="Pmax_Apn_Mg3Bi2+Mg3Sb2_GeTe.csv")
export_surface_csv(Apn_list, c1_list, eff_max_list, csv_path="effmax_Apn_Mg3Bi2+Mg3Sb2_GeTe.csv")

