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
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR, ExponentialLR
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# ==================== Loss function ====================
def loss_fn(model, X_norm, Y_norm, stats, X_raw):
    Y_pred = model(X_norm)

    v_true, q_true = Y_norm[:, 0], Y_norm[:, 1]
    v_pred, q_pred = Y_pred[:, 0], Y_pred[:, 1]

    v_loss = ((v_pred - v_true) ** 2).mean()
    q_loss = ((q_pred - q_true) ** 2).mean()

    loss = v_loss + 5*q_loss

    return loss, v_loss.item(), q_loss.item()

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

# ==================== Model training  ====================
def train_model(model, X_train, Y_train, stats, X_raw, epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    train_loss_history = []

    for epoch in range(epochs):            
        model.train()
        optimizer.zero_grad()
        loss, V_loss, Q_loss = loss_fn(model, X_train, Y_train, stats, X_raw)
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())
        
        scheduler.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Train: {loss.item():.4e} ")

    return train_loss_history, V_loss, Q_loss

# ==================== Model evaluation ====================
def evaluate_model(model, X_test, Y_test, stats, X_raw):
    model.eval()
    with torch.no_grad():

        test_loss, V_loss, Q_loss = loss_fn(model, X_test, Y_test, stats, X_raw)
        
        
        Y_pred_norm = model(X_test)
        Y_pred = Y_pred_norm * stats['Y_std'] + stats['Y_mean']
        Y_true = Y_test * stats['Y_std'] + stats['Y_mean']

        
        Y_pred = torch.expm1(Y_pred) - stats['Y_offset']
        Y_true = torch.expm1(Y_true) - stats['Y_offset']
        
        Y_pred_np = Y_pred.numpy()
        Y_true_np = Y_true.numpy()

        metrics = {}
        for i, name in enumerate(['Voltage', 'HeatFlux']):
            y_pred_i = Y_pred_np[:, i]
            y_true_i = Y_true_np[:, i]

            metrics[name] = {
                'MSE': mean_squared_error(y_true_i, y_pred_i),
                'RMSE': mean_squared_error(y_true_i, y_pred_i) ** 0.5,
                'MAE': mean_absolute_error(y_true_i, y_pred_i),
                'R2': r2_score(y_true_i, y_pred_i)
            }

    print("\n" + "=" * 50)
    print(f" Loss: {test_loss.item():.3e})")
    for k in metrics:
        print(f"- {k:10s}: MSE = {metrics[k]['MSE']:.3e} | RMSE = {metrics[k]['RMSE']:.3e} | MAE = {metrics[k]['MAE']:.3e} | R² = {metrics[k]['R2']:.4f}")
    print("=" * 50)
    
    return test_loss.item(), V_loss, Q_loss, metrics

# ==================== Save model ====================
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model is saved to {path}")

# ==================== Main ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

csv_file = os.path.join(parent_dir, "data", "SnS.csv")
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file {csv_file} does not exist!")
    
print("\n" + "=" * 50)
X_tensor, Y_tensor, stats, X_raw = load_dataset_from_csv(csv_file)

model = TEGNet().to(device)
print("\n" + "=" * 50)
print("Starting traning...")

# 训练模型
train_loss, V_train_loss, Q_train_loss = train_model(model, X_tensor, Y_tensor, stats, X_raw, epochs=6000, lr=1e-3)

print("\n" + "=" * 50)
model_dir = os.path.join(parent_dir, "model")
os.makedirs(model_dir, exist_ok=True) 

model_path = os.path.join(model_dir, "SnS.pth")
save_model(model, path=model_path)

