# cnn_camels.py
# Author: Sucharita Charan
# Date: Oct 2025
# Description: 3D CNN to predict halo LOS velocities from CAMELS data.

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

grid_f = "Grids_Mcdm_IllustrisTNG_1P_128_z=0.0.npy"
halo_f = "groups_090_1P.hdf5"
patch = 32
batch = 32
epochs = 50
lr = 1e-4
mass_cut = 1e11          # Msun/h
boxsize = 25.0           # Mpc/h
max_halo = 2500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def load_grid(gfile):
    arr = np.load(gfile, allow_pickle=True)
    grid = arr[0].astype(np.float32) if arr.ndim == 4 else arr.astype(np.float32)
    return grid
def load_halos(hfile):
    with h5py.File(hfile, "r") as f:
        g = f["Group"]
        pos = np.array(g["GroupPos"]) / 1000.0  # ckpc/h to Mpc/h
        vel = np.array(g["GroupVel"])

    
        if "Group_M_Mean200" in g:
            mass = np.array(g["Group_M_Mean200"]) * 1e10
        elif "GroupMass" in g:
            mass = np.array(g["GroupMass"]) * 1e10
        else:
            mass = np.array(g["Group_M_Crit200"]) * 1e10
    return pos, vel, mass


def extract_patch(grid, center, patch=patch, boxsize=boxsize):
    N = grid.shape[0]
    cell = boxsize / N
    idx = (center / cell - 0.5).astype(int)
    r = patch // 2
    xs = [(idx[0] + i) % N for i in range(-r, r)]
    ys = [(idx[1] + i) % N for i in range(-r, r)]
    zs = [(idx[2] + i) % N for i in range(-r, r)]
    return grid[np.ix_(xs, ys, zs)]
class HaloPatchDataset(Dataset):
    def __init__(self, grid, pos, vel, mass, mass_cut=mass_cut, max_n=max_halo):
        mask = mass > mass_cut
        pos, vel, mass = pos[mask], vel[mask], mass[mask]

        if len(pos) > max_n:
            sel = np.random.choice(len(pos), max_n, replace=False)
            pos, vel, mass = pos[sel], vel[sel], mass[sel]

        self.pos = pos
        self.vz = vel[:, 2].astype(np.float32)
        self.grid = grid
        print(f"[INFO] Loaded {len(self.pos)} halos above mass cut.")

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, idx):
        c = self.pos[idx]
        patch = extract_patch(self.grid, c)
        patch = patch / np.mean(patch) - 1.0
        patch = np.clip(patch, -1, 3)
        x = patch[np.newaxis, :, :, :].astype(np.float32)
        y = np.array(self.vz[idx], dtype=np.float32)
        return x, y
class simple3dCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
        )
        s = patch // 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * s * s * s, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(self.conv(x)).view(-1)
def train_and_eval():
    grid = load_grid(grid_f)
    pos, vel, mass = load_halos(halo_f)
    dataset =  HaloPatchDataset(grid, pos, vel, mass)

    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)

    model = simple3dCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = 1e9
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(train_ds)
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                preds.append(pred.cpu().numpy())
                truths.append(yb.cpu().numpy())
        preds, truths = np.concatenate(preds), np.concatenate(truths)
        val_loss = np.mean((preds - truths) ** 2)
        rho = pearsonr(truths, preds)[0]

        print(f"Epoch {epoch+1}/{epochs} | Train Loss = {total_loss:.4f} | Val Loss = {val_loss:.4f} | ρ = {rho:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "cnn_camels_best.pth") plt.figure(figsize=(6, 6))
    plt.scatter(truths, preds, s=6, alpha=0.5, c=preds-truths, cmap='viridis')
    lim = np.max(np.abs(truths))
    plt.plot([-lim, lim], [-lim, lim], 'r--')
    plt.xlabel("True LOS velocity [km/s]")
    plt.ylabel("Predicted LOS velocity [km/s]")
    plt.title(f"3D CNN velocity reconstruction (ρ = {rho:.3f})")
    plt.tight_layout()
    plt.savefig("cnn_velocity_recon.jpg", dpi=200)
    plt.show()



train_and_eval()
