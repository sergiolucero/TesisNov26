import numpy as np
import os

class Scenario:
    def __init__(self, C0, v, D, L, T, nx, nt, x, t, C):
        self.C0 = C0
        self.v = v
        self.D = D
        self.L = L
        self.T = T
        self.nx = nx
        self.nt = nt
        self.x = x
        self.t = t
        self.C = C

def save_scenarios(path, scenarios):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {}
    for i, sc in enumerate(scenarios):
        payload[f"sc{i}_C0"] = np.array(sc.C0)
        payload[f"sc{i}_v"] = np.array(sc.v)
        payload[f"sc{i}_D"] = np.array(sc.D)
        payload[f"sc{i}_L"] = np.array(sc.L)
        payload[f"sc{i}_T"] = np.array(sc.T)
        payload[f"sc{i}_x"] = sc.x
        payload[f"sc{i}_t"] = sc.t
        payload[f"sc{i}_C"] = sc.C
    payload["n_scenarios"] = np.array(len(scenarios))
    np.savez_compressed(path, **payload)
    print(f"[hydrology] saved {len(scenarios)} scenario(s) -> {path}")

# Define minimal example scenarios
scenarios = []

# Create ten simple scenarios
nx, nt = 100, 50
x = np.linspace(0, 1, nx)
t = np.linspace(0, 1, nt)
C0 = 1.0
v = 1.0
D = 0.1
L = 1.0
T = 1.0

for i in range(10):
    C = np.sin(np.pi * np.outer(t, x) + i)
    scenario = Scenario(C0=C0, v=v, D=D, L=L, T=T, nx=nx, nt=nt, x=x, t=t, C=C)
    scenarios.append(scenario)

# Save scenario dataset as expected by FNO training
save_scenarios('out/dataset.npz', scenarios)  

print("Structured sample dataset with 10 scenarios saved to out/dataset.npz")
