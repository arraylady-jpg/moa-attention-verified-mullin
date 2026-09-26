import torch
import struct

torch.manual_seed(0)

B, N, D, DFF = 2, 4, 8, 16

def save(name, t):
    arr = t.detach().numpy().astype('float64').flatten()
    with open(f"data/{name}.bin", "wb") as f:
        f.write(struct.pack(f"{len(arr)}d", *arr))

import os
os.makedirs("data", exist_ok=True)

X = torch.randn(B, N, D, dtype=torch.float64) * 0.5
gamma = torch.randn(D, dtype=torch.float64) * 0.5 + 1.0
GZ = torch.randn(B, N, D, dtype=torch.float64) * 0.1

Wg = torch.randn(D, DFF, dtype=torch.float64) * 0.3
Wu = torch.randn(D, DFF, dtype=torch.float64) * 0.3
Wd = torch.randn(DFF, D, dtype=torch.float64) * 0.3
GF = torch.randn(B, N, D, dtype=torch.float64) * 0.1

save("X", X)
save("gamma", gamma)
save("GZ", GZ)
save("Wg", Wg)
save("Wu", Wu)
save("Wd", Wd)
save("GF", GF)

print("Inputs generated:", B, N, D, DFF)
