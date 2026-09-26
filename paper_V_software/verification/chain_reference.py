import torch
import struct

B, N, D, DFF = 2, 4, 8, 16
EPS = 1e-6

def load(name, shape):
    n = 1
    for s in shape:
        n *= s
    with open(f"data/{name}.bin", "rb") as f:
        raw = f.read()
    arr = struct.unpack(f"{n}d", raw)
    return torch.tensor(arr, dtype=torch.float64).reshape(shape)

def save(name, t):
    arr = t.detach().numpy().astype('float64').flatten()
    with open(f"data/{name}.bin", "wb") as f:
        f.write(struct.pack(f"{len(arr)}d", *arr))

X = load("X", (B, N, D))
gamma = load("gamma", (D,))
Wg = load("Wg", (D, DFF))
Wu = load("Wu", (D, DFF))
Wd = load("Wd", (DFF, D))

r = torch.sqrt((X * X).mean(dim=-1, keepdim=True) + EPS)
Z = gamma * (X / r)

U = Z @ Wg
V = Z @ Wu
H = torch.nn.functional.silu(U) * V
F = H @ Wd

save("chain_ref_Z", Z)
save("chain_ref_F", F)

print("Chain reference (RMSNorm -> gated MLP) written.")
print("Z[0,0]:", Z[0,0].tolist())
print("F[0,0]:", F[0,0].tolist())
