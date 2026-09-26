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

X = load("X", (B, N, D)).requires_grad_(True)
gamma = load("gamma", (D,)).requires_grad_(True)
GZ = load("GZ", (B, N, D))

Wg = load("Wg", (D, DFF)).requires_grad_(True)
Wu = load("Wu", (D, DFF)).requires_grad_(True)
Wd = load("Wd", (DFF, D)).requires_grad_(True)
GF = load("GF", (B, N, D))

# --- RMSNorm forward ---
r = torch.sqrt((X * X).mean(dim=-1, keepdim=True) + EPS)
Z = gamma * (X / r)
save("ref_Z", Z)

# --- RMSNorm backward (via autograd, upstream grad GZ) ---
X.grad = None
gamma.grad = None
Z.backward(GZ, retain_graph=True)
save("ref_GX_norm", X.grad)
save("ref_Ggamma", gamma.grad)

# --- Gated MLP forward ---
Xm = load("X", (B, N, D)).requires_grad_(True)
U = Xm @ Wg
V = Xm @ Wu
H = torch.nn.functional.silu(U) * V
F = H @ Wd
save("ref_F", F)

# --- Gated MLP backward (via autograd, upstream grad GF) ---
Xm.grad = None
Wg.grad = None
Wu.grad = None
Wd.grad = None
F.backward(GF)
save("ref_GX_ffn", Xm.grad)
save("ref_GWg", Wg.grad)
save("ref_GWu", Wu.grad)
save("ref_GWd", Wd.grad)

print("Reference outputs written.")
print("Z[0,0]:", Z[0,0].tolist())
print("F[0,0]:", F[0,0].tolist())
