import torch
import struct
import os

torch.manual_seed(7)

B, N, D, DFF = 2, 4, 8, 16
EPS = 1e-6

os.makedirs("data", exist_ok=True)

def save(name, t):
    arr = t.detach().numpy().astype('float64').flatten()
    with open(f"data/{name}.bin", "wb") as f:
        f.write(struct.pack(f"{len(arr)}d", *arr))

X = (torch.randn(B, N, D, dtype=torch.float64) * 0.5).requires_grad_()
gamma1 = (torch.randn(D, dtype=torch.float64) * 0.3 + 1.0).requires_grad_()
gamma2 = (torch.randn(D, dtype=torch.float64) * 0.3 + 1.0).requires_grad_()
Wq = (torch.randn(D, D, dtype=torch.float64) * 0.3).requires_grad_()
Wk = (torch.randn(D, D, dtype=torch.float64) * 0.3).requires_grad_()
Wv = (torch.randn(D, D, dtype=torch.float64) * 0.3).requires_grad_()
Wg = (torch.randn(D, DFF, dtype=torch.float64) * 0.3).requires_grad_()
Wu = (torch.randn(D, DFF, dtype=torch.float64) * 0.3).requires_grad_()
Wd = (torch.randn(DFF, D, dtype=torch.float64) * 0.3).requires_grad_()
GY = torch.randn(B, N, D, dtype=torch.float64) * 0.1

def rmsnorm(x, g):
    r = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + EPS)
    return g * (x / r), r

# ---- Forward ----
Z1, r1 = rmsnorm(X, gamma1)

Q = Z1 @ Wq
K = Z1 @ Wk
V = Z1 @ Wv

scale = 1.0 / (D ** 0.5)
scores = scale * torch.einsum('bid,bjd->bij', Q, K)
A = torch.softmax(scores, dim=-1)
AttnOut = torch.einsum('bij,bjd->bid', A, V)

R1 = X + AttnOut

Z2, r2 = rmsnorm(R1, gamma2)

U = Z2 @ Wg
Vg = Z2 @ Wu
H = torch.nn.functional.silu(U) * Vg
F = H @ Wd

Y = R1 + F

# ---- Save forward tensors ----
save("fb_X", X); save("fb_gamma1", gamma1); save("fb_gamma2", gamma2)
save("fb_Wq", Wq); save("fb_Wk", Wk); save("fb_Wv", Wv)
save("fb_Wg", Wg); save("fb_Wu", Wu); save("fb_Wd", Wd)
save("fb_GY", GY)

save("fb_ref_Z1", Z1); save("fb_ref_Q", Q); save("fb_ref_K", K); save("fb_ref_V", V)
save("fb_ref_A", A); save("fb_ref_AttnOut", AttnOut); save("fb_ref_R1", R1)
save("fb_ref_Z2", Z2); save("fb_ref_U", U); save("fb_ref_H", H); save("fb_ref_Y", Y)

# ---- Backward ----
for t in [X, gamma1, gamma2, Wq, Wk, Wv, Wg, Wu, Wd]:
    if t.grad is not None:
        t.grad = None
Y.backward(GY)

save("fb_ref_GX", X.grad)
save("fb_ref_Ggamma1", gamma1.grad)
save("fb_ref_Ggamma2", gamma2.grad)
save("fb_ref_GWq", Wq.grad)
save("fb_ref_GWk", Wk.grad)
save("fb_ref_GWv", Wv.grad)
save("fb_ref_GWg", Wg.grad)
save("fb_ref_GWu", Wu.grad)
save("fb_ref_GWd", Wd.grad)

print("Full-block PyTorch reference written.")
print("Y[0,0]:", Y[0,0].tolist())
print("GX[0,0]:", X.grad[0,0].tolist())
