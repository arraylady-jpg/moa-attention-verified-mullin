import struct

TOL = 1.2e-4

def load(name, count):
    with open(f"data/{name}.bin", "rb") as f:
        raw = f.read()
    return struct.unpack(f"{count}d", raw)

def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

B, N, D, DFF = 2, 4, 8, 16
checks = [
    ("Z (RMSNorm forward)",      "ref_Z",       "c_Z",       B*N*D),
    ("G_X (RMSNorm backward)",   "ref_GX_norm", "c_GX_norm", B*N*D),
    ("G_gamma (RMSNorm backward)", "ref_Ggamma", "c_Ggamma",  D),
    ("F (gated MLP forward)",    "ref_F",       "c_F",       B*N*D),
    ("G_X (gated MLP backward)", "ref_GX_ffn",  "c_GX_ffn",  B*N*D),
    ("G_Wgate (gated MLP backward)", "ref_GWg", "c_GWg",     D*DFF),
    ("G_Wup (gated MLP backward)",   "ref_GWu", "c_GWu",     D*DFF),
    ("G_Wdown (gated MLP backward)", "ref_GWd", "c_GWd",     DFF*D),
    ("Z (RMSNorm forward)",      "ref_Z",       "f_Z",       B*N*D),
    ("G_X (RMSNorm backward)",   "ref_GX_norm", "f_GX_norm", B*N*D),
    ("G_gamma (RMSNorm backward)", "ref_Ggamma", "f_Ggamma",  D),
    ("F (gated MLP forward)",    "ref_F",       "f_F",       B*N*D),
    ("G_X (gated MLP backward)", "ref_GX_ffn",  "f_GX_ffn",  B*N*D),
    ("G_Wgate (gated MLP backward)", "ref_GWg", "f_GWg",     D*DFF),
    ("G_Wup (gated MLP backward)",   "ref_GWu", "f_GWu",     D*DFF),
    ("G_Wdown (gated MLP backward)", "ref_GWd", "f_GWd",     DFF*D),
]

print(f"{'tensor':38s} {'impl':6s} {'max abs error':>15s}  {'pass (< 1.2e-4)':>16s}")
all_pass = True
for label, ref, c, count in checks:
    impl = "C" if c.startswith("c_") else "F90"
    r = load(ref, count)
    c_ = load(c, count)
    d = maxdiff(r, c_)
    ok = d < TOL
    all_pass = all_pass and ok
    print(f"{label:38s} {impl:6s} {d:15.3e}  {'PASS' if ok else 'FAIL':>16s}")

print()
print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
