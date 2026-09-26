import struct

TOL = 1.2e-4

def load(name, count):
    with open(f"data/{name}.bin", "rb") as f:
        raw = f.read()
    return struct.unpack(f"{count}d", raw)

def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

B, N, D, DFF = 2, 4, 8, 16
NT = B * N
tensors = [
    ("Y (full block forward)",  "fb_ref_Y",       NT*D),
    ("G_X",                     "fb_ref_GX",      NT*D),
    ("G_gamma1",                "fb_ref_Ggamma1", D),
    ("G_gamma2",                "fb_ref_Ggamma2", D),
    ("G_Wq",                    "fb_ref_GWq",     D*D),
    ("G_Wk",                    "fb_ref_GWk",     D*D),
    ("G_Wv",                    "fb_ref_GWv",     D*D),
    ("G_Wgate",                 "fb_ref_GWg",     D*DFF),
    ("G_Wup",                   "fb_ref_GWu",     D*DFF),
    ("G_Wdown",                 "fb_ref_GWd",     DFF*D),
]
# (label, ref_name, count) -> derive C and GPU filenames by substituting the
# fb_ref_ prefix for fb_c_ (plain C) and fb_g_ (OpenACC GPU kernel)
checks = []
for label, ref, count in tensors:
    suffix = ref[len("fb_ref_"):]
    checks.append((label, ref, f"fb_c_{suffix}", f"fb_g_{suffix}", count))

print(f"{'tensor':28s} {'C max err':>12s} {'C pass':>8s}  {'GPU max err':>12s} {'GPU pass':>8s}")
all_pass = True
for label, ref, c, g, count in checks:
    r = load(ref, count)
    c_ = load(c, count)
    g_ = load(g, count)
    dc = maxdiff(r, c_)
    dg = maxdiff(r, g_)
    okc = dc < TOL
    okg = dg < TOL
    all_pass = all_pass and okc and okg
    print(f"{label:28s} {dc:12.3e} {'PASS' if okc else 'FAIL':>8s}  {dg:12.3e} {'PASS' if okg else 'FAIL':>8s}")

print()
print("ALL CHECKS PASSED (both C and GPU/OpenACC)" if all_pass else "SOME CHECKS FAILED")
