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
checks = [
    ("Y",        "fb_ref_Y",       "fb_f_Y",       NT*D),
    ("G_X",      "fb_ref_GX",      "fb_f_GX",      NT*D),
    ("G_gamma1", "fb_ref_Ggamma1", "fb_f_Ggamma1", D),
    ("G_gamma2", "fb_ref_Ggamma2", "fb_f_Ggamma2", D),
    ("G_Wq",     "fb_ref_GWq",     "fb_f_GWq",     D*D),
    ("G_Wk",     "fb_ref_GWk",     "fb_f_GWk",     D*D),
    ("G_Wv",     "fb_ref_GWv",     "fb_f_GWv",     D*D),
    ("G_Wgate",  "fb_ref_GWg",     "fb_f_GWg",     D*DFF),
    ("G_Wup",    "fb_ref_GWu",     "fb_f_GWu",     D*DFF),
    ("G_Wdown",  "fb_ref_GWd",     "fb_f_GWd",     DFF*D),
]

print(f"{'tensor':12s} {'max abs error':>15s}  {'pass (< 1.2e-4)':>16s}")
all_pass = True
for label, ref, f90, count in checks:
    r = load(ref, count)
    f = load(f90, count)
    d = maxdiff(r, f)
    ok = d < TOL
    all_pass = all_pass and ok
    print(f"{label:12s} {d:15.3e}  {'PASS' if ok else 'FAIL':>16s}")

print()
print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
