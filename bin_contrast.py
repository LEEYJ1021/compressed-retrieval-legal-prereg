import glob, numpy as np
from evalcore import boot_macro
for m in "AB":
    R = f"out_full/supp2/{m}_std"
    src = np.load(f"{R}/src.npy", allow_pickle=True); cont = np.load(f"{R}/cont.npy", allow_pickle=True)
    for f in sorted(glob.glob(f"{R}/bin_B*.npy")):
        B = int(f.split("bin_B")[1][:-4]); b = np.load(f)
        for fmt in ["sq4", "sq8"]:
            L = [np.load(g) for g in glob.glob(f"{R}/lad_s*_{fmt}_{B}.npy")]
            if L:
                e, lo, hi = boot_macro(b - np.mean(L, 0), src, cont, nb=2000)
                print(m, B, f"bin-{fmt}", f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]")
