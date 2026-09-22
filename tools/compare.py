#!/usr/bin/env python3
"""Compare C dumps against the MLX reference directory produced by mlx_reference.py.
usage: python tools/compare.py ref/fox out/fox   (out has mel.f32, enc.f32, tokens.txt)"""
import json, os, sys
import numpy as np

ref, out = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(ref, "meta.json")))
D = 1024

def load(path, cols):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(-1, cols) if a.size else a.reshape(0, cols)

ok = True
if os.path.exists(os.path.join(out, "mel.f32")):
    rm, cm = load(os.path.join(ref, "mel.f32"), 128), load(os.path.join(out, "mel.f32"), 128)
    print(f"mel: ref {rm.shape} c {cm.shape}", end=" ")
    if rm.shape == cm.shape:
        d = np.abs(rm - cm); print(f"max abs diff {d.max():.3e} mean {d.mean():.3e} (ref range {rm.min():.2f}..{rm.max():.2f})")
        ok &= d.max() < 1e-3
    else:
        print("SHAPE MISMATCH"); ok = False
if os.path.exists(os.path.join(out, "enc.f32")):
    re_, ce = load(os.path.join(ref, "enc_all.f32"), D), load(os.path.join(out, "enc.f32"), D)
    print(f"enc: ref {re_.shape} c {ce.shape}", end=" ")
    n = min(len(re_), len(ce))
    if n:
        d = np.abs(re_[:n] - ce[:n])
        rel = d.max() / (np.abs(re_[:n]).max() + 1e-9)
        print(f"max abs diff {d.max():.3e} (rel {rel:.2e}) mean {d.mean():.3e}; per-row max:", np.round(d.max(axis=1)[:8], 4), "...")
        ok &= rel < 1e-2 and re_.shape == ce.shape
    else:
        print("no rows"); ok = False
if os.path.exists(os.path.join(out, "tokens.txt")):
    rt = [int(x) for x in open(os.path.join(ref, "tokens.txt")).readline().split()]
    ct = [int(l) for l in open(os.path.join(out, "tokens.txt")) if l.strip()]
    same = rt == ct
    print(f"tokens: ref {len(rt)} c {len(ct)} identical={same}")
    if not same:
        for i, (a, b) in enumerate(zip(rt, ct)):
            if a != b: print(f"  first difference at {i}: ref {a} c {b}"); break
    ok &= same
print("PARITY OK" if ok else "PARITY FAILED")
sys.exit(0 if ok else 1)
