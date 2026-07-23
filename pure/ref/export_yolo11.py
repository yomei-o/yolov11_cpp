"""Export yolo11n (BN-folded) convs in the exact order the pure C++ forward consumes
them, plus a fixed input and reference boxes/scores. Handles C3k2 (Bottleneck / nested
C3k), C2PSA attention (qkv/proj/pe + ffn) and the DFL head (with depthwise convs).
Usage: python export_yolo11.py [imgsz]"""
import os, sys, torch, torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64

ym = YOLO("yolo11n.pt"); L = ym.model.model.eval()
convs = []   # (w, b, k, s, pad, groups, act)
def fuse(cv):
    conv, bn = cv.conv, cv.bn
    std = torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * (bn.weight / std).reshape(-1,1,1,1), bn.bias - bn.weight * bn.running_mean / std
def emitC(cv):                                   # ultralytics Conv / DWConv (conv+bn[+act])
    w, b = fuse(cv); c = cv.conv; act = 1 if isinstance(cv.act, nn.SiLU) else 0
    convs.append((w, b, c.kernel_size[0], c.stride[0], c.padding[0], c.groups, act))
def emitP(c): convs.append((c.weight, c.bias, c.kernel_size[0], c.stride[0], c.padding[0], c.groups, 0))
def emit_bott(b): emitC(b.cv1); emitC(b.cv2)
def emit_c3k(m): emitC(m.cv1); [emit_bott(b) for b in m.m]; emitC(m.cv2); emitC(m.cv3)
def emit_c3k2(m):
    emitC(m.cv1)
    for mm in m.m: (emit_c3k if isinstance(mm, C3k) else emit_bott)(mm)
    emitC(m.cv2)
def emit_sppf(m): emitC(m.cv1); emitC(m.cv2)
def emit_c2psa(m):
    emitC(m.cv1)
    for psa in m.m:
        a = psa.attn; emitC(a.qkv); emitC(a.proj); emitC(a.pe); emitC(psa.ffn[0]); emitC(psa.ffn[1])
    emitC(m.cv2)
def emit_seq(s):
    for x in s: (emitP if isinstance(x, nn.Conv2d) else emitC)(x)
def emit_detect(det):
    for i in range(3):
        for x in det.cv2[i]: (emitP if isinstance(x, nn.Conv2d) else emitC)(x)
        for x in det.cv3[i]:
            if isinstance(x, nn.Sequential): emit_seq(x)
            elif isinstance(x, nn.Conv2d): emitP(x)
            else: emitC(x)

EMIT = {"Conv": emitC, "C3k2": emit_c3k2, "SPPF": emit_sppf, "C2PSA": emit_c2psa}
for mod in L[:-1]:
    fn = EMIT.get(type(mod).__name__)
    if fn: fn(mod)                         # Upsample / Concat have no weights
det = L[-1]; emit_detect(det)

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(convs))]; blob = []
for i, (w, b, k, s, p, g, act) in enumerate(convs):
    save(f"w{i}.bin", w); save(f"b{i}.bin", b)
    blob.append(w.detach().cpu().numpy().ravel()); blob.append(b.detach().cpu().numpy().ravel())
    lines.append(f"{w.shape[0]} {w.shape[1]} {k} {s} {p} {g} {act}")
open(os.path.join(D, "manifest.txt"), "w").write("\n".join(lines) + "\n")
import numpy as np
np.concatenate(blob).astype(np.float32).tofile(os.path.join(D, "weights.bin"))

torch.manual_seed(0)
x = torch.randn(1, 3, IMG, IMG).cpu()
dm = ym.model.float().cpu(); dm.train()   # force CPU (reference is CPU-computed; safe on GPU hosts)
for mod in dm.modules():
    if isinstance(mod, torch.nn.BatchNorm2d): mod.eval()
with torch.no_grad(): y = dm(x)
save("x.bin", x); save("ref_boxes.bin", y["boxes"]); save("ref_scores.bin", y["scores"])
open(os.path.join(D, "io.txt"), "w").write(f"{IMG} {y['boxes'].shape[1]} {y['scores'].shape[1]} {det.reg_max} {y['boxes'].shape[2]}\n")
print(f"convs={len(convs)} imgsz={IMG} boxes={tuple(y['boxes'].shape)} scores={tuple(y['scores'].shape)}")
