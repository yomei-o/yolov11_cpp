"""Export yolo11n WITHOUT folding BN, in canonical order (for BN training + .pt
write-back). Writes into data_net/ alongside the fused export (reuses x.bin/ref_*.bin).
Manifest line: kind Co Ci k s pad groups eps act. Usage: python export_unfused11.py"""
import os, torch, torch.nn as nn
from ultralytics import YOLO
from yolo11_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)
import sys
ym = YOLO((sys.argv[1] if len(sys.argv)>1 else "yolo11n") + ".pt"); mods = walk(ym.model.model.eval())

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(mods))]
for i, (kind, mod) in enumerate(mods):
    if kind == "conv":
        conv, bn = mod.conv, mod.bn
        act = 1 if isinstance(mod.act, nn.SiLU) else 0
        save(f"cw{i}.bin", conv.weight)
        save(f"bg{i}.bin", bn.weight); save(f"bb{i}.bin", bn.bias)
        save(f"rm{i}.bin", bn.running_mean); save(f"rv{i}.bin", bn.running_var)
        Co, Ci = conv.weight.shape[0], conv.weight.shape[1]
        lines.append(f"1 {Co} {Ci} {conv.kernel_size[0]} {conv.stride[0]} {conv.padding[0]} {conv.groups} {bn.eps} {act}")
    else:
        save(f"cw{i}.bin", mod.weight); save(f"cb{i}.bin", mod.bias)
        Co, Ci = mod.weight.shape[0], mod.weight.shape[1]
        lines.append(f"0 {Co} {Ci} {mod.kernel_size[0]} {mod.stride[0]} {mod.padding[0]} {mod.groups} 0 0")
open(os.path.join(D, "manifest_unfused.txt"), "w").write("\n".join(lines) + "\n")
print(f"unfused: {len(mods)} layers")
