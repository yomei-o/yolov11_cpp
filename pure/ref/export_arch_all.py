"""Dump the architecture description (manifest_unfused.txt + names.txt) for EVERY size
n/s/m/l/x, built from the .yaml config (random init, NO pretrained download). These tiny
text files let the pure-C++ make_init_pt generate initial weights for any size with zero
Python. Writes pure/ref/arch/<model>/{manifest_unfused.txt,names.txt}.
Manifest line format matches export_unfused11.py: kind Co Ci k s pad groups eps act.
Usage: python export_arch_all.py"""
import os, sys, torch.nn as nn
from ultralytics import YOLO
from yolo11_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = sys.argv[1:] or ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]

for MODEL in SIZES:
    ym = YOLO(MODEL + ".yaml")          # architecture only, random init, no download
    mods = walk(ym.model.model.eval())
    qn = {id(m): nm for nm, m in ym.model.named_modules()}
    lines = [str(len(mods))]; names = []
    for kind, mod in mods:
        p = qn[id(mod)]
        if kind == "conv":
            conv, bn = mod.conv, mod.bn
            act = 1 if isinstance(mod.act, nn.SiLU) else 0
            names += [f"{p}.conv.weight", f"{p}.bn.weight", f"{p}.bn.bias", f"{p}.bn.running_mean", f"{p}.bn.running_var"]
            Co, Ci = conv.weight.shape[0], conv.weight.shape[1]
            lines.append(f"1 {Co} {Ci} {conv.kernel_size[0]} {conv.stride[0]} {conv.padding[0]} {conv.groups} {bn.eps} {act}")
        else:
            names += [f"{p}.weight", f"{p}.bias"]
            Co, Ci = mod.weight.shape[0], mod.weight.shape[1]
            lines.append(f"0 {Co} {Ci} {mod.kernel_size[0]} {mod.stride[0]} {mod.padding[0]} {mod.groups} 0 0")
    D = os.path.join(HERE, "arch", MODEL); os.makedirs(D, exist_ok=True)
    open(os.path.join(D, "manifest_unfused.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(D, "names.txt"), "w").write("\n".join(names) + "\n")
    print(f"{MODEL}: {len(mods)} layers, {len(names)} tensors -> arch/{MODEL}/")
