"""Canonical traversal of yolo11's conv-bearing modules, in the order the pure C++
forward consumes them (identical to export_yolo11.py). Shared by the unfused exporter
and the .pt write-back. Yields ('conv', Conv) or ('plain', nn.Conv2d)."""
import torch.nn as nn
from ultralytics.nn.modules.block import C3k

def walk(seq):
    order = []
    def emitC(cv): order.append(("conv", cv))
    def emitP(c): order.append(("plain", c))
    def bott(b): emitC(b.cv1); emitC(b.cv2)
    def c3k(m): emitC(m.cv1); [bott(x) for x in m.m]; emitC(m.cv2); emitC(m.cv3)
    def c3k2(m):
        emitC(m.cv1)
        for mm in m.m: (c3k if isinstance(mm, C3k) else bott)(mm)
        emitC(m.cv2)
    def sppf(m): emitC(m.cv1); emitC(m.cv2)
    def c2psa(m):
        emitC(m.cv1)
        for psa in m.m:
            a = psa.attn; emitC(a.qkv); emitC(a.proj); emitC(a.pe); emitC(psa.ffn[0]); emitC(psa.ffn[1])
        emitC(m.cv2)
    def detect(det):
        for i in range(3):
            for x in det.cv2[i]: (emitP if isinstance(x, nn.Conv2d) else emitC)(x)
            for x in det.cv3[i]:
                if isinstance(x, nn.Sequential):
                    for y in x: (emitP if isinstance(y, nn.Conv2d) else emitC)(y)
                elif isinstance(x, nn.Conv2d): emitP(x)
                else: emitC(x)
    W = {"Conv": emitC, "C3k2": c3k2, "SPPF": sppf, "C2PSA": c2psa}
    for mod in seq[:-1]:
        fn = W.get(type(mod).__name__)
        if fn: fn(mod)
    detect(seq[-1])
    return order
