"""'.pt' compatibility test (yolo11): load the C++-synth-trained weights (data_wb/) into
yolo11n in canonical order, save yolo11n_synth.pt, then run Ultralytics inference on the
held-out synthetic test image — confirming a C++-trained model runs in Ultralytics."""
import os, numpy as np, torch
from ultralytics import YOLO
from yolo11_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(HERE, "data_wb"); DS = os.path.join(HERE, "data_synth")
def r(n): return np.fromfile(os.path.join(DW, n), np.float32)

ym = YOLO("yolo11n.pt")
mods = walk(ym.model.model)
def load_(p, a):
    with torch.no_grad(): p.copy_(torch.from_numpy(a.astype(np.float32)).reshape(p.shape))
serr = 0.0
for i, (kind, mod) in enumerate(mods):
    if kind == "conv":
        pairs = [(r(f"cw{i}.bin"), mod.conv.weight), (r(f"bg{i}.bin"), mod.bn.weight), (r(f"bb{i}.bin"), mod.bn.bias),
                 (r(f"rm{i}.bin"), mod.bn.running_mean), (r(f"rv{i}.bin"), mod.bn.running_var)]
    else:
        pairs = [(r(f"cw{i}.bin"), mod.weight), (r(f"cb{i}.bin"), mod.bias)]
    for a, p in pairs:
        load_(p, a); serr = max(serr, float(np.abs(a.reshape(p.shape) - p.detach().numpy()).max()))
print(f"serialization max|diff| = {serr:.3e}  {'OK' if serr < 1e-6 else 'FAIL'}")

out = "yolo11n_synth.pt"; ym.save(out)
names = {0: "red", 1: "green", 2: "blue"}
res = YOLO(out).predict(os.path.join(DS, "te00.png"), imgsz=64, conf=0.25, verbose=False)[0]
print(f"\nUltralytics on the C++-trained {out}: {len(res.boxes)} detections on te00.png")
for b in res.boxes:
    c = int(b.cls.item()); x1, y1, x2, y2 = b.xyxy[0].tolist()
    print(f"  cls {c} ({names.get(c, c)}) conf={b.conf.item():.2f} xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
print("\n.pt round-trip OK — a C++-trained yolo11 model loads and runs in Ultralytics"
      if serr < 1e-6 else "MISMATCH")
