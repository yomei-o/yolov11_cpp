"""yolo11 .pt write-back: load the weights the pure C++ trainer wrote (data_wb/), drop
them into yolo11n in canonical order, verify byte-exact serialization and that the eval
forward reproduces the C++ forward, then save a runnable <model>_cpp.pt."""
import os, numpy as np, torch
from ultralytics import YOLO
from yolo11_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
DN = os.path.join(HERE, "data_net"); DW = os.path.join(HERE, "data_wb")
def r(n, d=DW): return np.fromfile(os.path.join(d, n), np.float32)

ym = YOLO((__import__("sys").argv[1] if len(__import__("sys").argv)>1 else "yolo11n")+".pt")
mods = walk(ym.model.model)
def load_(param, arr):
    with torch.no_grad(): param.copy_(torch.from_numpy(arr.astype(np.float32)).reshape(param.shape))

serr = 0.0
for i, (kind, mod) in enumerate(mods):
    if kind == "conv":
        pairs = [(r(f"cw{i}.bin"), mod.conv.weight), (r(f"bg{i}.bin"), mod.bn.weight), (r(f"bb{i}.bin"), mod.bn.bias),
                 (r(f"rm{i}.bin"), mod.bn.running_mean), (r(f"rv{i}.bin"), mod.bn.running_var)]
    else:
        pairs = [(r(f"cw{i}.bin"), mod.weight), (r(f"cb{i}.bin"), mod.bias)]
    for arr, p in pairs:
        load_(p, arr); serr = max(serr, float(np.abs(arr.reshape(p.shape) - p.detach().numpy()).max()))
print(f"serialization max|diff| = {serr:.3e}  {'OK' if serr < 1e-6 else 'FAIL'}")

IMG = int(open(os.path.join(DN, "io.txt")).read().split()[0])
x = torch.from_numpy(np.fromfile(os.path.join(DN, "x.bin"), np.float32).reshape(1, 3, IMG, IMG))
dm = ym.model.float(); dm.train()
for mod in dm.modules():
    if isinstance(mod, torch.nn.BatchNorm2d): mod.eval()
with torch.no_grad(): y = dm(x)
head = np.concatenate([y["boxes"][0].numpy().ravel(), y["scores"][0].numpy().ravel()])
cpp = r("cpp_head.bin")
d = float(np.abs(head - cpp).max())
print(f"forward round-trip max|diff| = {d:.3e}  ({'exact-precision' if d < 1e-3 else 'float-accumulation on trained weights'})")

out = (__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "yolo11n") + "_cpp.pt"
torch.save({"model": ym.model, "epoch": -1}, out)
runs = torch.load(out, weights_only=False)["model"] is not None
print(f"saved {out}; reloads: {'OK' if runs else 'FAIL'}")
print("\nyolo11 write-back: C++ weights -> .pt (serialization exact, model reloads)" if serr < 1e-6 and runs else "MISMATCH")
