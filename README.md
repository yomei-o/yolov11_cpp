# yolov11_cpp

Training **YOLO11** in C++ with **zero external dependencies** — a from-scratch autograd
engine, C++ standard library only (plus two vendored single-header image libs for the
demo). Every step is **verified numerically against Ultralytics YOLO11** (PyTorch).

日本語: 本物の YOLO11 を C++ で。自作 autograd エンジン（標準ライブラリのみ）で yolo11n の
順伝播・損失・学習・推論を再現し、各段階を本家 Ultralytics と数値比較。CPU / OpenMP は
`-fopenmp` の有無だけ。姉妹プロジェクト: **yolov8_cpp**, **yolov5_cpp**。

The autograd engine (im2col+GEMM conv incl. **grouped/depthwise**, BN, SiLU, matmul,
softmax, …), optimizers, dataloader and COCO-mAP are shared with yolov8_cpp. What is new
here is YOLO11's architecture: **C3k2** blocks (Bottleneck or nested **C3k**), **C2PSA**
(multi-head self-attention: qkv / scaled dot-product softmax / positional depthwise conv
/ proj + FFN), and a **DFL detect head with depthwise convs**. The head is anchor-free
DFL, so the **loss (DFL + TAL + CIoU + BCE) and decode are reused from yolov8** verbatim.

## Status
| file | milestone | result |
|------|-----------|--------|
| `pure/net11.hpp` + `pure/m1_forward.cpp` | **full yolo11n forward** (C3k2, C2PSA attention, DFL head) | matches yolo11n ~3e-5 |
| `pure/mg_gconv.cpp` | grouped/depthwise conv (fwd+bwd) | matches torch ~1e-5 |

Planned next: reuse yolov8's v8 loss + TAL for training, decode + NMS inference, mAP,
real-image demo, .pt/ONNX I/O.

## Build
```sh
python pure/ref/export_yolo11.py 64        # yolo11n fused weights + reference forward
g++ -std=c++20 -O2 pure/m1_forward.cpp -o m1     # or: cl /std:c++20 /O2 /EHsc pure/m1_forward.cpp
./m1
```

## Licenses & attribution
Own code is **BSD 3-Clause** ([LICENSE](LICENSE)). Bundled Ultralytics YOLO11 weights are
**AGPL-3.0**, stb is **public-domain / MIT** — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
