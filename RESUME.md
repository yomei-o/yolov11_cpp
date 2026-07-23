# RESUME — remaining work

Status of the pure-C++ YOLO11 training toolchain and what's left to make it a full
replacement for Ultralytics-quality training. Verified items live in [README.md](README.md);
this file is the forward-looking TODO.

## Done (pure C++, no Python at run time)
- Engine + full YOLO11 (C3k2, C2PSA attention, depthwise DFL head), all sizes (n/s/m/l/x),
  forward/loss/train/infer/mAP/ONNX/`.pt` — verified vs Ultralytics.
- Real training CLI (`pure/train_cli.cpp`): dataset scan → shuffled mini-batches → epochs →
  TAL → v8 loss → Adam(warmup+cosine+wd) → per-epoch val mAP@0.5 → `best.pt`/`last.pt`.
- Initial-weight `.pt` generated in C++ (`pure/make_init_pt.cpp`, `rand`/`from`), all sizes,
  zero-Python bootstrap; checkpoints load back into Ultralytics (0 unexpected).
- **Standard-YOLO dataset ingestion** — directory scan (`images/`↔`labels/`), normalised
  `cls xc yc w h` labels, arbitrary-size images letterboxed (`pure/dataset.hpp`
  `read_yolo_dataset` / `load_boxes_orig`).
- **Mosaic augmentation** (`make_mosaic`) + horizontal flip + brightness. `train_cli … <imgsz> <mosaic>`.
- GPU/CUDA seam (`pure/backend.hpp`) present; conv/matmul route through `bk::`.

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar) and compare final
   mAP@0.5:0.95 against Ultralytics. Only synthetic data has been checked so far.
2. **Richer augmentation** — HSV colour jitter, random affine (scale/translate/rotate/shear),
   mixup, and "close mosaic for the last N epochs". Only flip + brightness + mosaic exist.
3. **`data.yaml` + unified CLI** — parse `data.yaml` (train/val paths, `nc`, `names`) and add
   `train`/`val`/`detect`/`export` subcommands.
4. **Training-quality features** — EMA weights, resume-from-checkpoint, multi-scale training,
   rectangular val, label smoothing, separate bias/BN LR, mAP@0.5:0.95 in the val loop.
5. **Speed** — YOLO11's **C2PSA attention makes training the slowest of the four repos**;
   the deep attention graph is torn down each step with `free_graph`. Batching the attention
   forward and verifying the GPU path on real hardware are the main wins here.

## Notes / gotchas
- Everything is xyxy in the **letterboxed SxS pixel** space; GT and decoded detections share
  it. `load_boxes_orig` reads either label format into original pixels, then `lb_map` applies
  the letterbox transform.
- `make_init_pt … rand` trains but won't converge on tiny data; use `from <pretrained.pt>`.
- Attention was once B=1-only (crashed at B≥2, heap OOB found via `cl /fsanitize=address`);
  fixed with slice/concat_batch + per-batch attention + iterative build_topo + `free_graph`.
- Build: MSVC via `C:/prog/claude/cc11.sh`; `scratch/` must pre-exist.
