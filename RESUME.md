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
- **Augmentation** — mosaic + mixup + random-affine (rotate/scale/shear/translate) + HSV +
  flip, with **close-mosaic** (disable for last N epochs). Toggle via `AugCfg` / CLI flags.
- **Unified `yolo` CLI** (`pure/yolo.cpp`) reading `data.yaml`: `train` / `val` / `detect`
  (`export` still delegates to the standalone `onnx_export11`). Val reports **mAP@0.5 and
  mAP@0.5:0.95**.
- GPU/CUDA seam (`pure/backend.hpp`) present; conv/matmul route through `bk::`; GPU training
  done for all four (per the parallel session).

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar) and compare final
   mAP@0.5:0.95 against Ultralytics. Only synthetic data has been checked so far.
2. **Custom `nc`** — the head is fixed at 80 classes; a dataset with `nc != 80` needs the cls
   head resized + re-initialised. Today class ids must be < 80.
3. **`export` in the unified CLI** — fold BN from the `.pt` and emit ONNX in-CLI (today
   `yolo export` points at the standalone, onnxruntime-verified `onnx_export11`).
4. **Training-quality features** — EMA weights, resume-from-checkpoint, multi-scale, rect val,
   label smoothing, separate bias/BN LR. (mAP@0.5:0.95 in val — done.)
5. **Speed** — YOLO11's **C2PSA attention makes training the slowest of the four repos**; the
   deep attention graph is torn down each step with `free_graph`, and it **OOMs at imgsz 128
   B=4** on limited RAM (use imgsz≤96). Batching the attention forward is the main win.

## Notes / gotchas
- Everything is xyxy in the **letterboxed SxS pixel** space; GT and decoded detections share
  it. `load_boxes_orig` reads either label format into original pixels, then `lb_map` applies
  the letterbox transform.
- `make_init_pt … rand` trains but won't converge on tiny data; use `from <pretrained.pt>`.
- Attention was once B=1-only (crashed at B≥2, heap OOB found via `cl /fsanitize=address`);
  fixed with slice/concat_batch + per-batch attention + iterative build_topo + `free_graph`.
- Build: MSVC via `C:/prog/claude/cc11.sh`; `scratch/` must pre-exist.
