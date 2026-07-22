# Third-party notices / サードパーティ ライセンス表記

Bundled third-party components keep their own license and copyright.

## 1. Ultralytics YOLO11 — model weights & derived data
- Files: `weights/yolo11n/*` (BN-folded repack), any `yolo11*.pt`, reference tensors
  under `pure/ref/` (git-ignored).
- © Ultralytics. Source: https://github.com/ultralytics/ultralytics
- License: **GNU AGPL-3.0-or-later** — https://www.gnu.org/licenses/agpl-3.0.html
  Redistributing this repo must comply with AGPL-3.0 for these files.

## 2. stb — single-header image libraries
- Files: `pure/third_party/stb_image.h`, `pure/third_party/stb_image_write.h`
- Sean Barrett & contributors. https://github.com/nothings/stb
- License: Public Domain (Unlicense) OR MIT.

---
The project's own code (`pure/` engine, `ref/` scripts, docs) is **BSD 3-Clause** — see
[`LICENSE`](LICENSE). Distributing it with the AGPL-3.0 artifacts carries the AGPL
obligations for those files.
