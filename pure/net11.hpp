// Full yolo11n forward on the pure engine. New vs yolov8: C3k2 (Bottleneck or nested
// C3k), C2PSA (multi-head attention: qkv/softmax/matmul/pe/proj + FFN), and a DFL head
// with depthwise convs. Grouped conv is handled by conv2d(...,groups). Consumes convs
// in the order export_yolo11.py emits them. Head is anchor-free DFL (loss/decode = v8).
#pragma once
#include "autograd.hpp"
#include "ops2d.hpp"        // reshape, softmax_rows, mul_scalar
#include "linalg.hpp"       // matmul, transpose2d
#include <fstream>
#include <string>
#include <cmath>

struct ConvW { Tensor w, b; int64_t stride, pad, groups, act; };
struct Provider { std::vector<ConvW> convs; size_t i = 0; ConvW& next() { return convs[i++]; } };

inline std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  auto n = f.tellg(); f.seekg(0); std::vector<float> v(n / sizeof(float)); f.read((char*)v.data(), n); return v;
}
inline Provider load_net(const std::string& D) {
  std::ifstream f(D + "manifest.txt"); if (!f) { printf("run: python pure/ref/export_yolo11.py\n"); std::exit(1); }
  int n; f >> n; Provider p;
  for (int i = 0; i < n; ++i) {
    int64_t Co, Ci, k, s, pad, g, act; f >> Co >> Ci >> k >> s >> pad >> g >> act;
    ConvW c; c.stride = s; c.pad = pad; c.groups = g; c.act = act;
    c.w = from_data({Co, Ci, k, k}, rd(D + "w" + std::to_string(i) + ".bin"));
    c.b = from_data({Co}, rd(D + "b" + std::to_string(i) + ".bin"));
    p.convs.push_back(c);
  }
  return p;
}
inline Provider load_net_blob(const std::string& D) {
  std::ifstream f(D + "manifest.txt"); if (!f) { printf("run: python pure/ref/export_yolo11.py\n"); std::exit(1); }
  int n; f >> n; std::vector<float> blob = rd(D + "weights.bin"); Provider p; size_t off = 0;
  for (int i = 0; i < n; ++i) {
    int64_t Co, Ci, k, s, pad, g, act; f >> Co >> Ci >> k >> s >> pad >> g >> act;
    int64_t wn = Co * Ci * k * k; ConvW c; c.stride = s; c.pad = pad; c.groups = g; c.act = act;
    c.w = from_data({Co, Ci, k, k}, std::vector<float>(blob.begin()+off, blob.begin()+off+wn)); off += wn;
    c.b = from_data({Co}, std::vector<float>(blob.begin()+off, blob.begin()+off+Co)); off += Co;
    p.convs.push_back(c);
  }
  return p;
}

inline Tensor conv_apply(const Tensor& x, ConvW& c) {
  auto y = conv2d(x, c.w, c.b, c.stride, c.pad, c.groups);
  return c.act ? silu(y) : y;
}
inline Tensor cL(const Tensor& x, Provider& p) { return conv_apply(x, p.next()); }

inline Tensor bottleneck(const Tensor& x, Provider& p, bool sc) {
  auto h = conv_apply(x, p.next()); h = conv_apply(h, p.next());
  return sc ? add(h, x) : h;
}
inline Tensor c3k(const Tensor& x, Provider& p, int64_t inner_n, bool sc) {
  auto last = conv_apply(x, p.next());                 // cv1
  for (int64_t j = 0; j < inner_n; ++j) last = bottleneck(last, p, sc);
  auto y2 = conv_apply(x, p.next());                   // cv2
  return conv_apply(concat_ch({last, y2}), p.next());  // cv3
}
inline Tensor c3k2(const Tensor& x, Provider& p, int64_t n, bool is_c3k, int64_t inner_n, bool sc) {
  auto y0 = conv_apply(x, p.next());                   // cv1 -> 2c
  int64_t twoc = y0->shape[1], c = twoc / 2;
  std::vector<Tensor> outs = {slice_ch(y0, 0, c), slice_ch(y0, c, twoc)};
  Tensor last = outs[1];
  for (int64_t j = 0; j < n; ++j) { last = is_c3k ? c3k(last, p, inner_n, sc) : bottleneck(last, p, sc); outs.push_back(last); }
  return conv_apply(concat_ch(outs), p.next());        // cv2
}
inline Tensor sppf(const Tensor& x, Provider& p) {
  auto x1 = conv_apply(x, p.next());
  auto q1 = maxpool2d(x1, 5, 1, 2), q2 = maxpool2d(q1, 5, 1, 2), q3 = maxpool2d(q2, 5, 1, 2);
  return conv_apply(concat_ch({x1, q1, q2, q3}), p.next());
}

// Multi-head attention (ultralytics Attention). qkv/proj/pe are the next 3 convs.
inline Tensor attention(const Tensor& x, Provider& p, int64_t heads, int64_t key_dim, int64_t head_dim) {
  ConvW qkv = p.next(), proj = p.next(), pe = p.next();
  auto qkvo = conv_apply(x, qkv);                      // (1,h,H,W), act=False
  int64_t H = x->shape[2], W = x->shape[3], N = H * W, per = 2 * key_dim + head_dim;
  float scale = 1.f / std::sqrt((float)key_dim);
  std::vector<Tensor> xatt, vfull;
  for (int64_t hh = 0; hh < heads; ++hh) {
    int64_t off = hh * per;
    auto q = reshape(slice_ch(qkvo, off, off + key_dim), {key_dim, N});
    auto k = reshape(slice_ch(qkvo, off + key_dim, off + 2 * key_dim), {key_dim, N});
    auto v = reshape(slice_ch(qkvo, off + 2 * key_dim, off + per), {head_dim, N});
    auto attn = softmax_rows(mul_scalar(matmul(transpose2d(q), k), scale));   // (N,N), softmax over cols
    auto outh = matmul(v, transpose2d(attn));                                 // (head_dim, N)
    xatt.push_back(reshape(outh, {1, head_dim, H, W}));
    vfull.push_back(reshape(v, {1, head_dim, H, W}));
  }
  auto xa = concat_ch(xatt);                            // (1, dim, H, W)
  auto pev = conv_apply(concat_ch(vfull), pe);          // depthwise pe, act=False
  return conv_apply(add(xa, pev), proj);                // proj, act=False
}
inline Tensor c2psa(const Tensor& x, Provider& p, int64_t n, int64_t heads, int64_t key_dim, int64_t head_dim) {
  auto y = conv_apply(x, p.next());                     // cv1 -> 2*dim
  int64_t twoc = y->shape[1], dim = twoc / 2;
  auto a = slice_ch(y, 0, dim); auto b = slice_ch(y, dim, twoc);
  for (int64_t j = 0; j < n; ++j) {
    b = add(b, attention(b, p, heads, key_dim, head_dim));          // b + attn(b)
    auto f = conv_apply(b, p.next()); f = conv_apply(f, p.next());  // ffn
    b = add(b, f);
  }
  return conv_apply(concat_ch({a, b}), p.next());       // cv2
}

inline std::pair<Tensor, Tensor> detect_level(const Tensor& x, Provider& p) {
  auto hb = cL(x, p); hb = cL(hb, p); auto box = conv_apply(hb, p.next());   // cv2: k3,k3,plain
  auto hc = cL(x, p); hc = cL(hc, p); hc = cL(hc, p); hc = cL(hc, p);        // cv3: dw,pw,dw,pw
  auto cls = conv_apply(hc, p.next());                                       // plain
  return {box, cls};
}

// Full yolo11n. Returns the 3 (box,cls) level pairs (raw, pre-DFL).
inline std::vector<std::pair<Tensor, Tensor>> yolo11n_forward(const Tensor& x, Provider& p) {
  auto x0 = cL(x, p);
  auto x1 = cL(x0, p);
  auto x2 = c3k2(x1, p, 1, false, 0, true);
  auto x3 = cL(x2, p);
  auto x4 = c3k2(x3, p, 1, false, 0, true);
  auto x5 = cL(x4, p);
  auto x6 = c3k2(x5, p, 1, true, 2, true);
  auto x7 = cL(x6, p);
  auto x8 = c3k2(x7, p, 1, true, 2, true);
  auto x9 = sppf(x8, p);
  auto x10 = c2psa(x9, p, 1, 2, 32, 64);
  auto x11 = upsample_nearest(x10, 2);
  auto x12 = concat_ch({x11, x6});
  auto x13 = c3k2(x12, p, 1, false, 0, true);
  auto x14 = upsample_nearest(x13, 2);
  auto x15 = concat_ch({x14, x4});
  auto x16 = c3k2(x15, p, 1, false, 0, true);           // P3
  auto x17 = cL(x16, p);
  auto x18 = concat_ch({x17, x13});
  auto x19 = c3k2(x18, p, 1, false, 0, true);           // P4
  auto x20 = cL(x19, p);
  auto x21 = concat_ch({x20, x10});
  auto x22 = c3k2(x21, p, 1, true, 2, true);            // P5
  std::vector<std::pair<Tensor, Tensor>> out;
  for (auto& xi : {x16, x19, x22}) out.push_back(detect_level(xi, p));
  return out;
}
