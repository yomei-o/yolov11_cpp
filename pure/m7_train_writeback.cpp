// yolov11_cpp: train yolo11n in the pure engine with conv+BN separate, then write the
// updated weights (conv, BN params) back to flat .bin for the Python .pt bridge. Also
// dumps an eval-mode forward so the round-trip can be checked.
//   run: m7_train_writeback [iters]   (0 = round-trip check on original weights)
#include "net11_unfused.hpp"
#include "v8pure.hpp"
#include "tal.hpp"
#include "optim.hpp"
#include <cstdio>
#include <random>
#include <filesystem>

int main(int argc, char** argv) {
  const int ITERS = argc > 1 ? atoi(argv[1]) : 20;
  const std::string D = "pure/ref/data_net/", D2 = "pure/ref/data_wb/";
  std::filesystem::create_directories(D2);
  auto prov = load_net_unfused(D);
  std::vector<Tensor> params;
  for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind == 1) { params.push_back(L.gamma); params.push_back(L.beta); } else params.push_back(L.b); }
  Adam opt(params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, false);

  const int64_t B = 2, IMG = 64, NC = 80, RM = 16, M = 3, TOPK = 10; const float ALPHA = 0.5f, BETA = 6.0f;
  struct Lv { int64_t h, w; float s; }; std::vector<Lv> levels = {{8,8,8},{4,4,16},{2,2,32}};
  std::vector<float> ax, ay, ss, ai; for (auto& L : levels) for (int64_t y=0;y<L.h;++y) for (int64_t x=0;x<L.w;++x){ ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);ai.push_back((x+.5f)*L.s);ai.push_back((y+.5f)*L.s);}
  int64_t A = ss.size(), R = B*A; std::vector<float> ancx(R),ancy(R),strd(R);
  for (int64_t r=0;r<R;++r){int64_t a=r%A;ancx[r]=ax[a];ancy[r]=ay[a];strd[r]=ss[a];}
  auto img = make_tensor({B,3,IMG,IMG});
  { std::mt19937 rng(0); std::normal_distribution<float> nd(0,1); for (auto& v:img->data) v=nd(rng); }
  std::vector<float> gb = {8,8,40,44,30,10,60,40,10,30,34,60,4,4,44,52,28,24,60,60,6,34,38,60};
  std::vector<int64_t> gl = {12,40,7,63,3,25}; std::vector<float> mg(B*M,1.f);

  printf("iter |   total     box      cls      dfl\n");
  for (int it = 0; it < ITERS; ++it) {
    prov.i = 0; auto lvs = yolo11n_forward_u(img, prov, true);
    std::vector<Tensor> bx = {lvs[0].first,lvs[1].first,lvs[2].first}, cs = {lvs[0].second,lvs[1].second,lvs[2].second};
    auto pd = pack_levels(bx, B, A, 4*RM); auto ps = pack_levels(cs, B, A, NC);
    std::vector<float> pdb(R*4),pss(R*NC);
    for (int64_t r=0;r<R;++r){ for(int j=0;j<4;++j){float mx=-1e30f;for(int k=0;k<RM;++k)mx=std::max(mx,pd->data[r*64+j*RM+k]);double s=0;float d=0;std::vector<float>e(RM);for(int k=0;k<RM;++k){e[k]=std::exp(pd->data[r*64+j*RM+k]-mx);s+=e[k];}for(int k=0;k<RM;++k)d+=(float)(e[k]/s)*k;pdb[r*4+j]=d;}
      float axr=ax[r%A],ayr=ay[r%A],st=ss[r%A];float l=pdb[r*4],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];
      pdb[r*4]=(axr-l)*st;pdb[r*4+1]=(ayr-t)*st;pdb[r*4+2]=(axr+rr)*st;pdb[r*4+3]=(ayr+bb)*st;
      for(int64_t c=0;c<NC;++c)pss[r*NC+c]=1.f/(1.f+std::exp(-ps->data[r*NC+c])); }
    auto tal = tal_assign(pss,pdb,ai,gl,gb,mg,B,A,M,NC,TOPK,ALPHA,BETA);
    auto Lo = pure_v8_loss(pd,ps,ancx,ancy,strd,tal.tb,tal.ts,R,NC,RM);
    backward(Lo.total); opt.lr = cosine_lr(it,ITERS,1e-3f,3); opt.step();
    if (it%5==0) printf("%4d | %8.4f %8.4f %8.4f %8.4f\n", it, Lo.total->data[0],Lo.box->data[0],Lo.cls->data[0],Lo.dfl->data[0]);
    free_graph(Lo.total);
  }
  printf("training done (%d iters).\n", ITERS);

  auto wr = [&](const std::string& n, const std::vector<float>& v){ std::ofstream f(D2+n,std::ios::binary); f.write((const char*)v.data(), v.size()*sizeof(float)); };
  for (size_t i=0;i<prov.layers.size();++i){ auto& L=prov.layers[i]; std::string s=std::to_string(i);
    wr("cw"+s+".bin",L.w->data);
    if(L.kind==1){wr("bg"+s+".bin",L.gamma->data);wr("bb"+s+".bin",L.beta->data);wr("rm"+s+".bin",L.rm);wr("rv"+s+".bin",L.rv);} else wr("cb"+s+".bin",L.b->data); }
  auto x = from_data({1,3,IMG,IMG}, rd(D+"x.bin"));
  prov.i=0; auto hv = yolo11n_forward_u(x,prov,false);
  int64_t Atot=0; for(auto&lv:hv) Atot+=lv.first->shape[2]*lv.first->shape[3];
  int64_t NB=hv[0].first->shape[1], NS=hv[0].second->shape[1];
  std::vector<float> head; head.reserve((NB+NS)*Atot);
  { std::vector<float> b(NB*Atot),c(NS*Atot); int64_t off=0;
    for(auto&lv:hv){int64_t hw=lv.first->shape[2]*lv.first->shape[3];for(int64_t ch=0;ch<NB;++ch)for(int64_t a=0;a<hw;++a)b[ch*Atot+off+a]=lv.first->data[ch*hw+a];for(int64_t ch=0;ch<NS;++ch)for(int64_t a=0;a<hw;++a)c[ch*Atot+off+a]=lv.second->data[ch*hw+a];off+=hw;}
    head.insert(head.end(),b.begin(),b.end()); head.insert(head.end(),c.begin(),c.end()); }
  wr("cpp_head.bin", head);
  printf("wrote updated weights + eval forward to %s\n", D2.c_str());
  return 0;
}
