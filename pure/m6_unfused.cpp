#include "net11_unfused.hpp"
#include <cstdio>
int main(){
  const std::string D="pure/ref/data_net/"; auto p=load_net_unfused(D);
  std::ifstream io(D+"io.txt"); int64_t IMG; io>>IMG;
  auto x=from_data({1,3,IMG,IMG},rd(D+"x.bin"));
  auto h=yolo11n_forward_u(x,p,false);
  std::vector<float> out; for(auto&lv:h){out.insert(out.end(),lv.first->data.begin(),lv.first->data.end());}
  int64_t Atot=0; for(auto&lv:h) Atot+=lv.first->shape[2]*lv.first->shape[3];
  int64_t NB=h[0].first->shape[1], NS=h[0].second->shape[1];
  std::vector<float> boxes(NB*Atot),scores(NS*Atot);
  auto pack=[&](std::vector<float>&d,bool bx){int64_t off=0;for(auto&lv:h){const Tensor&t=bx?lv.first:lv.second;int64_t C=t->shape[1],hw=t->shape[2]*t->shape[3];for(int64_t c=0;c<C;++c)for(int64_t a=0;a<hw;++a)d[c*Atot+off+a]=t->data[c*hw+a];off+=hw;}};
  pack(boxes,true);pack(scores,false);
  auto rb=rd(D+"ref_boxes.bin"),rs=rd(D+"ref_scores.bin");
  double db=0,ds=0; for(size_t i=0;i<boxes.size();++i)db=std::max(db,(double)std::abs(boxes[i]-rb[i])); for(size_t i=0;i<scores.size();++i)ds=std::max(ds,(double)std::abs(scores[i]-rs[i]));
  printf("consumed %zu/%zu; boxes=%.3e scores=%.3e\n",p.i,p.layers.size(),db,ds);
  bool ok=db<1e-3&&ds<1e-3; printf("\n%s\n", ok?"yolo11 unfused: conv+BN+SiLU == yolo11n forward":"MISMATCH"); return ok?0:1;
}
