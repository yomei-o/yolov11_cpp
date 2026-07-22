#include "onnx_run.hpp"
#include "net11.hpp"
#include <cstdio>
int main(){
  const std::string D="pure/ref/data_net/";
  auto g=onx::load_onnx("yolo11n.onnx");
  std::ifstream io(D+"io.txt"); int64_t IMG,nc,na,no,nl; io>>IMG>>nc>>na>>no>>nl;
  auto x=from_data({1,3,IMG,IMG},rd(D+"x.bin"));
  printf("%zu nodes\n",g.nodes.size());
  auto vals=onx::run_onnx(g,x);
  int64_t NB=64,NS=nc,Atot=0; for(int l=0;l<3;++l){auto t=vals.at("box"+std::to_string(l));Atot+=t->shape[2]*t->shape[3];}
  std::vector<float> boxes(NB*Atot),scores(NS*Atot);
  auto pack=[&](std::vector<float>&d,const char*pre,int64_t C){int64_t off=0;for(int l=0;l<3;++l){auto t=vals.at(pre+std::to_string(l));int64_t hw=t->shape[2]*t->shape[3];for(int64_t c=0;c<C;++c)for(int64_t a=0;a<hw;++a)d[c*Atot+off+a]=t->data[c*hw+a];off+=hw;}};
  pack(boxes,"box",NB);pack(scores,"cls",NS);
  auto rb=rd(D+"ref_boxes.bin"),rs=rd(D+"ref_scores.bin");
  double db=0,ds=0;for(size_t i=0;i<boxes.size();++i)db=std::max(db,(double)std::abs(boxes[i]-rb[i]));for(size_t i=0;i<scores.size();++i)ds=std::max(ds,(double)std::abs(scores[i]-rs[i]));
  printf("pure reader boxes=%.3e scores=%.3e %s\n",db,ds,(db<1e-3&&ds<1e-3)?"OK":"FAIL");
  return 0;
}
