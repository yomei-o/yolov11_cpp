// yolov11_cpp: export yolo11n to a standard .onnx (opset 13, no deps). Beyond yolov8's
// ops it emits the C2PSA attention as Slice/Reshape/Transpose/MatMul/Softmax/Mul, and
// grouped/depthwise Conv (group attr). Input batch is 1. Outputs the 3 detect tensors.
//   run: onnx_export11   (needs pure/ref/data_net from export_yolo11.py)
#include "net11.hpp"
#include "onnx.hpp"
#include <string>
#include <cmath>
using namespace onx;

int main() {
  const std::string D = "pure/ref/data_net/";
  auto prov = load_net(D);
  std::ifstream io(D + "io.txt"); int64_t IMG, nc, na, no, nl; io >> IMG >> nc >> na >> no >> nl;
  Graph g; g.opset = 13;
  g.inputs.push_back({"images", {1, 3, IMG, IMG}});
  int uid = 0;
  auto U = [&](const char* p) { return std::string(p) + std::to_string(uid++); };

  auto conv_from = [&](const std::string& in, ConvW& c) -> std::string {
    int64_t Co = c.w->shape[0], Ci = c.w->shape[1], k = c.w->shape[2];
    std::string wn = U("w"), bn = U("b"), yn = U("cv");
    g.init_f.push_back({wn, {Co, Ci, k, k}, c.w->data}); g.init_f.push_back({bn, {Co}, c.b->data});
    onx::Node n; n.op_type="Conv"; n.name=yn; n.input={in,wn,bn}; n.output={yn};
    n.attr.push_back({"kernel_shape",A_INTS,0,0,"",{k,k},{}});
    n.attr.push_back({"strides",A_INTS,0,0,"",{c.stride,c.stride},{}});
    n.attr.push_back({"pads",A_INTS,0,0,"",{c.pad,c.pad,c.pad,c.pad},{}});
    n.attr.push_back({"group",A_INT,c.groups,0,"",{},{}});
    g.nodes.push_back(n);
    if (!c.act) return yn;
    std::string sn=U("sg"), mn=U("si"); g.nodes.push_back({"Sigmoid",sn,{yn},{sn},{}}); g.nodes.push_back({"Mul",mn,{yn,sn},{mn},{}}); return mn;
  };
  auto conv = [&](const std::string& in) -> std::string { return conv_from(in, prov.next()); };
  auto add=[&](const std::string&a,const std::string&b){std::string y=U("ad");g.nodes.push_back({"Add",y,{a,b},{y},{}});return y;};
  auto concat=[&](const std::vector<std::string>&xs){std::string y=U("ct");onx::Node n{"Concat",y,xs,{y},{}};n.attr.push_back({"axis",A_INT,1,0,"",{},{}});g.nodes.push_back(n);return y;};
  auto maxpool=[&](const std::string&in){std::string y=U("mp");onx::Node n{"MaxPool",y,{in},{y},{}};n.attr.push_back({"kernel_shape",A_INTS,0,0,"",{5,5},{}});n.attr.push_back({"strides",A_INTS,0,0,"",{1,1},{}});n.attr.push_back({"pads",A_INTS,0,0,"",{2,2,2,2},{}});g.nodes.push_back(n);return y;};
  auto resize=[&](const std::string&in){std::string sc=U("sc"),y=U("up");g.init_f.push_back({sc,{4},{1,1,2,2}});onx::Node n{"Resize",y,{in,"",sc},{y},{}};n.attr.push_back({"mode",A_STRING,0,0,"nearest",{},{}});n.attr.push_back({"coordinate_transformation_mode",A_STRING,0,0,"asymmetric",{},{}});n.attr.push_back({"nearest_mode",A_STRING,0,0,"floor",{},{}});g.nodes.push_back(n);return y;};
  auto slice=[&](const std::string&in,int64_t c0,int64_t c1){std::string s=U("st"),e=U("en"),a=U("ax"),y=U("sl");g.init_i.push_back({s,{1},{c0}});g.init_i.push_back({e,{1},{c1}});g.init_i.push_back({a,{1},{1}});g.nodes.push_back({"Slice",y,{in,s,e,a},{y},{}});return y;};
  auto reshape=[&](const std::string&in,std::vector<int64_t> shp){std::string s=U("rs"),y=U("rv");g.init_i.push_back({s,{(int64_t)shp.size()},shp});g.nodes.push_back({"Reshape",y,{in,s},{y},{}});return y;};
  auto transpose=[&](const std::string&in){std::string y=U("tr");onx::Node n{"Transpose",y,{in},{y},{}};n.attr.push_back({"perm",A_INTS,0,0,"",{1,0},{}});g.nodes.push_back(n);return y;};
  auto matmul=[&](const std::string&a,const std::string&b){std::string y=U("mm");g.nodes.push_back({"MatMul",y,{a,b},{y},{}});return y;};
  auto softmax=[&](const std::string&in){std::string y=U("sm");onx::Node n{"Softmax",y,{in},{y},{}};n.attr.push_back({"axis",A_INT,1,0,"",{},{}});g.nodes.push_back(n);return y;};
  auto mulc=[&](const std::string&in,float v){std::string s=U("k"),y=U("ms");g.init_f.push_back({s,{1},{v}});g.nodes.push_back({"Mul",y,{in,s},{y},{}});return y;};

  auto attention=[&](const std::string&x,int64_t H,int64_t W){
    ConvW& qkvw=prov.next(); ConvW& projw=prov.next(); ConvW& pew=prov.next();  // manifest order
    int64_t heads=2,kd=32,hd=64,per=2*kd+hd,N=H*W; float scale=1.f/std::sqrt((float)kd);
    std::string qkv=conv_from(x,qkvw); std::vector<std::string> xatt,vfull;
    for(int64_t hh=0;hh<heads;++hh){ int64_t off=hh*per;
      std::string q=reshape(slice(qkv,off,off+kd),{kd,N}), k=reshape(slice(qkv,off+kd,off+2*kd),{kd,N}), v=reshape(slice(qkv,off+2*kd,off+per),{hd,N});
      std::string attn=softmax(mulc(matmul(transpose(q),k),scale));
      xatt.push_back(reshape(matmul(v,transpose(attn)),{1,hd,H,W})); vfull.push_back(reshape(v,{1,hd,H,W})); }
    std::string xa=concat(xatt), vf=concat(vfull);
    std::string pev=conv_from(vf,pew);              // pe (depthwise)
    return conv_from(add(xa,pev),projw);            // proj
  };
  auto c3k=[&](const std::string&x,int nb){std::string last=conv(x);for(int i=0;i<nb;++i){std::string h=conv(last);h=conv(h);last=add(h,last);}std::string y2=conv(x);return conv(concat({last,y2}));};
  auto c3k2=[&](const std::string&x,int n,bool is_c3k,int inner)->std::string{std::string y0=conv(x);/*need channels*/ int64_t twoc=prov.convs[prov.i-1].w->shape[0],c=twoc/2;std::vector<std::string> outs={slice(y0,0,c),slice(y0,c,twoc)};std::string last=outs[1];for(int j=0;j<n;++j){last=is_c3k?c3k(last,inner):[&]{std::string h=conv(last);h=conv(h);return add(h,last);}();outs.push_back(last);}return conv(concat(outs));};
  auto sppf=[&](const std::string&x){std::string x1=conv(x),q1=maxpool(x1),q2=maxpool(q1),q3=maxpool(q2);return conv(concat({x1,q1,q2,q3}));};
  auto c2psa=[&](const std::string&x,int64_t H,int64_t W){std::string y=conv(x);int64_t twoc=prov.convs[prov.i-1].w->shape[0],dim=twoc/2;std::string a=slice(y,0,dim),b=slice(y,dim,twoc);b=add(b,attention(b,H,W));std::string f=conv(b);f=conv(f);b=add(b,f);return conv(concat({a,b}));};

  int64_t S=IMG;
  std::string x0=conv("images"),x1=conv(x0),x2=c3k2(x1,1,false,0),x3=conv(x2),x4=c3k2(x3,1,false,0),
    x5=conv(x4),x6=c3k2(x5,1,true,2),x7=conv(x6),x8=c3k2(x7,1,true,2),x9=sppf(x8),
    x10=c2psa(x9,S/32,S/32),x11=resize(x10),x12=concat({x11,x6}),x13=c3k2(x12,1,false,0),
    x14=resize(x13),x15=concat({x14,x4}),x16=c3k2(x15,1,false,0),
    x17=conv(x16),x18=concat({x17,x13}),x19=c3k2(x18,1,false,0),
    x20=conv(x19),x21=concat({x20,x10}),x22=c3k2(x21,1,true,2);
  std::string ins[3]={x16,x19,x22};
  for(int i=0;i<3;++i){ std::string hb=conv(ins[i]);hb=conv(hb);std::string box=conv(hb);
    std::string hc=conv(ins[i]);hc=conv(hc);hc=conv(hc);hc=conv(hc);std::string cls=conv(hc);
    int64_t st=8<<i;
    g.nodes.push_back({"Identity",U("id"),{box},{"box"+std::to_string(i)},{}});
    g.nodes.push_back({"Identity",U("id"),{cls},{"cls"+std::to_string(i)},{}});
    g.outputs.push_back({"box"+std::to_string(i),{1,4*(int64_t)16,S/st,S/st}});
    g.outputs.push_back({"cls"+std::to_string(i),{1,nc,S/st,S/st}});
  }
  save_onnx(g,"yolo11n.onnx");
  printf("wrote yolo11n.onnx (%zu nodes, %zu f-inits, %zu i-inits, consumed %zu/%zu convs)\n",
         g.nodes.size(),g.init_f.size(),g.init_i.size(),prov.i,prov.convs.size());
  return 0;
}
