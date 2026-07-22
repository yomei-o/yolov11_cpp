import os, numpy as np, torch, torch.nn.functional as F
D=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_gconv"); os.makedirs(D,exist_ok=True)
torch.manual_seed(0)
def run(tag,C,Cout,g):
    x=torch.randn(1,C,10,10,dtype=torch.float64,requires_grad=True)
    w=torch.randn(Cout,C//g,3,3,dtype=torch.float64,requires_grad=True)
    b=torch.randn(Cout,dtype=torch.float64,requires_grad=True)
    gy=torch.randn(1,Cout,10,10,dtype=torch.float64)
    y=F.conv2d(x,w,b,stride=1,padding=1,groups=g); (y*gy).sum().backward()
    for n,t in [("x",x),("w",w),("b",b),("gy",gy),("y",y),("dx",x.grad),("dw",w.grad),("db",b.grad)]:
        t.detach().float().numpy().tofile(os.path.join(D,f"{tag}_{n}.bin"))
    return f"{tag} {C} {Cout} {g}"
lines=[run("dw",8,8,8), run("g2",8,16,2)]
open(os.path.join(D,"meta.txt"),"w").write("\n".join(lines)+"\n")
print("gconv ref:",lines)
