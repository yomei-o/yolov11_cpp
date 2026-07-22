import os, numpy as np, onnx, onnxruntime as ort
HERE=os.path.dirname(os.path.abspath(__file__)); DN=os.path.join(HERE,"data_net")
m=onnx.load("yolo11n.onnx"); onnx.checker.check_model(m); print("onnx.checker: OK")
IMG,NB,NS,RM,Atot=open(os.path.join(DN,"io.txt")).read().split(); IMG,NB,NS,Atot=int(IMG),int(NB),int(NS),int(Atot)
x=np.fromfile(os.path.join(DN,"x.bin"),np.float32).reshape(1,3,IMG,IMG)
sess=ort.InferenceSession("yolo11n.onnx",providers=["CPUExecutionProvider"])
outs={o.name:v for o,v in zip(sess.get_outputs(),sess.run(None,{"images":x}))}
def pack(pre,C):
    d=np.zeros((C,Atot),np.float32);off=0
    for l in range(3): t=outs[f"{pre}{l}"][0];hw=t.shape[1]*t.shape[2];d[:,off:off+hw]=t.reshape(C,hw);off+=hw
    return d
boxes,scores=pack("box",NB),pack("cls",NS)
rb=np.fromfile(os.path.join(DN,"ref_boxes.bin"),np.float32).reshape(NB,Atot)
rs=np.fromfile(os.path.join(DN,"ref_scores.bin"),np.float32).reshape(NS,Atot)
db,ds=np.abs(boxes-rb).max(),np.abs(scores-rs).max()
print(f"onnxruntime boxes={db:.3e} scores={ds:.3e}  {'OK' if db<1e-3 and ds<1e-3 else 'FAIL'}")
print("\nyolo11 ONNX(write): C++ .onnx runs in onnxruntime == yolo11n forward" if db<1e-3 and ds<1e-3 else "MISMATCH")
