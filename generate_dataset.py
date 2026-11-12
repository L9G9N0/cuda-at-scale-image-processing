import argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument("--output-dir",default="data/input"); p.add_argument("--count",type=int,default=100)
p.add_argument("--width",type=int,default=256); p.add_argument("--height",type=int,default=256)
a=p.parse_args()
if min(a.count,a.width,a.height)<1: p.error("all dimensions/count must be positive")
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
y,x=np.mgrid[0:a.height,0:a.width]
for i in range(a.count):
 rng=np.random.default_rng(2026+i)
 im=np.empty((a.height,a.width,3),dtype=np.uint8)
 im[:,:,0]=(x*(i+1)+y*3)%256; im[:,:,1]=(y*(i+2)+40)%256; im[:,:,2]=((x+y)*2+i*11)%256
 im=np.clip(im.astype(np.uint16)+rng.integers(0,32,size=im.shape),0,255).astype(np.uint8)
 with (out/f"image_{i:04d}.ppm").open("wb") as f:
  f.write(f"P6\n{a.width} {a.height}\n255\n".encode()); f.write(im.tobytes())
print(f"Generated {a.count} images at {a.width}x{a.height} in {out}")
