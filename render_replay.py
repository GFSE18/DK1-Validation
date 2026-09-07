"""Render a snapshot from actually recorded simulation states, not a posed animation."""
from pathlib import Path
import argparse
import sys
import struct
import zlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.validation-deps'))
import numpy as np
import mujoco

def png(path,pixels):
    h,w,_=pixels.shape
    def chunk(name,data):return struct.pack('!I',len(data))+name+data+struct.pack('!I',zlib.crc32(name+data)&0xffffffff)
    raw=b''.join(b'\x00'+pixels[y].tobytes() for y in range(h))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))

def render(folder,t):
    folder=Path(folder)
    recording=np.load(folder/'replay.npz')
    frame=int(np.argmin(np.abs(recording['time']-t)))
    m=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
    m.vis.global_.offwidth=1000;m.vis.global_.offheight=800
    d=mujoco.MjData(m)
    d.qpos[:]=recording['qpos'][frame];d.qvel[:]=recording['qvel'][frame];d.ctrl[:]=recording['ctrl'][frame]
    mujoco.mj_forward(m,d)
    camera=mujoco.MjvCamera();camera.lookat[:]=[d.qpos[0],0,.24];camera.distance=.95;camera.azimuth=135;camera.elevation=-15
    with mujoco.Renderer(m,height=800,width=1000) as renderer:
        renderer.update_scene(d,camera=camera)
        image=renderer.render().copy()
    output=folder/f'snapshot_{recording["time"][frame]:.2f}s.png';png(output,image);print(output)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--time',type=float,default=9)
    a=p.parse_args();render(a.folder,a.time)
