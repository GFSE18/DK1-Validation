"""Generate the editable, standalone first-pass MJCF model from the shared chat."""
import math
import xml.etree.ElementTree as E
from pathlib import Path

TORSO_MASS_KG = 0.6

root = E.Element('mujoco', model='TS20 500mm humanoid - estimated prototype')
def add(parent, tag, **attrs):
    return E.SubElement(parent, tag, {k:str(v) for k,v in attrs.items()})
root.append(E.Comment(' SI units: metres, kg, seconds, radians. X forward, Y left, Z up. All masses, joint limits and gains are estimates. Primitive masses INCLUDE motor allocations; do not add motors again. No CAD meshes required. '))
add(root,'compiler',angle='radian',autolimits='true',inertiafromgeom='true')
add(root,'option',timestep='0.001',integrator='implicitfast',gravity='0 0 -9.81')
add(root,'size',nuser_actuator='3')
default=add(root,'default')
add(default,'joint',type='hinge',damping='0.02',armature='0.00001')
add(default,'geom',contype='2',conaffinity='1',friction='0.8 0.005 0.0001',rgba='0.45 0.55 0.65 1')
for name,peak,rated,speed,kp,kv in [('ts20_50',2,0.7,31.4,12,0.25),('ts20_100',4,1.5,15.7,35,0.6)]:
    d=add(default,'default',**{'class':name})
    add(d,'position',kp=kp,kv=kv,gear='1',ctrllimited='true',forcelimited='true',forcerange=f'-{peak} {peak}',user=f'{rated} {speed} {50 if peak==2 else 100}')
root.append(E.Comment(' Actuator user fields = rated output torque Nm, reference max output speed rad/s, reduction ratio. Values transcribed from shared chat, not independently verified datasheet values. Only peak torque is enforced. No thermal or torque-speed envelope. gear=1 because limits are at gearbox output. '))
visual=add(root,'visual'); add(visual,'global',azimuth='135',elevation='-15'); add(visual,'headlight',diffuse='0.7 0.7 0.7')
world=add(root,'worldbody')
add(world,'light',pos='0 -1 2',dir='0 0 -1')
add(world,'geom',name='floor',type='plane',size='2 2 0.05',contype='1',conaffinity='2',rgba='0.22 0.25 0.28 1')
pelvis=add(world,'body',name='pelvis',pos='0 0 0.295')
add(pelvis,'freejoint',name='floating_base')
def box(p,name,pos,size,mass,color=None):
    args=dict(name=name,type='box',pos=pos,size=size,mass=mass)
    if color: args['rgba']=color
    return add(p,'geom',**args)
box(pelvis,'pelvis_shape','0 0 0.0175','0.026 0.05 0.0175',0.35)
joints=[]
def jointbody(p,name,pos,axis,limits,strong=False):
    b=add(p,'body',name=name+'_link',pos=pos)
    ran=' '.join(f'{math.radians(x):.8f}' for x in limits)
    add(b,'joint',name=name,axis=axis,range=ran)
    joints.append((name,ran,'ts20_100' if strong else 'ts20_50'))
    return b
def motor_mass(b,name):
    add(b,'geom',name=name+'_housing',type='sphere',size='0.01',mass='0.035',rgba='0.18 0.2 0.22 1')
for side,y,color in [('left',0.04,'0.2 0.5 0.85 1'),('right',-0.04,'0.85 0.4 0.2 1')]:
    yaw=jointbody(pelvis,side+'_hip_yaw',f'0 {y} 0','0 0 1',(-45,45)); motor_mass(yaw,side+'_hip_yaw')
    roll=jointbody(yaw,side+'_hip_roll','0 0 0','1 0 0',(-35,35),True); motor_mass(roll,side+'_hip_roll')
    thigh=jointbody(roll,side+'_hip_pitch','0 0 0','0 1 0',(-90,45),True)
    box(thigh,side+'_thigh','0 0 -0.06','0.015 0.0125 0.06',0.18,color)
    shank=jointbody(thigh,side+'_knee','0 0 -0.12','0 1 0',(0,140),True)
    box(shank,side+'_shank','0 0 -0.0675','0.013 0.012 0.0675',0.16,color)
    ankle=jointbody(shank,side+'_ankle_pitch','0 0 -0.135','0 1 0',(-60,45),True); motor_mass(ankle,side+'_ankle_pitch')
    foot=jointbody(ankle,side+'_ankle_roll','0 0 0','1 0 0',(-30,30),True)
    box(foot,side+'_ankle_block','0 0 -0.015','0.012 0.012 0.015',0.035,color)
    box(foot,side+'_foot','0.015 0 -0.035','0.05 0.0275 0.005',0.085,color)
    add(foot,'site',name=side+'_sole',type='box',pos='0.015 0 -0.039',size='0.049 0.0265 0.002',rgba='0 1 0 0.2')
torso=jointbody(pelvis,'waist_yaw','0 0 0.035','0 0 1',(-45,45))
box(torso,'torso_shape','0 0 0.035','0.026 0.06 0.035',TORSO_MASS_KG)
for side,y in [('left',0.0725),('right',-0.0725)]:
    shoulder=jointbody(torso,side+'_shoulder_pitch',f'0 {y} 0.07','0 1 0',(-150,90)); motor_mass(shoulder,side+'_shoulder_pitch')
    roll=jointbody(shoulder,side+'_shoulder_roll','0 0 0','1 0 0',(-100,100)); motor_mass(roll,side+'_shoulder_roll')
    upper=jointbody(roll,side+'_shoulder_yaw','0 0 0','0 0 1',(-90,90))
    box(upper,side+'_upper_arm','0 0 -0.04','0.01 0.01 0.04',0.075)
    fore=jointbody(upper,side+'_elbow','0 0 -0.08','0 1 0',(-140,0))
    box(fore,side+'_forearm','0 0 -0.04','0.009 0.009 0.04',0.06)
    box(fore,side+'_hand','0 0 -0.10','0.01 0.015 0.02',0.015)
neck=jointbody(torso,'neck_yaw','0 0 0.085','0 0 1',(-70,70)); motor_mass(neck,'neck_yaw')
head=jointbody(neck,'neck_pitch','0 0 0','0 1 0',(-35,35))
box(head,'head_shape','0 0 0.0425','0.0275 0.0325 0.0425',0.155)
act=add(root,'actuator')
sensor=add(root,'sensor')
for name,ran,cls in joints:
    add(act,'position',name=name+'_servo',joint=name,ctrlrange=ran,**{'class':cls})
    add(sensor,'jointpos',name=name+'_angle',joint=name)
    add(sensor,'jointvel',name=name+'_speed',joint=name)
    add(sensor,'actuatorfrc',name=name+'_torque',actuator=name+'_servo')
for side in ['left','right']:
    add(sensor,'touch',name=side+'_foot_contact',site=side+'_sole')
keys=add(root,'keyframe')
zero=[0.0]*len(joints)
add(keys,'key',name='stand',qpos=' '.join(map(str,[0,0,.295,1,0,0,0]+zero)),ctrl=' '.join(map(str,zero)))
angles=[(-.25 if '_hip_pitch' in n or '_ankle_pitch' in n else .5 if '_knee' in n else 0) for n,_,_ in joints]
z=.04+.255*math.cos(.25)
x=-(.135-.12)*math.sin(.25)
add(keys,'key',name='shallow_squat',qpos=' '.join(map(str,[x,0,z,1,0,0,0]+angles)),ctrl=' '.join(map(str,angles)))
root.append(E.Comment(' Robot self-collision disabled via collision masks for this idealized co-located joint skeleton. All robot shapes can contact floor. Joint ranges and gains are preliminary. Knee uses positive flexion about +Y; squat hip/ankle angles are negative. Free base has no balance controller. '))
E.indent(root,space='  ')
path=Path(__file__).with_name('ts20_humanoid_500mm.xml')
E.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
print(path)
print(f'{len(joints)} actuated joints')
