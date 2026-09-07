"""Measured physical-property import and bench identification; no invented calibration."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.validation-deps'))
import numpy as np
import mujoco

def templates():
    m=mujoco.MjModel.from_xml_path(str(ROOT/'ts20_humanoid_v2.xml'))
    folder=ROOT/'calibration';folder.mkdir(exist_ok=True)
    physical={'schema_version':1,'measurement_source':None,'bodies':{}}
    for i in range(1,m.nbody):
        physical['bodies'][m.body(i).name]={
            'mass_kg':None,'com_m':None,'fullinertia_kg_m2':None,
            'note':'Local BODY frame. Inertia about COM: Ixx,Iyy,Izz,Ixy,Ixz,Iyz. Matrix off-diagonal convention, not unsigned CAD products.'}
    path=folder/'physical_measurements.template.json'
    if not path.exists():path.write_text(json.dumps(physical,indent=2),encoding='utf-8')
    path=folder/'bench_measurements.template.csv'
    if not path.exists():
        path.write_text('time_s,angle_rad,speed_rad_s,torque_Nm,external_load_Nm\n',encoding='utf-8')
    print('Templates:',folder)

def apply_physical(path):
    source=Path(path); spec=json.loads(source.read_text(encoding='utf-8'))
    if not spec.get('measurement_source'):raise ValueError('A traceable measurement_source is required.')
    tree=ET.parse(ROOT/'ts20_humanoid_v2.xml')
    root=tree.getroot();root.find('compiler').set('inertiafromgeom','auto')
    nodes={b.get('name'):b for b in root.findall('.//body')}
    unknown=set(spec.get('bodies',{}))-set(nodes)
    if unknown:raise ValueError('Unknown bodies: '+str(unknown))
    applied=[];missing=[]
    for name,node in nodes.items():
        props=spec.get('bodies',{}).get(name,{})
        fields=[props.get('mass_kg'),props.get('com_m'),props.get('fullinertia_kg_m2')]
        if all(v is None for v in fields):missing.append(name);continue
        if any(v is None for v in fields):raise ValueError('Incomplete measured properties for '+name)
        mass=float(fields[0]);com=np.array(fields[1],dtype=float);I=np.array(fields[2],dtype=float)
        if com.shape!=(3,) or I.shape!=(6,) or not np.isfinite(np.r_[mass,com,I]).all() or mass<=0:
            raise ValueError('Invalid units, shape or numeric values for '+name)
        matrix=np.array([[I[0],I[3],I[4]],[I[3],I[1],I[5]],[I[4],I[5],I[2]]])
        eigen=np.linalg.eigvalsh(matrix)
        if eigen[0]<=0 or eigen[2]>eigen[0]+eigen[1]+1e-12:
            raise ValueError('Nonphysical inertia tensor for '+name)
        old=node.find('inertial')
        if old is not None:node.remove(old)
        ET.SubElement(node,'inertial',pos=' '.join(map(str,com)),mass=str(mass),fullinertia=' '.join(map(str,I)))
        applied.append(name)
    if not applied:raise ValueError('No measured body properties provided; no calibrated model created.')
    ET.indent(root,space='  ')
    output=ROOT/'ts20_humanoid_measured.xml'
    xml=ET.tostring(root,encoding='unicode')
    m=mujoco.MjModel.from_xml_string(xml)  # Validate before saving.
    tree.write(output,encoding='utf-8',xml_declaration=True)
    status={'status':'measured_body_properties_imported','measurement_source':spec['measurement_source'],
            'applied_bodies':applied,'remaining_unmeasured_bodies':missing,'total_mass_kg':float(m.body_mass.sum()),
            'all_body_properties_measured':not missing,'actuator_dynamics_calibrated':False,
            'hardware_calibrated':False,'note':'Mass import alone does not validate actuator dynamics, sensors or motion.',
            'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    output.with_suffix('.calibration.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2));print(output)

def fit_bench(path,known_load_inertia=None):
    """Fit total output inertia + viscous/Coulomb friction in a rigid one-axis rig.

    external_load_Nm must be known opposing gravity/load torque. This does not
    identify a whole robot, torque-speed capability, thermal properties or gain delay.
    """
    with Path(path).open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if len(rows)<100:raise ValueError('At least 100 actual measured samples are required.')
    a=np.array([[float(r[k]) for k in ['time_s','angle_rad','speed_rad_s','torque_Nm','external_load_Nm']] for r in rows])
    if not np.isfinite(a).all() or np.any(np.diff(a[:,0])<=0):raise ValueError('Invalid samples or non-increasing time.')
    t,angle,v,torque,load=a.T
    if not (np.any(v>.1) and np.any(v<-.1)):raise ValueError('Excite both rotation directions above 0.1 rad/s.')
    # Derivative from measured velocity; omit endpoints and near-zero reversals.
    acc=np.gradient(v,t)
    valid=(np.abs(v)>.05)&(np.arange(len(v))>2)&(np.arange(len(v))<len(v)-3)
    X=np.c_[acc[valid],v[valid],np.sign(v[valid])]
    y=(torque-load)[valid]
    train=np.arange(len(y))%5!=0
    norms=np.linalg.norm(X[train],axis=0)
    if np.any(norms<1e-9):raise ValueError('Insufficient acceleration or velocity excitation.')
    condition=np.linalg.cond(X[train]/norms)
    if condition>100:raise ValueError('Measurements cannot distinguish inertia and friction; collect more varied motion.')
    params=np.linalg.lstsq(X[train],y[train],rcond=None)[0]
    if np.any(params<0):raise ValueError('Nonphysical fit. Check torque signs, load model, noise and excitation.')
    residual=y-X@params
    rmse=float(np.sqrt(np.mean(residual[~train]**2)))
    scale=max(float(np.sqrt(np.mean(y[~train]**2))),1e-9)
    result={'status':'candidate_fit_requires_independent_validation','source':str(Path(path).resolve()),
            'samples':len(rows),'total_output_inertia_kg_m2':float(params[0]),
            'viscous_Nm_per_rad_s':float(params[1]),'coulomb_Nm':float(params[2]),
            'holdout_rmse_Nm':rmse,'holdout_relative_rmse':rmse/scale,'scaled_condition':float(condition),
            'hardware_calibrated':False,
            'note':'Holdout samples are interleaved, not an independent experiment. Total inertia includes fixture/load. Do not copy total inertia into armature.'}
    if known_load_inertia is not None:
        residual_J=float(params[0]-known_load_inertia)
        if residual_J<0:raise ValueError('Known load inertia exceeds fitted total inertia.')
        result['candidate_reflected_rotor_inertia_kg_m2']=residual_J
    out=ROOT/'calibration'/'bench_fit.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--templates',action='store_true')
    p.add_argument('--apply-physical');p.add_argument('--fit-bench');p.add_argument('--known-load-inertia',type=float)
    args=p.parse_args()
    if args.apply_physical:apply_physical(args.apply_physical)
    elif args.fit_bench:fit_bench(args.fit_bench,args.known_load_inertia)
    else:templates()
