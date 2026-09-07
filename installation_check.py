"""Nominal TS20 envelope screening, explicitly not SolidWorks assembly validation."""
from pathlib import Path
import json
import math
import sys
import argparse
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.validation-deps'))
import numpy as np
import mujoco

def axis_quat(axis):
    axis=np.array(axis,dtype=float);axis/=np.linalg.norm(axis)
    z=np.array([0.,0.,1.]); dot=float(z@axis)
    if dot<-.99999:return np.array([0.,1.,0.,0.])
    q=np.r_[1+dot,np.cross(z,axis)];return q/np.linalg.norm(q)

def run(placements=None):
    tree=ET.parse(ROOT/'ts20_humanoid_v2.xml');root=tree.getroot()
    bodies={b.get('name'):b for b in root.findall('.//body')}
    m0=mujoco.MjModel.from_xml_path(str(ROOT/'ts20_humanoid_v2.xml'))
    source=json.loads(Path(placements).read_text(encoding='utf-8')) if placements else {}
    mapping=source.get('motors',{})
    expected=[m0.joint(int(j)).name for j in m0.actuator_trnid[:,0]]
    if set(mapping)-set(expected):raise ValueError('Placement file contains unknown joints.')
    assumed=[];template={'measurement_source':None,'motors':{}}
    for name in expected:
        jid=m0.joint(name).id;bodyname=m0.body(int(m0.jnt_bodyid[jid])).name
        default={'body':bodyname,'center_m':[0,0,0],'axis':m0.jnt_axis[jid].tolist()}
        template['motors'][name]={'body':bodyname,'center_m':None,'axis':None}
        entry=mapping.get(name,{})
        measured=entry.get('center_m') is not None and entry.get('axis') is not None
        if measured:
            if not source.get('measurement_source'):raise ValueError('measurement_source required for supplied motor placements.')
            spec=entry
        else:spec=default;assumed.append(name)
        if spec['body'] not in bodies:raise ValueError('Unknown mounting body '+spec['body'])
        center=np.array(spec['center_m'],dtype=float);axis=np.array(spec['axis'],dtype=float)
        if center.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[center,axis]).all() or np.linalg.norm(axis)<1e-8:
            raise ValueError('Invalid placement for '+name)
        quat=axis_quat(axis)
        ET.SubElement(bodies[spec['body']],'geom',name='envelope_'+name,type='cylinder',
                      size='.01 .0175',pos=' '.join(map(str,center)),quat=' '.join(map(str,quat)),
                      mass='0',contype='0',conaffinity='0',group='4',rgba='0.95 0.15 0.15 0.35')
    folder=ROOT/'installation';folder.mkdir(exist_ok=True)
    tp=folder/'motor_placements.template.json'
    if not tp.exists():tp.write_text(json.dumps(template,indent=2),encoding='utf-8')
    ET.indent(root,space='  ')
    out=folder/'motor_envelopes.xml';tree.write(out,encoding='utf-8',xml_declaration=True)
    m=mujoco.MjModel.from_xml_path(str(out));d=mujoco.MjData(m)
    gids=[m.geom('envelope_'+n).id for n in expected]
    intersections=[]
    # Neutral and crouch samples only. Real links/cables/fasteners are not known.
    for key in range(m.nkey):
        mujoco.mj_resetDataKeyframe(m,d,key);mujoco.mj_forward(m,d)
        for i,g1 in enumerate(gids):
            for j in range(i+1,len(gids)):
                g2=gids[j]
                distance=float(mujoco.mj_geomDistance(m,d,g1,g2,.01,None))
                # Exact-center cylinder queries can return zero even in overlap.
                # Intersections of inscribed balls give a sufficient overlap test.
                centers1=[d.geom_xpos[g1]+d.geom_xmat[g1].reshape(3,3)[:,2]*z for z in [-.0075,0,.0075]]
                centers2=[d.geom_xpos[g2]+d.geom_xmat[g2].reshape(3,3)[:,2]*z for z in [-.0075,0,.0075]]
                core_gap=min(np.linalg.norm(a-b)-.02 for a in centers1 for b in centers2)
                if distance<-.0001 or core_gap<-.0001:
                    intersections.append({'pose':m.key(key).name,'motor_a':expected[i],'motor_b':expected[j],
                                          'signed_distance_m':min(distance,float(core_gap)),
                                          'method':'signed_distance_or_inscribed_sphere_overlap',
                                          'placement_assumed':expected[i] in assumed or expected[j] in assumed})
    report={'status':'nominal_envelope_conflicts' if intersections else 'no_motor_envelope_conflicts_in_sampled_poses',
            'mechanical_installation_validated':False,'source_geometry':'datasheet nominal cylinder diameter20mm length35mm',
            'pose_count':m.nkey,'assumed_placements':assumed,'intersections':intersections,
            'missing_checks':['actual assembly axis offsets','real flange/housing shape','motor-to-link interference',
                              'fasteners and bearings','wire bend radius and routing','load ratings and fits',
                              'continuous full-range swept-volume collision','manufacturing tolerances'],
            'available_cad':'TS20.SLDPRT is one component, not a robot assembly; it has not been parsed.',
            'interpretation':'Assumed placements indicate the idealized skeleton is not a physically packaged assembly. Negative distances are envelope-screening estimates, not manufacturing clearances.'}
    (folder/'installation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    text=['# 电机安装占位检查','',f'采样姿态：{m.nkey}；电机包络相交记录：{len(intersections)}；未确定安装位置：{len(assumed)}。','',
          '**尚未通过机械安装验证。** 检查对象是 Ø20×35 mm 圆柱包络；默认将电机中心置于关节轴心。当前髋、肩和踝的轴心重合，会产生占位冲突。',
          '', '这不能证明真实设计一定冲突，也不能证明没有报告冲突的位置可以装配。需要真实电机位置、轴间偏移、支架、轴承、紧固件和线缆模型。',
          '', '|姿态|电机 A|电机 B|包络有符号距离 mm|','|---|---|---|---:|']
    text += [f"|{r['pose']}|{r['motor_a']}|{r['motor_b']}|{1000*r['signed_distance_m']:.2f}|" for r in intersections]
    (folder/'安装检查.md').write_text('\n'.join(text),encoding='utf-8')
    print('Motor envelope conflicts:',len(intersections),'Assumed placements:',len(assumed));print(folder/'安装检查.md')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--placements');a=p.parse_args();run(a.placements)
