"""Floating-base TS20 motion tests, whole-body feedback and MIT torque interface.

No base forces, welded supports, state resets during motion, or prescribed qpos.
The QP only computes joint torques. MuJoCo advances actual contact dynamics.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time
import sys
import html
import contextlib
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.validation-deps'))
import numpy as np
import mujoco
import scipy.sparse as sp
import osqp

DT_CONTROL = .01
SQUAT_DEPTH_M = .050
MOTION_TIME_SCALE = .50  # 1.0 = current pace; lower values make commanded motions faster.

def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*x*(10 + x*(-15+6*x))

class Motion:
    """Slow quasi-static gait: transfer weight, swing, settle, repeat."""
    def __init__(self, model, data, mode):
        self.mode = mode
        self.com0 = data.subtree_com[1].copy()
        self.feet0 = np.array([data.site(s+'_sole').xpos.copy() for s in ['left','right']])
        self.duration = {'stand':8., 'squat':26., 'single':16., 'walk':34.}[mode] * MOTION_TIME_SCALE

    def sample(self, t):
        feet = self.feet0.copy()
        com = self.com0.copy()
        support = [True, True]
        phase = 'settle'
        if self.mode == 'squat' and t > 2*MOTION_TIME_SCALE:
            u = min((t-2*MOTION_TIME_SCALE)/(8*MOTION_TIME_SCALE), 3)
            cycle = min(int(u), 2)
            p = u-cycle
            depth = SQUAT_DEPTH_M * (.5-.5*math.cos(2*math.pi*p))
            com[2] -= depth
            phase = 'squat_cycle_'+str(cycle+1)
        elif self.mode == 'single':
            com[1] = self.feet0[0,1]*smooth((t-2*MOTION_TIME_SCALE)/(3*MOTION_TIME_SCALE))
            if t >= 6*MOTION_TIME_SCALE and t < 12*MOTION_TIME_SCALE:
                feet[1,2] += .025*smooth((t-6*MOTION_TIME_SCALE)/(2*MOTION_TIME_SCALE))
                support[1] = False
                phase = 'right_foot_lift_hold'
            elif t >= 12*MOTION_TIME_SCALE and t < 14*MOTION_TIME_SCALE:
                feet[1,2] += .025*(1-smooth((t-12*MOTION_TIME_SCALE)/(2*MOTION_TIME_SCALE)))
                support[1] = False
                phase = 'right_foot_lower'
            elif t >= 14*MOTION_TIME_SCALE:
                com[1] = self.feet0[0,1]*(1-smooth((t-14*MOTION_TIME_SCALE)/(2*MOTION_TIME_SCALE)))
                phase = 'return_double_support'
            else:
                phase = 'weight_transfer'
        elif self.mode == 'walk' and t > 2*MOTION_TIME_SCALE:
            # Four 6.4-second steps at the default 0.8 time scale, 35 mm advancement per foot placement.
            step = min(int((t-2*MOTION_TIME_SCALE)/(8*MOTION_TIME_SCALE)), 3)
            p = min((t-2*MOTION_TIME_SCALE)-8*MOTION_TIME_SCALE*step, 8*MOTION_TIME_SCALE)
            for k in range(step):
                feet[1 if k%2 == 0 else 0,0] += .035
            swing = 1 if step%2 == 0 else 0
            stance = 1-swing
            start_com = self.com0.copy()
            if step:
                start_com[0] = feet[:,0].mean() + self.com0[0]-self.feet0[:,0].mean()
            target = start_com.copy()
            target[:2] = feet[stance,:2]
            target[0] += self.com0[0]-self.feet0[:,0].mean()
            if p < 3*MOTION_TIME_SCALE:
                com = start_com + smooth(p/(3*MOTION_TIME_SCALE))*(target-start_com)
                phase = 'step_'+str(step+1)+'_transfer'
            elif p < 6*MOTION_TIME_SCALE:
                com = target
                u = (p-3*MOTION_TIME_SCALE)/(3*MOTION_TIME_SCALE)
                feet[swing,0] += .035*smooth(u)
                feet[swing,2] += .02*math.sin(math.pi*u)**2
                support[swing] = False
                phase = 'step_'+str(step+1)+'_swing'
            else:
                feet[swing,0] += .035
                end = start_com.copy()
                end[0] = feet[:,0].mean() + self.com0[0]-self.feet0[:,0].mean()
                com = target + smooth((p-6*MOTION_TIME_SCALE)/(2*MOTION_TIME_SCALE))*(end-target)
                phase = 'step_'+str(step+1)+'_settle'
        return com, feet, support, phase

    def target(self, t):
        c,f,s,p = self.sample(t)
        h=.002
        ca,fa,_,_=self.sample(max(t-h,0))
        cb,fb,_,_=self.sample(t+h)
        return c,f,s,p,(cb-ca)/(2*h),(fb-fa)/(2*h)

def orientation_error(rot):
    # Desired orientation is world identity; world-frame small-angle error.
    return -.5*np.array([rot[2,1]-rot[1,2],rot[0,2]-rot[2,0],rot[1,0]-rot[0,1]])

class Controller:
    def __init__(self, m, d, policy):
        self.m=m
        self.jids=m.actuator_trnid[:,0]
        self.dofs=m.jnt_dofadr[self.jids]
        self.qids=m.jnt_qposadr[self.jids]
        self.qhome=d.qpos[self.qids].copy()
        self.names=[m.joint(int(j)).name for j in self.jids]
        self.rated=m.actuator_user[:,0].copy()
        self.speed=m.actuator_user[:,1].copy()
        self.peak=m.actuator_forcerange[:,1].copy()
        self.policy=policy
        self.overload=np.zeros(m.nu)
        self.exhausted=np.zeros(m.nu,dtype=bool)
        config=json.loads((ROOT/'motor_config.json').read_text(encoding='utf-8'))
        self.kp=float(config['mit']['kp'])
        self.kd=float(config['mit']['kd'])
        if not 0<=self.kp<=64 or not 0<=self.kd<=2:raise ValueError('MIT gains exceed datasheet command range.')
        m.actuator_biasprm[:,1]=-self.kp
        m.actuator_biasprm[:,2]=-self.kd
        self.prevJ={}
        self.sites=[m.site(s+'_sole').id for s in ['left','right']]
        self.last_qp='not_run'
        self.failures=0
        self.qref=self.qhome.copy()
        self.vref=np.zeros(m.nu)
        self.ff=np.zeros(m.nu)
        self.last_time=-1

    def limits(self):
        if self.policy=='rated': return self.rated.copy()
        return np.where(self.exhausted,self.rated,self.peak)

    def jac(self,d,site):
        jp=np.zeros((3,self.m.nv)); jr=jp.copy()
        mujoco.mj_jacSite(self.m,d,jp,jr,site)
        return np.vstack((jp,jr))

    def jdot(self,key,J,vel):
        old=self.prevJ.get(key)
        self.prevJ[key]=J.copy()
        return np.zeros(J.shape[0]) if old is None else (J-old)@vel/DT_CONTROL

    def update(self,d,target):
        m=self.m
        c,feet,support,phase,cv,fv=target
        nv,nu=m.nv,m.nu
        active=[i for i in range(2) if support[i]]
        # Four unilateral point forces per supporting foot create a bounded CoP.
        Jpoints=[]
        for i in active:
            sid=self.sites[i]
            rot=d.site_xmat[sid].reshape(3,3)
            for x,y in [(-.048,-.025),(-.048,.025),(.048,-.025),(.048,.025)]:
                point=d.site_xpos[sid]+rot@np.array([x,y,-.001])
                jp=np.zeros((3,nv))
                mujoco.mj_jac(m,d,jp,None,point,int(m.site_bodyid[sid]))
                Jpoints.append(jp)
        Jforce=np.concatenate([j.T for j in Jpoints],axis=1)
        nf=Jforce.shape[1]
        nx=nv+nu+nf
        rows=[]; goals=[]
        def task(J,acc,weight):
            A=np.zeros((len(acc),nx)); A[:,:nv]=J
            rows.append(A*weight); goals.append(np.asarray(acc)*weight)
        # COM feedback regulates lateral weight shift without external support.
        jc=np.zeros((3,nv)); mujoco.mj_jacSubtreeCom(m,d,jc,1)
        ac=70*(c-d.subtree_com[1])+16*(cv-jc@d.qvel)-self.jdot('com',jc,d.qvel)
        task(jc,np.clip(ac,-8,8),12)
        jp=np.zeros((3,nv)); jr=jp.copy()
        mujoco.mj_jacBody(m,d,jp,jr,1)
        ar=90*orientation_error(d.xmat[1].reshape(3,3))-18*(jr@d.qvel)-self.jdot('pelvis',jr,d.qvel)
        task(jr,np.clip(ar,-30,30),8)
        for i,sid in enumerate(self.sites):
            J=self.jac(d,sid)
            err=np.r_[feet[i]-d.site_xpos[sid],orientation_error(d.site_xmat[sid].reshape(3,3))]
            desvel=np.r_[fv[i],np.zeros(3)]
            acc=140*err+24*(desvel-J@d.qvel)-self.jdot('foot'+str(i),J,d.qvel)
            task(J,np.clip(acc,-35,35),20 if support[i] else 12)
        # Low-priority posture keeps arms and redundant joints near home.
        post=np.zeros((nu,nv)); post[np.arange(nu),self.dofs]=1
        weights=np.array([.05 if any(s in n for s in ['hip','knee','ankle']) else 1.5 for n in self.names])
        task(post*weights[:,None],(20*(self.qhome-d.qpos[self.qids])-7*d.qvel[self.dofs])*weights,1.)
        A=np.vstack(rows); b=np.concatenate(goals)
        reg=np.r_[np.full(nv,1e-5),np.full(nu,.0002),np.full(nf,1e-6)]
        P=2*(A.T@A+np.diag(reg)); q=-2*A.T@b
        M=np.zeros((nv,nv)); mujoco.mj_fullM(m,d,M)
        S=np.zeros((nv,nu)); S[self.dofs,np.arange(nu)]=1
        dynamics=np.concatenate((M,-S,-Jforce),axis=1)
        bias=d.qfrc_bias-d.qfrc_passive
        cons=[dynamics]; low=[-bias]; high=[-bias]
        identity=np.eye(nx)
        lo=np.full(nx,-np.inf); hi=np.full(nx,np.inf)
        lo[:nv]=-100; hi[:nv]=100
        lim=self.limits(); lo[nv:nv+nu]=-lim; hi[nv:nv+nu]=lim
        # Anticipatory joint-limit bounds, including their current velocity.
        for k,j in enumerate(self.jids):
            dof=self.dofs[k]; angle=d.qpos[self.qids[k]]; v=d.qvel[dof]
            lo[dof]=np.clip(50*(m.jnt_range[j,0]-angle)-12*v,-100,100)
            hi[dof]=np.clip(50*(m.jnt_range[j,1]-angle)-12*v,-100,100)
        for k in range(nf//3):
            offset=nv+nu+3*k
            lo[offset+2]=0; hi[offset+2]=60
            fr=np.zeros((4,nx))
            for r,(axis,sign) in enumerate([(0,1),(0,-1),(1,1),(1,-1)]):
                fr[r,offset+axis]=sign; fr[r,offset+2]=-.6
            cons.append(fr); low.append(np.full(4,-np.inf)); high.append(np.zeros(4))
        cons.append(identity);low.append(lo);high.append(hi)
        solver=osqp.OSQP()
        solver.setup(P=sp.csc_matrix(np.triu(P)),q=q,A=sp.csc_matrix(np.vstack(cons)),
                     l=np.concatenate(low),u=np.concatenate(high),verbose=False,
                     eps_abs=2e-4,eps_rel=2e-4,max_iter=5000,polishing=False)
        result=solver.solve()
        self.last_qp=result.info.status
        if result.info.status_val not in (1,2) or result.x is None:
            self.failures+=1
            self.ff=np.zeros(nu);self.qref=d.qpos[self.qids].copy();self.vref=np.zeros(nu)
            return
        acceleration=result.x[:nv]
        torque=result.x[nv:nv+nu]
        self.qref=d.qpos[self.qids].copy()
        self.vref=d.qvel[self.dofs].copy()
        self.aref=acceleration[self.dofs].copy()
        self.ff=torque.copy()
        self.last_time=d.time

    def torque(self,d):
        elapsed=max(0,d.time-self.last_time)
        a=getattr(self,'aref',np.zeros(self.m.nu))
        qdes=self.qref+self.vref*elapsed+.5*a*elapsed**2
        vdes=self.vref+a*elapsed
        raw=self.kp*(qdes-d.qpos[self.qids])+self.kd*(vdes-d.qvel[self.dofs])+self.ff
        applied=np.clip(raw,-self.limits(),self.limits())
        return raw,applied,qdes,vdes

    def observe(self,actual):
        self.overload+=(np.abs(actual)>self.rated+1e-6)*self.m.opt.timestep
        self.exhausted|=self.overload>=.3-1e-12

def foot_contacts(m,d):
    forces=np.zeros(2)
    bad=[]
    for k in range(d.ncon):
        ct=d.contact[k]
        names=[m.geom(int(g)).name for g in [ct.geom1,ct.geom2]]
        if 'floor' not in names: continue
        other=names[1] if names[0]=='floor' else names[0]
        f=np.zeros(6);mujoco.mj_contactForce(m,d,k,f)
        if other in ('left_foot','right_foot'):
            forces[0 if other=='left_foot' else 1]+=max(0,f[0])
        elif f[0]>.1:
            bad.append(other)
    return forces,bad

def write_report(folder,summary,history,traces):
    # Standalone SVG charts embedded into a local HTML report; no network assets.
    charts=[]
    for col,title in [(1,'Pelvis height (m)'),(2,'COM lateral position (m)'),(3,'Left / right foot normal force (N)')]:
        data=np.array(history)
        series=[data[:,col]] if col!=3 else [data[:,3],data[:,4]]
        lo=min(float(v.min()) for v in series);hi=max(float(v.max()) for v in series)
        span=max(hi-lo,.001)
        paths=[]
        for i,v in enumerate(series):
            stride=max(1,len(v)//1000)
            pts=' '.join(f'{50+700*data[k,0]/max(data[-1,0],.01):.1f},{170-140*(v[k]-lo)/span:.1f}' for k in range(0,len(v),stride))
            paths.append(f'<polyline fill="none" stroke="{["#1676bd","#db6b24"][i]}" stroke-width="2" points="{pts}"/>')
        charts.append(f'<h3>{title}</h3><svg viewBox="0 0 800 210"><text x="5" y="25">{hi:.3f}</text><text x="5" y="175">{lo:.3f}</text><path d="M50 20V175H760" fill="none" stroke="#aaa"/>'+''.join(paths)+f'<text x="650" y="200">{data[-1,0]:.1f} seconds</text></svg>')
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(row[k]))}</td>' for k in ['joint','peak_abs_Nm','rms_Nm','peak_abs_rad_s','peak_abs_W','above_rated_s','saturation_s'])+'</tr>' for row in summary['joints'])
    body=f'''<!doctype html><meta charset="utf-8"><title>TS20 test report</title><style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#223}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}svg{{width:100%;max-width:800px}}pre{{white-space:pre-wrap}}select{{font:inherit;padding:8px}}</style>
    <h1>TS20 {summary['mode']} / {summary['status']}</h1><p>本结果基于估计质量与理想化机械结构。带载扭矩—速度和热模型尚未确定，尚未完成实机校准或机械安装验证。</p>
    <p>时长：{summary['duration_s']:.1f} 秒 · 质量：{summary['mass_kg']:.2f} kg · 前进：{summary['forward_progress_m']*1000:.1f} mm · 最大机身倾角：{summary['evidence']['max_tilt_deg']:.2f}°</p>
    <h2>关节曲线</h2><p>选择关节查看输出扭矩、角速度、机械功率和控制跟踪。扭矩虚线为额定值；角速度默认按实测范围缩放。功率不是电池耗电功率。</p><select id="joint"></select> <label><input type="checkbox" id="reference">显示空载转速参考线（不是带载上限）</label><div id="jointcharts"></div>
    <details><summary>详细验证参数与可追溯信息</summary><pre>{html.escape(json.dumps({k:v for k,v in summary.items() if k!='joints'},indent=2))}</pre></details>
    {''.join(charts)}<h2>Joint statistics (whole recorded run)</h2><table><tr><th>Joint</th><th>Peak Nm</th><th>RMS Nm</th><th>Peak rad/s</th><th>Peak |W|</th><th>Above rated s</th><th>Saturation s</th></tr>{table}</table>
    <p>Blue: left foot; orange: right foot. CSV files contain all samples and per-joint statistics. Mechanical power is not battery draw.</p>'''
    body+='<script>const traces='+json.dumps(traces,separators=(',',':'))+';const stats='+json.dumps(summary['joints'])+';</script>'
    body+='''<script>
    const select=document.getElementById('joint');
    Object.keys(traces).forEach(n=>{const o=document.createElement('option');o.textContent=n;o.value=n;select.append(o)});
    function chart(title,data,cols,ref){
      let vals=cols.flatMap(c=>data.map(r=>r[c]));if(ref)vals.push(-ref,ref);
      let lo=Math.min(...vals),hi=Math.max(...vals),span=Math.max(hi-lo,.0001);lo-=span*.08;hi+=span*.08;
      const x=t=>55+680*t/Math.max(data[data.length-1][0],.01),y=v=>175-145*(v-lo)/(hi-lo);
      let svg='<h3>'+title+'</h3><svg viewBox="0 0 800 210"><path d="M55 25V175H745" fill="none" stroke="#aaa"/>';
      svg+='<text x="0" y="30">'+hi.toFixed(3)+'</text><text x="0" y="175">'+lo.toFixed(3)+'</text>';
      if(ref)[-ref,ref].forEach(r=>{svg+='<path stroke="#c33" stroke-dasharray="5 4" d="M55 '+y(r)+'H745"/>'});
      cols.forEach((c,i)=>{svg+='<polyline fill="none" stroke="'+['#1676bd','#db6b24'][i]+'" stroke-width="1.5" points="'+data.map(r=>x(r[0]).toFixed(1)+','+y(r[c]).toFixed(1)).join(' ')+'"/>'});
      return svg+'<text x="600" y="202">'+data[data.length-1][0].toFixed(1)+' seconds</text></svg>';
    }
    function draw(){const n=select.value,d=traces[n],s=stats.find(s=>s.joint===n);document.getElementById('jointcharts').innerHTML=
      chart('输出扭矩 Nm（蓝）／请求扭矩（橙）',d,[1,6],s.rated_Nm)+
      chart('角速度 rad/s；空载参考 '+s.no_load_rad_s.toFixed(3),d,[2],document.getElementById('reference').checked?s.no_load_rad_s:null)+
      chart('机械功率 W：正值输出，负值吸收',d,[3],null)+
      chart('MIT 局部目标角度（蓝）／实际角度（橙），rad',d,[4,5],null);}
    select.value=Object.keys(traces).find(n=>n==='left_knee')||select.value;select.onchange=draw;document.getElementById('reference').onchange=draw;draw();
    </script>'''
    (folder/'report.html').write_text(body,encoding='utf-8')

def run(args):
    model_path=Path(getattr(args,'model',None) or ROOT/'ts20_humanoid_v2.xml').resolve()
    m=mujoco.MjModel.from_xml_path(str(model_path))
    d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,1)
    mujoco.mj_forward(m,d)
    control=Controller(m,d,args.policy)
    motion=Motion(m,d,args.motion)
    duration=args.duration or motion.duration
    folder=ROOT/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.motion+'_'+args.policy)
    folder.mkdir(parents=True)
    model_bytes=model_path.read_bytes()
    (folder/'model.xml').write_bytes(model_bytes)
    for source in ['run_tests.py','motor_config.json']:
        (folder/source).write_bytes((ROOT/source).read_bytes())
    accum={k:np.zeros(m.nu) for k in ['sq','peak','speed','power','over','sat','energy','negative','error']}
    history=[];traces={n:[] for n in control.names};replay=[]
    cycle_heights={str(i):[] for i in range(1,4)}
    swing_evidence={str(i):0. for i in range(1,5)}
    evidence={'single_support_s':0.,'right_clearance_max_m':0.,'left_clearance_max_m':0.,
              'pelvis_min_m':float(d.qpos[2]),'pelvis_max_m':float(d.qpos[2]),
              'max_tilt_deg':0.,'max_com_error_m':0.,'max_foot_error_m':0.}
    base_x0=float(d.qpos[0]); initial_feet=motion.feet0.copy()
    final_status='completed';reason=''
    viewer_context=contextlib.nullcontext(None)
    if args.viewer:
        from mujoco import viewer as mjviewer
        viewer_context=mjviewer.launch_passive(m,d)
    samples=0
    try:
        with (folder/'timeseries.csv').open('w',newline='',encoding='utf-8') as file, viewer_context as viewer:
            writer=csv.writer(file)
            writer.writerow(['time_s','phase','pelvis_x_m','pelvis_y_m','pelvis_z_m','com_x_m','com_y_m','com_z_m',
                             'left_normal_N','right_normal_N','joint','target_rad','angle_rad','target_rad_s','speed_rad_s',
                             'requested_Nm','applied_Nm','mechanical_W','above_rated','saturated','no_load_reference_exceeded','peak_budget_exhausted'])
            start=time.perf_counter()
            nsteps=int(round(duration/m.opt.timestep))
            for step in range(nsteps):
                mujoco.mj_forward(m,d)
                target=motion.target(d.time)
                if step%10==0: control.update(d,target)
                raw,tau,qdes,vdes=control.torque(d)
                # Affine actuator evaluates MIT feedback implicitly for stability.
                m.actuator_forcerange[:,0]=-control.limits()
                m.actuator_forcerange[:,1]=control.limits()
                d.ctrl[:]=control.kp*qdes+control.kd*vdes+control.ff
                mujoco.mj_forward(m,d)
                actual=d.actuator_force.copy()
                control.observe(actual)
                vel=d.qvel[control.dofs].copy()
                err=qdes-d.qpos[control.qids]
                power=actual*vel
                forces,bad=foot_contacts(m,d)
                tilt=math.degrees(math.acos(np.clip(d.xmat[1].reshape(3,3)[2,2],-1,1)))
                dt=m.opt.timestep
                above=np.abs(actual)>control.rated+1e-6
                saturated=np.abs(raw-actual)>1e-5
                accum['sq']+=actual**2*dt
                accum['peak']=np.maximum(accum['peak'],np.abs(actual))
                accum['speed']=np.maximum(accum['speed'],np.abs(vel))
                accum['power']=np.maximum(accum['power'],np.abs(power))
                accum['over']+=above*dt;accum['sat']+=saturated*dt
                accum['energy']+=np.maximum(power,0)*dt;accum['negative']+=np.minimum(power,0)*dt
                accum['error']+=err**2*dt
                feet=np.array([d.site(s+'_sole').xpos.copy() for s in ['left','right']])
                if args.motion=='squat' and d.time>=2*MOTION_TIME_SCALE:
                    ci=str(min(int((d.time-2*MOTION_TIME_SCALE)/(8*MOTION_TIME_SCALE))+1,3));cycle_heights[ci].append(float(d.qpos[2]))
                if args.motion=='walk' and '_swing' in target[3]:
                    si=int(target[3].split('_')[1]);sw=1 if si%2 else 0
                    if forces[1-sw]>5 and forces[sw]<.5 and feet[sw,2]-initial_feet[sw,2]>.012:
                        swing_evidence[str(si)]+=dt
                if forces[0]>5 and forces[1]<.5 and feet[1,2]-initial_feet[1,2]>.012:
                    evidence['single_support_s']+=dt
                for i,s in enumerate(['left','right']):
                    evidence[s+'_clearance_max_m']=max(evidence[s+'_clearance_max_m'],float(feet[i,2]-initial_feet[i,2]))
                evidence['pelvis_min_m']=min(evidence['pelvis_min_m'],float(d.qpos[2]))
                evidence['pelvis_max_m']=max(evidence['pelvis_max_m'],float(d.qpos[2]))
                evidence['max_tilt_deg']=max(evidence['max_tilt_deg'],tilt)
                evidence['max_com_error_m']=max(evidence['max_com_error_m'],float(np.linalg.norm(d.subtree_com[1]-target[0])))
                evidence['max_foot_error_m']=max(evidence['max_foot_error_m'],float(np.max(np.linalg.norm(feet-target[1],axis=1))))
                if step%10==0:
                    replay.append((float(d.time),d.qpos.copy(),d.qvel.copy(),d.ctrl.copy()))
                    history.append([float(d.time),float(d.qpos[2]),float(d.subtree_com[1,1]),*map(float,forces)])
                    for j,name in enumerate(control.names):
                        writer.writerow([round(d.time,6),target[3],*d.qpos[:3],*d.subtree_com[1],*forces,name,qdes[j],d.qpos[control.qids[j]],vdes[j],vel[j],raw[j],actual[j],power[j],int(above[j]),int(saturated[j]),int(abs(vel[j])>control.speed[j]),int(control.exhausted[j])])
                        if step%50==0:
                            traces[name].append([round(float(x),6) for x in [d.time,actual[j],vel[j],power[j],qdes[j],d.qpos[control.qids[j]],raw[j]]])
                samples+=1
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    final_status='failed';reason='nonfinite_state';break
                if d.qpos[2]<.17 or tilt>35 or bad:
                    final_status='failed';reason='fall_or_nonfoot_ground_contact:'+','.join(bad);break
                if control.failures>5:
                    final_status='failed';reason='QP_solver_failures';break
                mujoco.mj_step(m,d)
                if viewer and step%20==0:
                    if not viewer.is_running():
                        final_status='interrupted';reason='viewer_closed';break
                    viewer.sync()
                    remaining=d.time-(time.perf_counter()-start)
                    if remaining>0:time.sleep(min(remaining,.02))
    except KeyboardInterrupt:
        final_status='interrupted';reason='keyboard_interrupt'
    except Exception as exc:
        final_status='failed';reason='runtime_error:'+type(exc).__name__+':'+str(exc)
        (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    elapsed=samples*m.opt.timestep
    progress=float(d.qpos[0]-base_x0)
    full_duration=elapsed>=motion.duration-.02
    evidence['squat_cycle_depths_m']={k:max(v)-min(v) for k,v in cycle_heights.items() if v}
    evidence['walk_verified_swing_s']=swing_evidence
    if final_status=='completed':
        if not full_duration:final_status='partial';reason='shortened_test'
        elif args.motion=='single' and evidence['single_support_s']<2:
            final_status='failed';reason='insufficient_single_support'
        elif evidence['max_com_error_m']>.01 or evidence['max_foot_error_m']>.01 or evidence['max_tilt_deg']>10:
            final_status='failed';reason='excessive_task_tracking_error'
        elif args.motion=='walk' and (progress<.045 or min(swing_evidence.values())<.5):
            final_status='failed';reason='insufficient_walk_progress_or_clearance'
        elif args.motion=='squat' and (len(evidence['squat_cycle_depths_m'])!=3 or min(evidence['squat_cycle_depths_m'].values())<.025):
            final_status='failed';reason='insufficient_squat_depth'
        else:final_status='passed_simulation'
    stats=[]
    for j,name in enumerate(control.names):
        stats.append(dict(joint=name,variant=int(m.actuator_user[j,2]),no_load_rad_s=float(m.actuator_user[j,1]),rated_Nm=float(control.rated[j]),peak_abs_Nm=round(float(accum['peak'][j]),5),
                          rms_Nm=round(float(np.sqrt(accum['sq'][j]/max(elapsed,1e-9))),5),peak_abs_rad_s=round(float(accum['speed'][j]),5),
                          peak_abs_W=round(float(accum['power'][j]),5),above_rated_s=round(float(accum['over'][j]),4),saturation_s=round(float(accum['sat'][j]),4),
                          positive_mechanical_J=round(float(accum['energy'][j]),5),negative_mechanical_J=round(float(accum['negative'][j]),5),
                          mit_target_rms_error_rad=round(float(np.sqrt(accum['error'][j]/max(elapsed,1e-9))),6)))
    summary=dict(mode=args.motion,status=final_status,reason=reason,duration_s=elapsed,policy=args.policy,
                 mass_kg=float(m.body_mass.sum()),forward_progress_m=progress,evidence=evidence,qp_failures=control.failures,
                 mujoco_version=mujoco.__version__,model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                 controller_sha256=hashlib.sha256((ROOT/'run_tests.py').read_bytes()).hexdigest(),
                 external_base_assistance=False,self_collision_enabled=False,
                 hardware_calibrated=False,mechanical_installation_validated=False,
                 loaded_speed_envelope_known=False,thermal_model_known=False,joints=stats)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if replay:
        np.savez_compressed(folder/'replay.npz',time=np.array([r[0] for r in replay]),
                            qpos=np.array([r[1] for r in replay]),qvel=np.array([r[2] for r in replay]),ctrl=np.array([r[3] for r in replay]))
    with (folder/'joint_statistics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    if history:write_report(folder,summary,history,traces)
    print(json.dumps({k:v for k,v in summary.items() if k!='joints'},indent=2))
    print('REPORT:',folder/'report.html')
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--motion',choices=['stand','squat','single','walk'],default='squat')
    parser.add_argument('--policy',choices=['rated','peak-budget'],default='rated')
    parser.add_argument('--duration',type=float)
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--model',help='Optional XML with imported measured body properties')
    args=parser.parse_args()
    run(args)
