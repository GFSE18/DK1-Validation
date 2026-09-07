"""Re-render report presentation from existing telemetry without re-running physics."""
from pathlib import Path
import csv
import json
import html
import hashlib
from run_tests import ROOT,write_report

def refresh(folder):
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    names=[j['joint'] for j in summary['joints']]
    traces={n:[] for n in names};history=[]
    with (folder/'timeseries.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t=float(row['time_s']);name=row['joint']
            if name==names[0]:history.append([t,*[float(row[k]) for k in ['pelvis_z_m','com_y_m','left_normal_N','right_normal_N']]])
            if round(t*100)%5==0:
                traces[name].append([t,*[float(row[k]) for k in ['applied_Nm','speed_rad_s','mechanical_W','target_rad','angle_rad','requested_Nm']]])
    write_report(folder,summary,history,traces)
    pictures=sorted(folder.glob('snapshot_*.png'))
    if pictures:
        path=folder/'report.html'
        body=path.read_text(encoding='utf-8')
        images='<h2>实际仿真状态截图</h2>'+''.join(f'<img style="max-width:700px;width:100%" src="{p.name}" alt="{p.stem}">' for p in pictures)
        body=body.replace('<h2>关节曲线</h2>',images+'<h2>关节曲线</h2>')
        path.write_text(body,encoding='utf-8')

def index():
    latest={}
    current_model_sha=hashlib.sha256((ROOT/'ts20_humanoid_v2.xml').read_bytes()).hexdigest()
    for path in sorted((ROOT/'results').glob('*/summary.json')):
        s=json.loads(path.read_text(encoding='utf-8'))
        if s.get('model_sha256') != current_model_sha:continue
        if s['policy']=='rated' and s['status']=='passed_simulation':latest[s['mode']]=(path.parent,s)
    rows=[]
    for mode in ['stand','squat','single','walk']:
        if mode not in latest:continue
        folder,s=latest[mode];refresh(folder)
        maxj=max(s['joints'],key=lambda j:j['peak_abs_Nm'])
        rows.append(f'<tr><td>{dict(stand="站立",squat="连续蹲起",single="单腿站立",walk="缓慢走路")[mode]}</td><td>{s["duration_s"]:.0f} 秒</td><td>{maxj["joint"]}: {maxj["peak_abs_Nm"]:.3f} Nm</td><td><a href="{folder.name}/report.html">查看曲线和统计</a></td></tr>')
    body='''<!doctype html><meta charset="utf-8"><title>TS20 验证结果</title><style>body{font:17px system-ui;max-width:1000px;margin:60px auto;padding:20px;color:#223}table{width:100%;border-collapse:collapse}td,th{padding:16px;text-align:left;border-bottom:1px solid #ddd}a{color:#1676bd}</style>
    <h1>TS20 模型 v2 · 仿真验证结果</h1><p>4.11 kg 估计质量（躯干 2.0 kg），额定扭矩策略，无外部机身辅助，机器人自身碰撞关闭。</p>
    <table><tr><th>动作</th><th>时长</th><th>本次最大关节扭矩</th><th>报告</th></tr>'''+''.join(rows)+'''</table>
    <h2>尚未完成的验证</h2><p>真实带载扭矩—速度曲线和热模型未知；实机校准等待测量数据。安装占位检查发现默认髋、肩、踝、颈等轴心处的圆柱包络相交，不能判定机械装配通过。</p>
    <p>请双击工作目录中的 Start_Tests.cmd 运行新测试，参阅 使用说明.md。数据报告保存在 results 中；本页汇总生成时最新通过的额定扭矩测试，不代表所有历史测试都通过。</p>'''
    (ROOT/'results'/'index.html').write_text(body,encoding='utf-8')
    print(ROOT/'results'/'index.html')

if __name__=='__main__':index()
