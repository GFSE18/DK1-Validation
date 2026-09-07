"""Simple local menu for running and inspecting the robot tests."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    os.chdir(ROOT)
    print('TS20 机器人验证 / Robot validation')
    print('1 站立   2 连续蹲起   3 单腿站立   4 缓慢走路')
    print('5 全部测试（后台计算）   6 电机安装占位检查   7 校准数据模板')
    choice=input('请选择 / Choose [1-7]: ').strip()
    mapping={'1':'stand','2':'squat','3':'single','4':'walk'}
    commands=[]
    if choice in mapping:
        commands=[['run_tests.py','--motion',mapping[choice],'--viewer']]
    elif choice=='5':commands=[['run_tests.py','--motion',m] for m in mapping.values()]
    elif choice=='6':commands=[['installation_check.py']]
    elif choice=='7':commands=[['calibration.py','--templates']]
    else:print('无效选项 / Invalid choice');return
    for cmd in commands:
        code=subprocess.call([sys.executable,'-X','utf8',*cmd],cwd=ROOT)
        if code:print('运行异常；请查看上面的错误。 / Process error:',code);break
    if choice in mapping or choice=='5':
        reports=sorted((ROOT/'results').glob('*/report.html'),key=lambda p:p.stat().st_mtime)
        if reports:
            print('报告 / Report:',reports[-1])
            try:
                os.startfile(str(reports[-1]))
            except OSError:
                print('Please open this report manually:',reports[-1])
    elif choice=='6':
        target=ROOT/'installation'/'安装检查.md'
        try:os.startfile(str(target))
        except OSError:print('Please open this report manually:',target)
    elif choice=='7':
        target=ROOT/'calibration'
        try:os.startfile(str(target))
        except OSError:print('Templates are in:',target)
    try:
        input('按 Enter 关闭 / Press Enter to close...')
    except EOFError:
        pass

if __name__=='__main__':main()
