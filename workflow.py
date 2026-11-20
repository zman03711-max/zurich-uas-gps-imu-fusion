"""One-command reproduction of the server coursework or a real-data demo."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true')
    p.add_argument('--source',default='data/AGZ',help='AGZ numerical-log directory or full AGZ.zip')
    args=p.parse_args()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLBACKEND':'Agg'}
    logs=ROOT/('run_logs_demo' if args.demo else 'run_logs');logs.mkdir(exist_ok=True)
    for folder in ['configs','results','outputs','figures']:(ROOT/folder).mkdir(exist_ok=True)
    commands=[[sys.executable,'code/test_pipeline.py']]
    if args.demo: commands.append([sys.executable,'code/demo.py'])
    else:
        source=ROOT/'third_party/KF-GINS-source'
        commands.extend([
            [sys.executable,'code/preprocess.py','preprocess','--source',args.source,'--out','data','--config','configs/preprocess.json'],
            [sys.executable,'code/experiment.py'],[sys.executable,'code/benchmarks.py'],
            ['cmake','-S',str(source),'-B',str(source/'build'),'-DCMAKE_BUILD_TYPE=Release'],
            ['cmake','--build',str(source/'build'),'-j','4'],
            [sys.executable,'code/public_kf_gins.py'],[sys.executable,'code/make_outputs.py'],
            [sys.executable,'code/position_only_control.py'],
            [sys.executable,'code/plot_fair.py'],
            [sys.executable,'code/finalize.py']])
    records=[]
    for i,command in enumerate(commands,1):
        print('RUN',*command,flush=True);tick=time.perf_counter()
        process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,errors='replace')
        records.append(dict(command=command,exit_code=process.returncode,wall_s=time.perf_counter()-tick))
        (logs/f'reproduce_{i:02d}.log').write_text(process.stdout+'\n'+process.stderr,encoding='utf-8')
        (logs/'reproduce.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        if process.returncode: print(process.stdout,process.stderr);raise SystemExit(process.returncode)
    print('ALL_REPRODUCTION_STEPS_PASSED')


if __name__=='__main__':main()
