"""Real 60 s AGZ sample; 12 core runs, no compiler required."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT,MAIN,events_for,estimate,score,scenarios


def main():
    data=ROOT/'demo';out=ROOT/'results_demo';out.mkdir(exist_ok=True)
    gps=pd.read_csv(data/'gps.csv');imu=pd.read_csv(data/'imu.csv');ref=pd.read_csv(data/'reference.csv')
    seg=pd.read_csv(data/'segment.csv').iloc[0]
    cfg=json.loads((ROOT/'configs/locked.json').read_text())
    metrics=[]
    for sc in scenarios(seg):
        if sc['scenario_id'] not in ['normal','drop_5_at_20','noise_5_seed_11','delay_1.0']:continue
        prepared=events_for(gps,imu,seg,sc,cfg);trajectories=[]
        for method in MAIN:
            traj,updates,elapsed,details=estimate(method,prepared,cfg)
            _,m=score(traj,ref,seg,sc,method);metrics.extend(m)
            frame=pd.DataFrame(traj,columns=['t_s','east_m','north_m','up_m','ve_m_s','vn_m_s','vu_m_s'])
            frame['method']=method;trajectories.append(frame)
        pd.concat(trajectories).to_csv(out/f'{sc["scenario_id"]}_trajectories.csv',index=False)
    frame=pd.DataFrame(metrics);frame.to_csv(out/'metrics.csv',index=False)
    frame.query("phase=='overall'").pivot(index='scenario_id',columns='method',values='rmse_m').plot.bar(figsize=(9,4),rot=10)
    plt.ylabel('Horizontal RMSE (m)');plt.tight_layout();plt.savefig(out/'demo_comparison.png',dpi=200);plt.close()
    assert len(frame.query("phase=='overall'"))==12
    assert frame.coverage.min()==1.
    print('DEMO_COMPLETE: 12 real-data runs, full reference coverage')


if __name__=='__main__':main()
