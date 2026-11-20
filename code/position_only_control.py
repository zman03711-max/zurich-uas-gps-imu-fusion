"""Common GNSS-position-only control; same initial state and reference epochs."""
import json
import numpy as np
import pandas as pd
from experiment import ROOT,events_for,estimate,scenarios,score


def hold_position(prepared):
    position=prepared[0][:3].copy();rows=[]
    for t,kind,row in prepared[3]:
        if kind==1: position=np.array([row.east_m,row.north_m,row.up_m])
        if kind==2: rows.append([t,*position,0.,0.,0.])
    return np.array(rows)


def main():
    gps=pd.read_csv(ROOT/'data/gps.csv');imu=pd.read_csv(ROOT/'data/imu.csv');ref=pd.read_csv(ROOT/'data/reference.csv')
    cfg=json.loads((ROOT/'configs/locked.json').read_text())
    # Position likelihood unchanged; effectively disable the velocity likelihood.
    cfg.update(sigma_velocity=1e8,velocity_weight=0.)
    segments=pd.read_csv(ROOT/'configs/segments.csv').query("purpose=='test'")
    all_metrics=[];all_errors=[];checks=[]
    for seg in segments.itertuples():
        for sc in scenarios(seg):
            if sc['family'] not in ['normal','noise','outlier','dropout']:continue
            prepared=events_for(gps,imu,seg,sc,cfg)
            folder=ROOT/'results/position_only'/seg.segment_id/sc['scenario_id'];folder.mkdir(parents=True,exist_ok=True)
            outputs=[]
            for method in ['gps_only','fixed_gain','kf_fixed','adaptive']:
                trajectory=hold_position(prepared) if method=='gps_only' else estimate(method,prepared,cfg)[0]
                e,m=score(trajectory,ref,seg,sc,method);all_errors.append(e);all_metrics.extend(m)
                frame=pd.DataFrame(trajectory,columns=['t_s','east_m','north_m','up_m','ve_m_s','vn_m_s','vu_m_s'])
                frame['method']=method;outputs.append(frame)
            pd.concat(outputs).to_csv(folder/'trajectories.csv',index=False)
            print(seg.segment_id,sc['scenario_id'],'position-only complete',flush=True)
        # Check that subsequent GPS velocity fields cannot materially affect controls.
        sc=dict(family='normal',strength=0.,seed=0,scenario_id='normal')
        a=estimate('adaptive',events_for(gps,imu,seg,sc,cfg),cfg)[0]
        changed=gps.copy();changed.loc[changed.t_s>seg.start_s,['ve_m_s','vn_m_s','vu_m_s']]+=1000.
        b=estimate('adaptive',events_for(changed,imu,seg,sc,cfg),cfg)[0]
        diff=float(np.max(np.abs(a-b)));assert diff<1e-7
        checks.append(dict(segment_id=seg.segment_id,max_effect_of_1000_m_s_velocity_change=diff))
    metrics=pd.DataFrame(all_metrics);metrics.to_csv(ROOT/'results/position_only_metrics.csv',index=False)
    pd.concat(all_errors).to_csv(ROOT/'results/position_only_errors.csv',index=False)
    public=pd.read_csv(ROOT/'results/public_metrics.csv')
    combined=pd.concat([metrics,public],ignore_index=True).query("phase=='overall'")
    combined.groupby(['family','strength','method']).agg(mean_rmse_m=('rmse_m','mean'),std_rmse_m=('rmse_m','std'),
        mean_3d_rmse_m=('rmse_3d_m','mean'),n_cases=('rmse_m','size')).to_csv(ROOT/'results/tables/position_only_fair_comparison.csv')
    (ROOT/'results/position_only_validation.json').write_text(json.dumps(checks,indent=2))
    print('POSITION_ONLY_COMPLETE',len(metrics.query("phase=='overall'")),flush=True)


if __name__=='__main__':main()
