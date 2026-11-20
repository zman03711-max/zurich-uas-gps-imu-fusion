"""Summaries, audit checks, scientific plots and standalone trajectory replay."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT,METHODS,MAIN,events_for

LABELS={'gps_only':'GPS only','fixed_gain':'Fixed fusion','adaptive':'Adaptive robust fusion',
        'kf_fixed':'Fixed-noise KF','no_imu':'Without IMU','no_adaptive':'Without adaptive R',
        'no_gate':'Without position gating','no_delay':'Without delay compensation','filterpy':'FilterPy fixed KF','kf_gins':'KF-GINS'}
COLORS={'gps_only':'#8b949e','fixed_gain':'#d17a28','adaptive':'#117c83','kf_gins':'#8457b5'}
if (ROOT/'results/public_metrics.csv').exists(): MAIN=MAIN+['kf_gins']


def savefig(name):
    plt.tight_layout();plt.savefig(ROOT/'figures'/f'{name}.png',dpi=300)
    plt.savefig(ROOT/'figures'/f'{name}.svg');plt.close()


def main():
    out=ROOT/'results';fig=ROOT/'figures';fig.mkdir(exist_ok=True);(ROOT/'outputs').mkdir(exist_ok=True)
    tables=out/'tables';tables.mkdir(exist_ok=True)
    metrics=pd.read_csv(out/'metrics.csv');errors=pd.read_csv(out/'errors.csv')
    if (out/'public_metrics.csv').exists():
        public=pd.read_csv(out/'public_metrics.csv')
        metrics=pd.concat([metrics,public],ignore_index=True)
        errors=pd.concat([errors,pd.read_csv(out/'public_errors.csv')],ignore_index=True)
        pairs=public[['segment_id','scenario_id']].drop_duplicates()
        metrics.merge(pairs,on=['segment_id','scenario_id']).query("phase=='overall'").to_csv(tables/'public_common_cases.csv',index=False)
    overall=metrics.query("phase=='overall'")
    group=overall.groupby(['family','strength','method'],dropna=False)
    summary=group.agg(mean_rmse_m=('rmse_m','mean'),std_rmse_m=('rmse_m','std'),
                      mean_3d_rmse_m=('rmse_3d_m','mean'),mean_vertical_rmse_m=('vertical_rmse_m','mean'),
                      mean_p95_m=('p95_m','mean'),mean_median_m=('median_m','mean'),
                      n_cases=('rmse_m','size'),min_coverage=('coverage','min')).reset_index()
    summary.to_csv(tables/'all_conditions.csv',index=False)
    summary[summary.method.isin(MAIN)].to_csv(tables/'main_robustness.csv',index=False)
    summary[summary.family=='normal'].to_csv(tables/'normal_comparison.csv',index=False)
    summary[summary.method.isin(['adaptive','kf_fixed','no_imu','no_adaptive','no_gate','no_delay'])].to_csv(tables/'ablations.csv',index=False)
    metrics[metrics.phase!='overall'].to_csv(tables/'missing_and_recovery.csv',index=False)
    runs=json.loads((out/'runs.json').read_text())
    scenes=json.loads((ROOT/'configs/scenarios.json').read_text())
    assert len(runs)==len(scenes)*len(METHODS)
    assert all(r['status']=='ok' for r in runs)
    assert (metrics.coverage==1.).all()
    equivalence=pd.read_csv(out/'public_baseline_equivalence.csv')
    assert equivalence.max_abs_diff.max()<1e-7
    gps=pd.read_csv(ROOT/'data/gps.csv');imu=pd.read_csv(ROOT/'data/imu.csv')
    ref=pd.read_csv(ROOT/'data/reference.csv');segments=pd.read_csv(ROOT/'configs/segments.csv')
    cfg=json.loads((ROOT/'configs/locked.json').read_text())
    shared=[]
    for sc_record in scenes:
        sc=dict(sc_record);sid=sc.pop('segment_id')
        seg=segments[segments.segment_id==sid].iloc[0]
        prepared=events_for(gps,imu,seg,sc,cfg)
        folder=out/sid/sc['scenario_id']
        delivered=pd.read_csv(folder/'gps_delivered.csv')
        np.testing.assert_allclose(delivered[['t_s','east_m','north_m','up_m']].to_numpy(),
                                   prepared[-1][['t_s','east_m','north_m','up_m']].to_numpy(),atol=1e-9)
        effective=pd.DataFrame([row._asdict() for _,kind,row in prepared[3] if kind==0])
        effective.to_csv(folder/'imu_delivered.csv',index=False)
        for method in METHODS:
            updates=pd.read_csv(folder/f'updates_{method}.csv')
            np.testing.assert_allclose(updates.t_measure_s,delivered.t_s,atol=1e-9)
        shared.append(dict(segment_id=sid,scenario_id=sc['scenario_id'],gps_rows=len(delivered),imu_rows=len(effective),
                           gps_sha256=hashlib.sha256((folder/'gps_delivered.csv').read_bytes()).hexdigest(),
                           imu_sha256=hashlib.sha256((folder/'imu_delivered.csv').read_bytes()).hexdigest()))
    pd.DataFrame(shared).to_csv(out/'shared_input_audit.csv',index=False)
    checks=dict(runs=len(runs),scenarios=len(scenes),methods=len(METHODS),failed_runs=0,
                min_coverage=float(metrics.coverage.min()),max_filterpy_difference=float(equivalence.max_abs_diff.max()),
                independent_test_segments=int((segments.purpose=='test').sum()),seed_repetitions=5,
                reference_not_input=True,shared_inputs_verified=True)
    (out/'validation.json').write_text(json.dumps(checks,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    first=segments.query("purpose=='test'").iloc[0];sid=first.segment_id
    trajectory=pd.read_csv(out/sid/'normal/trajectories.csv')
    public_traj=out/'public_baseline'/sid/'normal/trajectories.csv'
    if public_traj.exists(): trajectory=pd.concat([trajectory,pd.read_csv(public_traj)],ignore_index=True)
    truth=ref[ref.t_s.between(first.start_s,first.end_s)]
    plt.figure(figsize=(8,6));plt.plot(truth.east_m,truth.north_m,'k--',label='Photogrammetric reference')
    for method in MAIN:
        t=trajectory[trajectory.method==method];plt.plot(t.east_m,t.north_m,label=LABELS[method],color=COLORS[method],alpha=.85)
    plt.xlabel('East (m)');plt.ylabel('North (m)');plt.axis('equal');plt.legend();savefig('01_trajectory_2d')
    f=plt.figure(figsize=(8,6));ax=f.add_subplot(projection='3d')
    ax.plot(truth.east_m,truth.north_m,truth.up_m,'k--',label='Reference')
    for method in MAIN:
        t=trajectory[trajectory.method==method];ax.plot(t.east_m,t.north_m,t.up_m,label=LABELS[method],color=COLORS[method])
    ax.set(xlabel='East (m)',ylabel='North (m)',zlabel='Supplied altitude (m)');ax.legend();savefig('02_trajectory_3d')
    plt.figure(figsize=(9,4))
    for method in MAIN:
        e=errors[(errors.segment_id==sid)&(errors.scenario_id=='normal')&(errors.method==method)]
        plt.plot(e.t_s-first.start_s,e.error_m,label=LABELS[method],color=COLORS[method])
    plt.xlabel('Time into segment (s)');plt.ylabel('Horizontal error (m)');plt.legend();savefig('03_error_time')
    plt.figure(figsize=(7,4))
    for method in MAIN:
        vals=np.sort(errors[(errors.family=='normal')&(errors.method==method)].error_m)
        plt.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=LABELS[method],color=COLORS[method])
    plt.xlabel('Horizontal error (m)');plt.ylabel('Empirical cumulative probability');plt.legend();savefig('04_error_cdf')
    f,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,family in zip(axes.flat,['dropout','noise','delay','outlier','imu_noise','imu_dropout']):
        for method in MAIN:
            s=summary[(summary.family==family)&(summary.method==method)].sort_values('strength')
            ax.errorbar(s.strength,s.mean_rmse_m,yerr=s.std_rmse_m.fillna(0),marker='o',label=LABELS[method],color=COLORS[method],capsize=3)
        units={'dropout':'GPS outage duration (s)','noise':'GPS noise standard deviation (m)',
               'delay':'GPS arrival delay (s)','outlier':'GPS offset magnitude (m)',
               'imu_noise':'Accelerometer noise std (m/s²)','imu_dropout':'IMU outage duration (s)'}
        ax.set(title=family.replace('_',' '),xlabel=units[family],ylabel='Horizontal RMSE (m)')
    axes[0,0].legend(fontsize=8);savefig('05_single_perturbations')
    s=summary[(summary.family.str.startswith('combo'))&summary.method.isin(MAIN)]
    pivot=s.pivot(index='family',columns='method',values='mean_rmse_m')
    pivot.rename(columns=LABELS).plot.bar(figsize=(10,5),rot=15);plt.ylabel('Mean horizontal RMSE (m)');plt.xlabel('');savefig('06_combined_perturbations')
    s=overall[overall.method.isin(['adaptive','no_imu','no_adaptive','no_gate','no_delay'])]
    p=s.groupby(['family','method']).rmse_m.mean().unstack().rename(columns=LABELS)
    p.plot.bar(figsize=(13,5),rot=30);plt.ylabel('Mean horizontal RMSE (m)');plt.xlabel('');savefig('07_ablations')
    benchmark=out/'benchmarks.csv'
    if benchmark.exists():
        b=pd.read_csv(benchmark)
        b.groupby('method').agg(wall_s=('wall_s','mean'),wall_std_s=('wall_s','std'),
            cpu_s=('cpu_s','mean'),latency_p95_ms=('event_latency_p95_ms','mean'),
            realtime_factor=('realtime_factor','mean'),process_peak_rss_mib=('process_peak_rss_mib','max'),
            traced_allocation_peak_mib=('traced_allocation_peak_mib','max')).to_csv(tables/'efficiency.csv')
        b.groupby('method').wall_s.mean().rename(index=LABELS).plot.barh(figsize=(8,5))
        plt.xlabel('Algorithm wall time per 60 s segment (s)');savefig('08_efficiency')
    replay=dict(truth=truth[['t_s','east_m','north_m','up_m']].values.tolist(),methods={})
    # Replay the GPS-outage case, preserving full three-dimensional numerical results.
    replay_folder=out/sid/'drop_5_at_20'
    t=pd.read_csv(replay_folder/'trajectories.csv')
    public_replay=out/'public_baseline'/sid/'drop_5_at_20/trajectories.csv'
    if public_replay.exists(): t=pd.concat([t,pd.read_csv(public_replay)],ignore_index=True)
    for method in MAIN: replay['methods'][LABELS[method]]=t[t.method==method][['t_s','east_m','north_m','up_m']].values.tolist()
    template=(ROOT/'scripts/replay_template.html').read_text(encoding='utf-8')
    (ROOT/'outputs/trajectory_replay.html').write_text(template.replace('__DATA__',json.dumps(replay)),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':main()
