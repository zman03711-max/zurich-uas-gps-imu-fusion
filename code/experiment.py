"""Reproducible Zurich GPS/IMU coursework. No reference access in estimate()."""
from pathlib import Path
import argparse, json, time, platform, sys, hashlib, importlib.metadata
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from filterpy.kalman import KalmanFilter

ROOT=Path(__file__).resolve().parents[1]
METHODS=['gps_only','fixed_gain','kf_fixed','adaptive','no_imu','no_adaptive','no_gate','no_delay','filterpy']
MAIN=['gps_only','fixed_gain','adaptive']
DEFAULT=dict(position_weight=.2,velocity_weight=.2,sigma_position=3.,sigma_velocity=.5,
             sigma_acceleration=.8,tilt_tau_s=2.,adaptive_cap=100.,nis_threshold=5.991,
             max_gap_s=.35,output_period_s=.1,initialization_s=5.,gate_threshold=100.,
             delayed_velocity_sigma=.5,imu_acceleration_limit=6.)


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi


class Attitude:
    """Raw FRD gyro propagation with accelerometer tilt correction; no pose file.

    Units m/s2, rad/s are adopted PX4 conventions, not recovered acquisition metadata.
    Gravity tilt correction assumes low translational dynamics. Yaw is not observable
    from acceleration; initialize from GPS course when speed >=0.8m/s, else zero.
    """
    def __init__(self,initial_imu,initial_gps,cfg):
        f=initial_imu[['ax','ay','az']].median().to_numpy()
        roll=np.arctan2(-f[1],-f[2]); pitch=np.arctan2(f[0],np.hypot(f[1],f[2]))
        ve,vn=initial_gps[['ve_m_s','vn_m_s']].iloc[-1]
        speed=np.hypot(ve,vn)
        yaw=np.arctan2(ve,vn) if speed>=.8 else 0.
        self.rotation=Rotation.from_euler('xyz',[roll,pitch,yaw])
        stationary=(np.hypot(initial_gps.ve_m_s,initial_gps.vn_m_s).max()<.5 and
                    np.linalg.norm(initial_imu[['ax','ay','az']],axis=1).std()<.5 and
                    initial_imu[['gx','gy','gz']].std().max()<.15)
        self.bias=initial_imu[['gx','gy','gz']].median().to_numpy() if stationary else np.zeros(3)
        self.details=dict(yaw_observable_from_initial_course=bool(speed>=.8),initial_yaw_rad=float(yaw),
                          stationary_bias_initialization=bool(stationary),gyro_bias_rad_s=self.bias.tolist(),
                          initial_quaternion_xyzw=self.rotation.as_quat().tolist())
        self.cfg=cfg

    def step(self,row,dt):
        f=np.array([row.ax,row.ay,row.az]); w=np.array([row.gx,row.gy,row.gz])-self.bias
        if dt>0:
            self.rotation=self.rotation*Rotation.from_rotvec(w*dt)
            if 7<np.linalg.norm(f)<12:
                r,p,y=self.rotation.as_euler('xyz')
                mr=np.arctan2(-f[1],-f[2]); mp=np.arctan2(f[0],np.hypot(f[1],f[2]))
                k=1-np.exp(-dt/self.cfg['tilt_tau_s'])
                self.rotation=Rotation.from_euler('xyz',[r+k*wrap(mr-r),p+k*wrap(mp-p),y])
        a=self.rotation.apply(f)+[0,0,9.80665]
        return a[[1,0,2]]*[1,1,-1]


def disturbed(gps,segment,scenario):
    """Materialize one identical observation stream for all methods."""
    start=float(segment.start_s); end=float(segment.end_s)
    rows=gps[(gps.t_s>start)&(gps.t_s<=end)].copy()
    a,b=scenario.get('window',[np.inf,np.inf])
    inwindow=(rows.t_s>=a)&(rows.t_s<b)
    if scenario['family'] in ['dropout','combo_dropout_noise']: rows=rows[~inwindow].copy()
    rng=np.random.default_rng(scenario['seed'])
    # Noise applies after a protected 10s warmup, not during initialization.
    if scenario['family'] in ['noise','combo_dropout_noise','combo_noise_delay']:
        active=rows.t_s>=start+10
        sigma=scenario.get('gps_noise_m',scenario['strength'])
        rows.loc[active,['east_m','north_m','up_m']]+=rng.normal(0,sigma,(int(active.sum()),3))
    if scenario['family'] in ['outlier','combo_outlier_delay']:
        active=(rows.t_s>=start+20)&(rows.t_s<start+23)
        rows.loc[active,['east_m','north_m']]+=np.array([scenario.get('outlier_m',scenario['strength']),0.])
    rows['t_arrival_s']=rows.t_s
    if scenario['family'] in ['delay','combo_noise_delay','combo_outlier_delay']:
        rows.loc[rows.t_s>=start+10,'t_arrival_s']+=scenario.get('delay_s',scenario['strength'])
    return rows[rows.t_arrival_s<=end].sort_values('t_arrival_s')


def events_for(gps,imu,segment,scenario,cfg):
    start,end=float(segment.start_s),float(segment.end_s)
    initial_gps=gps[gps.t_s<=start].tail(5)
    initial_imu=imu[(imu.t_s>=start-cfg['initialization_s'])&(imu.t_s<=start)&imu.valid]
    if len(initial_imu)<20 or initial_gps.empty: raise ValueError('Insufficient causal initialization')
    g0=initial_gps.iloc[-1]
    if start-g0.t_s>2: raise ValueError('Stale GPS initializer')
    att=Attitude(initial_imu,initial_gps,cfg)
    old=initial_imu.iloc[-1]
    init=np.array([g0.east_m,g0.north_m,g0.up_m,g0.ve_m_s,g0.vn_m_s,g0.vu_m_s])
    init[:3]+=init[3:]*(start-g0.t_s)
    incoming=disturbed(gps,segment,scenario)
    working_imu=imu[(imu.t_s>start)&(imu.t_s<=end)].copy()
    if scenario['family'] in ['imu_noise','combo_imu_noise_dropout']:
        active=working_imu.t_s>=start+10
        rng=np.random.default_rng(scenario['seed']+100000)
        sigma=scenario.get('imu_noise_m_s2',scenario['strength'])
        working_imu.loc[active,['ax','ay','az']]+=rng.normal(0,sigma,(int(active.sum()),3))
        working_imu.loc[active,['gx','gy','gz']]+=rng.normal(0,.02*sigma,(int(active.sum()),3))
    if scenario['family']=='imu_dropout':
        a,b=scenario['window'];working_imu=working_imu[~working_imu.t_s.between(a,b,inclusive='left')]
    if scenario['family']=='combo_imu_noise_dropout':
        a,b=scenario['window'];incoming=incoming[~incoming.t_s.between(a,b,inclusive='left')]
    events=[(start,2,None),(end,2,None)]
    events.extend((float(r.t_s),0,r) for r in working_imu.itertuples())
    events.extend((float(r.t_arrival_s),1,r) for r in incoming.itertuples())
    events.extend((round(float(t),9),2,None) for t in np.arange(start+cfg['output_period_s'],end,cfg['output_period_s']))
    events.sort(key=lambda x:(x[0],x[1]))
    return init,att,old,events,incoming


def estimate(method,prepared,cfg,record=True):
    """Each independent method has identical causal controls, initialization and data.

    Timing includes attitude processing, propagation, updates, output construction.
    No file I/O, disturbances, or scoring occurs inside this function.
    """
    init,att_init,old,events,_=prepared
    import copy
    att=copy.deepcopy(att_init)
    x=init.copy(); P=np.diag([9.]*3+[1.]*3)
    R0=np.diag([cfg['sigma_position']**2]*3+[cfg['sigma_velocity']**2]*3)
    I=np.eye(6)
    robust=method in ['adaptive','no_imu','no_adaptive','no_gate','no_delay']
    public=KalmanFilter(dim_x=6,dim_z=6) if method=='filterpy' else None
    if public:
        public.x=x.copy();public.P=P.copy();public.H=I.copy();public.R=R0.copy()
    last=events[0][0]; held=old
    outputs=[]; updates=[]
    tick=time.perf_counter();cpu_tick=time.process_time();step_latencies=[]
    for t,kind,row in events:
        step_tick=time.perf_counter()
        dt=t-last
        if dt>1e-12:
            stale=not bool(held.valid) or t-float(held.t_s)>cfg['max_gap_s']+1e-8
            a=np.zeros(3) if method in ['gps_only','no_imu'] or stale else att.step(held,dt)
            if robust and np.linalg.norm(a)>cfg['imu_acceleration_limit']:
                a*=cfg['imu_acceleration_limit']/np.linalg.norm(a)
            F=I.copy(); F[:3,3:]=np.eye(3)*dt
            B=np.vstack([np.eye(3)*(.5*dt*dt),np.eye(3)*dt])
            Q=B@B.T*cfg['sigma_acceleration']**2*(25 if stale else 1)+I*1e-10*dt
            if public:
                public.predict(u=a,B=B,F=F,Q=Q); x=public.x.copy();P=public.P.copy()
            else:
                x=F@x+B@a; P=F@P@F.T+Q
        last=t
        if kind==0: held=row
        if kind==1:
            z=np.array([row.east_m,row.north_m,row.up_m,row.ve_m_s,row.vn_m_s,row.vu_m_s])
            age=max(0.,t-float(row.t_s))
            if robust and method!='no_delay': z[:3]+=z[3:]*age
            innovation=z-x
            Spos=P[:2,:2]+R0[:2,:2]
            nis=float(innovation[:2]@np.linalg.solve(Spos,innovation[:2]))
            scale=float(np.clip(nis/cfg['nis_threshold'],1,cfg['adaptive_cap'])) if robust and method!='no_adaptive' else 1.
            R=R0.copy(); R[:3,:3]*=scale
            if robust and method!='no_delay': R[:3,:3]+=np.eye(3)*(age*cfg['delayed_velocity_sigma'])**2
            rejected=bool(robust and method!='no_gate' and nis>cfg['gate_threshold'])
            if rejected: R[:3,:3]=np.eye(3)*1e12
            before=x.copy()
            if method=='gps_only': x=z.copy()
            elif method=='fixed_gain':
                w=np.array([cfg['position_weight']]*3+[cfg['velocity_weight']]*3)
                x=(1-w)*x+w*z
            elif public:
                public.update(z,R=R);x=public.x.copy();P=public.P.copy()
            else:
                K=np.linalg.solve(P+R,P.T).T
                x=x+K@innovation
                P=(I-K)@P@(I-K).T+K@R@K.T
                P=(P+P.T)*.5
            if record:
                updates.append([t,float(row.t_s),nis,scale,before[0],z[0],x[0],before[1],z[1],x[1],int(rejected),age])
        if kind==2 and record: outputs.append([t,*x])
        step_latencies.append(time.perf_counter()-step_tick)
    elapsed=time.perf_counter()-tick
    details={**att_init.details,'cpu_s':time.process_time()-cpu_tick,
             'event_latency_p50_ms':float(np.quantile(step_latencies,.5)*1000),
             'event_latency_p95_ms':float(np.quantile(step_latencies,.95)*1000),
             'event_count':len(events)}
    return np.array(outputs),np.array(updates),elapsed,details


def score(traj,ref,segment,scenario,method):
    """Reference access is strictly outside estimate, interpolation is causal CV."""
    start,end=float(segment.start_s),float(segment.end_s)
    r=ref[(ref.t_s>=start)&(ref.t_s<=end)].copy()
    ix=np.searchsorted(traj[:,0],r.t_s.to_numpy(),side='right')-1
    valid=ix>=0; idx=np.maximum(ix,0)
    age=r.t_s.to_numpy()-traj[idx,0]
    valid &= age<=.100001
    est=traj[idx,1:4]+traj[idx,4:7]*age[:,None]
    delta=est-r[['east_m','north_m','up_m']].to_numpy()
    r['error_m']=np.linalg.norm(delta[:,:2],axis=1)
    r['error_3d_m']=np.linalg.norm(delta,axis=1)
    r['vertical_error_m']=np.abs(delta[:,2])
    r['scored']=valid
    r['method']=method;r['segment_id']=segment.segment_id;r['scenario_id']=scenario['scenario_id']
    r['family']=scenario['family'];r['strength']=scenario['strength'];r['seed']=scenario['seed']
    r['phase']='overall'
    is_dropout=scenario['family'] in ['dropout','combo_dropout_noise','combo_imu_noise_dropout','imu_dropout']
    if is_dropout:
        a,b=scenario['window']
        r.loc[(r.t_s>=a)&(r.t_s<b),'phase']='missing'
        r.loc[(r.t_s>=b)&(r.t_s<b+5),'phase']='recovery'
        r.loc[(r.t_s<a)|(r.t_s>=b+5),'phase']='normal'
    metrics=[]
    for phase,mask in [('overall',np.ones(len(r),bool)),('missing',r.phase=='missing'),('recovery',r.phase=='recovery')]:
        if phase!='overall' and not is_dropout: continue
        sub=r[mask];e=sub.loc[sub.scored,'error_m']
        metrics.append(dict(segment_id=segment.segment_id,scenario_id=scenario['scenario_id'],method=method,
                            family=scenario['family'],strength=scenario['strength'],seed=scenario['seed'],phase=phase,
                            n_ref=len(sub),n_scored=len(e),coverage=len(e)/len(sub) if len(sub) else 0,
                            rmse_m=float(np.sqrt(np.mean(e**2))) if len(e) else None,
                            p95_m=float(e.quantile(.95)) if len(e) else None,
                            median_m=float(e.median()) if len(e) else None,
                            rmse_3d_m=float(np.sqrt(np.mean(sub.loc[sub.scored,'error_3d_m']**2))) if len(e) else None,
                            vertical_rmse_m=float(np.sqrt(np.mean(sub.loc[sub.scored,'vertical_error_m']**2))) if len(e) else None,
                            status='ok' if len(e) else 'no_reference'))
    return r,metrics


def select_segments(imu,ref):
    valid=imu[imu.valid].copy()
    valid['block']=(valid.t_s.diff()>.35).cumsum()
    candidates=[]
    for _,b in valid.groupby('block'):
        start=float(np.ceil(b.t_s.iloc[0]+6))
        while start+60<=float(b.t_s.iloc[-1]):
            # Reject any invalid measurement event inside the whole initial/test range.
            window=imu[imu.t_s.between(start-5,start+60)]
            if window.valid.all() and len(ref[ref.t_s.between(start,start+60)])>=55:
                candidates.append((start,start+60))
            start+=90
    early=[v for v in candidates if v[0]<300]
    later=[v for v in candidates if v[0]>=400]
    if not early or len(later)<4: raise ValueError('Insufficient continuous data')
    chosen=[('tune',early[0])]+[('test',later[i]) for i in np.linspace(0,len(later)-1,4,dtype=int)]
    selected=[]
    for purpose,(a,b) in chosen:
        selected.append(dict(segment_id=f'{purpose}_{len(selected)+1:02}',start_s=a,end_s=b,purpose=purpose,
                             n_ref=int(ref.t_s.between(a,b).sum())))
    if not any(s['purpose']=='tune' for s in selected) or len(selected)<5:
        raise ValueError(f'Insufficient deterministic continuous segments: {selected}')
    return pd.DataFrame(selected)


def scenarios(segment):
    start=float(segment.start_s)
    rows=[dict(scenario_id='normal',family='normal',strength=0.,seed=0)]
    for duration in [1,3,5]:
        for offset in [20,40]:
            rows.append(dict(scenario_id=f'drop_{duration}_at_{offset}',family='dropout',strength=float(duration),
                             seed=offset,window=[start+offset,start+offset+duration]))
    for sigma in [1,3,5]:
        for seed in [11,22,33,44,55]:
            rows.append(dict(scenario_id=f'noise_{sigma}_seed_{seed}',family='noise',strength=float(sigma),seed=seed))
    for delay in [.2,.5,1.]:
        rows.append(dict(scenario_id=f'delay_{delay}',family='delay',strength=delay,seed=0))
    for magnitude in [10,20,50]:
        rows.append(dict(scenario_id=f'outlier_{magnitude}',family='outlier',strength=magnitude,seed=0))
    for duration in [1,3,5]:
        rows.append(dict(scenario_id=f'imu_drop_{duration}',family='imu_dropout',strength=duration,seed=0,
                         window=[start+20,start+20+duration]))
    for sigma in [.2,.5,1.]:
        for seed in [11,22,33,44,55]:
            rows.append(dict(scenario_id=f'imu_noise_{sigma}_seed_{seed}',family='imu_noise',strength=sigma,seed=seed))
    for seed in [11,22,33,44,55]:
        rows.extend([
            dict(scenario_id=f'combo_drop_noise_{seed}',family='combo_dropout_noise',strength=3.,seed=seed,
                 window=[start+20,start+25],gps_noise_m=3.),
            dict(scenario_id=f'combo_noise_delay_{seed}',family='combo_noise_delay',strength=3.,seed=seed,
                 gps_noise_m=3.,delay_s=.5),
            dict(scenario_id=f'combo_imu_drop_{seed}',family='combo_imu_noise_dropout',strength=.5,seed=seed,
                 window=[start+20,start+25],imu_noise_m_s2=.5)])
    rows.append(dict(scenario_id='combo_outlier_delay',family='combo_outlier_delay',strength=20.,seed=0,
                     outlier_m=20.,delay_s=.5))
    return rows


def main(args):
    out=ROOT/('results_demo' if args.demo else 'results');out.mkdir(exist_ok=True)
    begin=time.perf_counter()
    data=ROOT/('demo' if args.demo else 'data')
    gps=pd.read_csv(data/'gps.csv');imu=pd.read_csv(data/'imu.csv');ref=pd.read_csv(data/'reference.csv')
    load_s=time.perf_counter()-begin
    if args.demo:
        segment=pd.read_csv(ROOT/'demo/segment.csv').iloc[0]
        segments=pd.DataFrame([segment]);configs=json.loads((ROOT/'configs/locked.json').read_text())
    else:
        segments=select_segments(imu,ref);segments.to_csv(ROOT/'configs/segments.csv',index=False)
        # Explicit parameter search confined to one earlier tuning segment.
        tune=segments.query("purpose=='tune'").iloc[0]
        tune_scenes=[dict(scenario_id='tune_normal',family='normal',strength=0.,seed=0),
                     dict(scenario_id='tune_noise3',family='noise',strength=3.,seed=101)]
        trials=[]
        for weight in [.2,.5,.8]:
            for sigma in [1.,3.,5.]:
                cfg={**DEFAULT,'position_weight':weight,'velocity_weight':weight,'sigma_position':sigma}
                values=[]
                for sc in tune_scenes:
                    prepared=events_for(gps,imu,tune,sc,cfg)
                    for method in ['fixed_gain','adaptive']:
                        t,_,_,_=estimate(method,prepared,cfg)
                        _,m=score(t,ref,tune,sc,method);values.append(m[0]['rmse_m'])
                trials.append(dict(weight=weight,sigma_position=sigma,objective_rmse=float(np.mean(values))))
        pd.DataFrame(trials).to_csv(out/'tuning.csv',index=False)
        best=min(trials,key=lambda x:x['objective_rmse'])
        configs={**DEFAULT,'position_weight':best['weight'],'velocity_weight':best['weight'],'sigma_position':best['sigma_position']}
        dump(ROOT/'configs/locked.json',configs)
    all_metrics=[];all_errors=[];runs=[];timings=[];all_scenarios=[]
    equivalence=[]
    for segment in segments.query("purpose=='test'").itertuples():
        scene_list=scenarios(segment)
        if args.demo:
            scene_list=[s for s in scene_list if s['scenario_id'] in ['normal','drop_5_at_20','noise_5_seed_11','delay_1.0']]
        for sc in scene_list:
            prepared=events_for(gps,imu,segment,sc,configs)
            folder=out/segment.segment_id/sc['scenario_id'];folder.mkdir(parents=True,exist_ok=True)
            prepared[-1].to_csv(folder/'gps_delivered.csv',index=False)
            all_scenarios.append(dict(segment_id=segment.segment_id,**sc))
            trajectories=[];trace_by_method={}
            for method in METHODS:
                try:
                    traj,updates,elapsed,init=estimate(method,prepared,configs)
                    if not np.isfinite(traj).all(): raise ValueError('Nonfinite result')
                    e,m=score(traj,ref,segment,sc,method)
                    all_errors.append(e);all_metrics.extend(m)
                    df=pd.DataFrame(traj,columns=['t_s','east_m','north_m','up_m','ve_m_s','vn_m_s','vu_m_s'])
                    df['method']=method;df['segment_id']=segment.segment_id;df['scenario_id']=sc['scenario_id']
                    trajectories.append(df);trace_by_method[method]=traj
                    pd.DataFrame(updates,columns=['t_arrival_s','t_measure_s','nis','noise_scale','predicted_e','gps_e','updated_e','predicted_n','gps_n','updated_n','position_rejected','observation_age_s']).to_csv(folder/f'updates_{method}.csv',index=False)
                    runs.append(dict(segment_id=segment.segment_id,scenario_id=sc['scenario_id'],method=method,status='ok',rows=len(traj),algorithm_s=elapsed,initialization=init))
                except Exception as exc:
                    runs.append(dict(segment_id=segment.segment_id,scenario_id=sc['scenario_id'],method=method,status='failed',error=repr(exc)))
                    raise
            pd.concat(trajectories).to_csv(folder/'trajectories.csv',index=False)
            equivalence.append(dict(segment_id=segment.segment_id,scenario_id=sc['scenario_id'],
                                    max_abs_diff=float(np.max(np.abs(trace_by_method['kf_fixed']-trace_by_method['filterpy'])))))
            print(segment.segment_id,sc['scenario_id'],'done',flush=True)
        # First call warmup; 5 independent repetitions/method; file I/O excluded.
        normal=scenarios(segment)[0];prepared=events_for(gps,imu,segment,normal,configs)
        for method in METHODS:
            estimate(method,prepared,configs)
            for trial in range(5):
                _,_,elapsed,_=estimate(method,prepared,configs)
                timings.append(dict(segment_id=segment.segment_id,method=method,trial=trial+1,
                                    duration_s=segment.end_s-segment.start_s,algorithm_s=elapsed,
                                    input_load_s=load_s,realtime_factor=(segment.end_s-segment.start_s)/elapsed))
    pd.DataFrame(all_metrics).to_csv(out/'metrics.csv',index=False)
    pd.concat(all_errors).to_csv(out/'errors.csv',index=False)
    pd.DataFrame(timings).to_csv(out/'timing.csv',index=False)
    pd.DataFrame(equivalence).to_csv(out/'public_baseline_equivalence.csv',index=False)
    dump(out/'scenarios.json' if args.demo else ROOT/'configs/scenarios.json',all_scenarios)
    dump(out/'runs.json',runs)
    dump(out/'environment.json' if args.demo else ROOT/'run_logs/environment.json',dict(python=sys.version,platform=platform.platform(),
        cpu=platform.processor(),versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','pyproj','matplotlib','filterpy']},
        input_sha256={n:sha(data/n) for n in ['gps.csv','imu.csv','reference.csv']},
        code_sha256=sha(__file__),total_s=time.perf_counter()-begin,
        scope='raw IMU under explicitly adopted FRD/NED conventions; no use of OnboardPose or reference attitude',
        timing='per method: attitude, prediction, update and output construction; excludes loading, disturbance and scoring'))
    print('COMPLETE',len(runs),'method/segment/scenario runs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true');main(p.parse_args())
