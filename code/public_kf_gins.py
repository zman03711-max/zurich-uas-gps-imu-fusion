"""Run unmodified KF-GINS: official 60 s demo and a declared AGZ subset."""
import json,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer,Geod
from scipy.spatial.transform import Rotation
from experiment import ROOT,events_for,scenarios,score

SOURCE=ROOT/'third_party/KF-GINS-source'


def binary():
    found=[p for p in SOURCE.rglob('KF-GINS') if p.is_file()]
    if len(found)!=1: raise ValueError(f'Public binary not unique: {found}')
    return found[0]


def run_config(config,folder):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
    begin=time.perf_counter()
    p=subprocess.run([str(binary()),str(path)],cwd=SOURCE,capture_output=True,text=True,errors='replace',timeout=60)
    elapsed=time.perf_counter()-begin
    (folder/'terminal.log').write_text(p.stdout+'\n'+p.stderr)
    if p.returncode: raise RuntimeError(f'KF-GINS exit {p.returncode}: {folder}')
    nav=np.loadtxt(folder/'KF_GINS_Navresult.nav',ndmin=2)
    if len(nav)<10 or not np.isfinite(nav).all(): raise ValueError('Invalid public trajectory')
    return nav,elapsed


def official_demo():
    data=ROOT/'third_party/official_demo';out=ROOT/'results/public_demo'
    cfg=yaml.safe_load((data/'kf-gins.original.yaml').read_text(encoding='utf-8',errors='replace'))
    cfg.update(imupath=str(data/'Leador-A15.txt'),gnsspath=str(data/'GNSS-RTK.txt'),
               outputpath=str(out),starttime=456300,endtime=456360)
    nav,elapsed=run_config(cfg,out)
    truth=np.loadtxt(data/'truth.nav')
    lat=np.interp(nav[:,1],truth[:,1],truth[:,2]);lon=np.interp(nav[:,1],truth[:,1],truth[:,3])
    alt=np.interp(nav[:,1],truth[:,1],truth[:,4])
    _,_,distance=Geod(ellps='WGS84').inv(nav[:,3],nav[:,2],lon,lat)
    result=dict(scope='official data subset 456300-456360 s; functional reproduction, not all paper tables',
                rows=len(nav),horizontal_rmse_m=float(np.sqrt(np.mean(distance**2))),
                vertical_rmse_m=float(np.sqrt(np.mean((nav[:,4]-alt)**2))),
                cpp_end_to_end_s=elapsed)
    (out/'validation.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(dict(t_s=nav[:,1],horizontal_error_m=distance,vertical_error_m=nav[:,4]-alt)).to_csv(out/'errors.csv',index=False)
    print('OFFICIAL_DEMO',json.dumps(result),flush=True)


def agz_config(prepared,segment,folder,noise,metadata,cfg):
    init,att,held,events,incoming=prepared
    origin=np.array(metadata['origin_utm32n_m'])
    inverse=Transformer.from_crs(32632,4326,always_xy=True)
    lon,lat=inverse.transform(init[0]+origin[0],init[1]+origin[1])
    # First and last padding increments are outside the processed interval.
    # Within the interval each increment uses only the preceding logged sample.
    rows=[];previous=float(segment.start_s)
    rows.append([previous,*np.array([held.gx,held.gy,held.gz])*.1,*np.array([held.ax,held.ay,held.az])*.1])
    points=[(t,row) for t,kind,row in events if kind==0]
    points.append((float(segment.end_s),None))
    points.append((float(segment.end_s)+.1,None))
    for t,row in points:
        if t>previous+1e-10:
            dt=t-previous
            rows.append([t,*np.array([held.gx,held.gy,held.gz])*dt,*np.array([held.ax,held.ay,held.az])*dt])
            previous=t
        if row is not None: held=row
    folder.mkdir(parents=True,exist_ok=True)
    np.savetxt(folder/'imu.txt',np.array(rows),fmt='%.12f')
    glon,glat=inverse.transform(incoming.east_m+origin[0],incoming.north_m+origin[1])
    gnss=np.c_[incoming.t_s,glat,glon,incoming.up_m,np.full((len(incoming),3),cfg['sigma_position'])]
    # Guard record avoids dropping the last real observation at EOF; not consumed.
    guard=gnss[-1].copy();guard[0]=float(segment.end_s)+2
    np.savetxt(folder/'gnss.txt',np.vstack([gnss,guard]),fmt='%.12f')
    euler=Rotation.from_quat(att.details['initial_quaternion_xyzw']).as_euler('xyz',degrees=True)
    configuration=dict(imupath=str(folder/'imu.txt'),gnsspath=str(folder/'gnss.txt'),outputpath=str(folder),
        imudatalen=7,imudatarate=10,starttime=float(segment.start_s),endtime=float(segment.end_s),
        initpos=[float(lat),float(lon),float(init[2])],initvel=[float(init[4]),float(init[3]),float(-init[5])],
        initatt=euler.tolist(),initgyrbias=(att.bias*180/np.pi*3600).tolist(),initaccbias=[0.]*3,
        initgyrscale=[0.]*3,initaccscale=[0.]*3,initposstd=[3.]*3,initvelstd=[1.]*3,
        initattstd=[10.,10.,90.],imunoise=dict(arw=[noise['arw']]*3,vrw=[noise['vrw']]*3,
        gbstd=[1000.]*3,abstd=[5000.]*3,gsstd=[300.]*3,asstd=[300.]*3,corrtime=1.),antlever=[0.]*3)
    return configuration


def public_case(gps,imu,ref,segment,scene,folder,noise,metadata,cfg):
    prepared=events_for(gps,imu,segment,scene,cfg)
    config=agz_config(prepared,segment,folder,noise,metadata,cfg)
    nav,elapsed=run_config(config,folder)
    transform=Transformer.from_crs(4326,32632,always_xy=True)
    east,north=transform.transform(nav[:,3],nav[:,2]);origin=metadata['origin_utm32n_m']
    native=np.c_[nav[:,1],east-origin[0],north-origin[1],nav[:,4],nav[:,6],nav[:,5],-nav[:,7]]
    native=np.vstack([np.r_[segment.start_s,prepared[0]],native])
    native=native[np.argsort(native[:,0],kind='stable')]
    grid=np.round(np.r_[np.arange(segment.start_s,segment.end_s,.1),segment.end_s],9)
    ix=np.searchsorted(native[:,0],grid,side='right')-1
    age=grid-native[ix,0]
    if age.max()>.350001: raise ValueError('Public output has an unexpected long gap')
    states=native[ix,1:].copy();states[:,:3]+=states[:,3:]*age[:,None]
    trajectory=np.c_[grid,states]
    e,m=score(trajectory,ref,segment,scene,'kf_gins')
    frame=pd.DataFrame(trajectory,columns=['t_s','east_m','north_m','up_m','ve_m_s','vn_m_s','vu_m_s'])
    frame['method']='kf_gins';frame['segment_id']=segment.segment_id;frame['scenario_id']=scene['scenario_id']
    frame.to_csv(folder/'trajectories.csv',index=False)
    return e,m,dict(segment_id=segment.segment_id,scenario_id=scene['scenario_id'],status='ok',
                    cpp_end_to_end_s=elapsed,rows=len(nav))


def main():
    official_demo()
    gps=pd.read_csv(ROOT/'data/gps.csv');imu=pd.read_csv(ROOT/'data/imu.csv');ref=pd.read_csv(ROOT/'data/reference.csv')
    segments=pd.read_csv(ROOT/'configs/segments.csv');cfg=json.loads((ROOT/'configs/locked.json').read_text())
    metadata=json.loads((ROOT/'data/metadata.json').read_text())
    tune=segments.query("purpose=='tune'").iloc[0]
    trials=[]
    for arw in [10.,30.,100.]:
        for vrw in [10.,30.]:
            noise=dict(arw=arw,vrw=vrw)
            scene=dict(family='normal',strength=0.,seed=0,scenario_id='normal')
            folder=ROOT/'results/public_tuning'/f'arw_{arw}_vrw_{vrw}'
            _,m,_=public_case(gps,imu,ref,tune,scene,folder,noise,metadata,cfg)
            trials.append({**noise,'objective_rmse_m':m[0]['rmse_m']})
    pd.DataFrame(trials).to_csv(ROOT/'results/public_tuning.csv',index=False)
    noise=min(trials,key=lambda x:x['objective_rmse_m'])
    (ROOT/'configs/public_locked.json').write_text(json.dumps(noise,indent=2))
    errors=[];metrics=[];records=[]
    for segment in segments.query("purpose=='test'").itertuples():
        for scene in scenarios(segment):
            if scene['family'] not in ['normal','noise','outlier','dropout']: continue
            folder=ROOT/'results/public_baseline'/segment.segment_id/scene['scenario_id']
            e,m,record=public_case(gps,imu,ref,segment,scene,folder,noise,metadata,cfg)
            errors.append(e);metrics.extend(m);records.append(record)
            print(segment.segment_id,scene['scenario_id'],'public complete',flush=True)
    pd.DataFrame(metrics).to_csv(ROOT/'results/public_metrics.csv',index=False)
    pd.concat(errors).to_csv(ROOT/'results/public_errors.csv',index=False)
    (ROOT/'results/public_runs.json').write_text(json.dumps(records,indent=2))
    print('PUBLIC_COMPLETE',len(records),flush=True)


if __name__=='__main__': main()
