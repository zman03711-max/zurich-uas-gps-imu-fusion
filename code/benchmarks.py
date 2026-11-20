"""Fresh-process, single-thread measurements; no file I/O in timed estimate()."""
import argparse,json,os,subprocess,sys,tracemalloc,resource
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT,METHODS,events_for,estimate


def worker(method):
    cfg=json.loads((ROOT/'configs/locked.json').read_text())
    gps=pd.read_csv(ROOT/'data/gps.csv');imu=pd.read_csv(ROOT/'data/imu.csv')
    seg=pd.read_csv(ROOT/'configs/segments.csv').query("purpose=='test'").iloc[0]
    sc=dict(family='normal',strength=0.,seed=0,scenario_id='normal')
    prepared=events_for(gps,imu,seg,sc,cfg)
    estimate(method,prepared,cfg)
    rows=[]
    for repeat in range(5):
        _,_,elapsed,details=estimate(method,prepared,cfg)
        rows.append(dict(method=method,repeat=repeat+1,wall_s=elapsed,cpu_s=details['cpu_s'],
                         event_latency_p50_ms=details['event_latency_p50_ms'],
                         event_latency_p95_ms=details['event_latency_p95_ms'],
                         realtime_factor=(seg.end_s-seg.start_s)/elapsed))
    process_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    tracemalloc.start();estimate(method,prepared,cfg)
    _,tracked_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    result=dict(method=method,segment_id=seg.segment_id,measurements=rows,
                process_peak_rss_mib=process_peak,
                traced_allocation_peak_mib=tracked_peak/1024**2,
                memory_scope='fresh process includes interpreter, imports and loaded logs; allocation peak only the additional estimate call')
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=METHODS);args=p.parse_args()
    if args.worker: worker(args.worker);return
    results=[]
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for method in METHODS:
        run=subprocess.run([sys.executable,__file__,'--worker',method],capture_output=True,text=True,env=env,check=True)
        results.append(json.loads(run.stdout));print(method,'benchmark complete',flush=True)
    (ROOT/'results/benchmarks.json').write_text(json.dumps(results,indent=2))
    rows=[]
    for r in results:
        for m in r['measurements']:
            rows.append({**m,'process_peak_rss_mib':r['process_peak_rss_mib'],
                         'traced_allocation_peak_mib':r['traced_allocation_peak_mib']})
    pd.DataFrame(rows).to_csv(ROOT/'results/benchmarks.csv',index=False)


if __name__=='__main__':main()
