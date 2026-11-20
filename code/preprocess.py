"""Wu Yan: audited Zurich MAV preprocessing and causal GPS/IMU baselines.

Run from the project root. Reference data is never loaded by run().
"""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial.transform import Rotation

FILES = ['OnboardGPS.csv', 'RawAccel.csv', 'RawGyro.csv',
         'GroundTruthAGL.csv', 'OnboardPose.csv', 'BarometricPressure.csv',
         'GroundTruthAGM.csv', 'StreetViewGPS.csv']


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None
        self.names = self.zip.namelist() if self.zip else [str(p) for p in self.path.rglob('*') if p.is_file()]

    def read(self, name, optional=False):
        aliases = {name.lower()}
        if name == 'OnboardGPS.csv':
            aliases.add('onbordgps.csv')
        matches = [n for n in self.names if PurePosixPath(n.replace('\\', '/')).name.lower() in aliases]
        if len(matches) != 1:
            if optional and not matches:
                return None
            raise ValueError(f'{name}: expected one member, found {matches}')
        return self.zip.read(matches[0]) if self.zip else Path(matches[0]).read_bytes()

    def close(self):
        if self.zip:
            self.zip.close()


def parse_csv(raw):
    """Allow trailing empty placeholders, reject malformed nonempty rows."""
    rows = csv.reader(io.StringIO(raw.decode('utf-8-sig')))
    header = [v.strip() for v in next(rows)]
    while header and not header[-1]:
        header.pop()
    if not all(header) or len(set(header)) != len(header):
        raise ValueError('Empty or duplicate field names')
    records, line_ids = [], []
    for line, row in enumerate(rows, 2):
        row = [v.strip() for v in row]
        if not any(row):
            continue
        while len(row) > len(header) and not row[-1]:
            row.pop()
        if len(row) != len(header):
            raise ValueError(f'CSV line {line}: {len(row)} fields, expected {len(header)}')
        records.append(row)
        line_ids.append(line)
    df = pd.DataFrame(records, columns=header).apply(pd.to_numeric, errors='raise')
    df['_source_line'] = line_ids
    for c in ['Timpstemp', 'Timestemp', 'Timestamp', 'timestamp']:
        if c in df:
            if df[c].isna().any() or (df[c] % 1 != 0).any():
                raise ValueError('Timestamp must be a nonempty integer in microseconds')
            df = df.rename(columns={c: 't_us'})
            df['t_us'] = df.t_us.astype('int64')
    return df


def ordered(df):
    return df.sort_values(['t_us', '_source_line'], kind='stable').drop_duplicates('t_us', keep='last').copy()


def audit_tables(tables, out):
    audits, anomalies, dictionary = [], [], []
    for name, df in tables.items():
        row = dict(file=name, rows=len(df), columns=len(df.columns)-1,
                   missing_cells=int(df.isna().sum().sum()),
                   nonfinite_cells=int((~np.isfinite(df.to_numpy(dtype=float))).sum()))
        for c in df.columns:
            if c != '_source_line':
                dictionary.append(dict(file=name, field=c, dtype=str(df[c].dtype),
                                       min=df[c].min(), max=df[c].max()))
        if 't_us' in df:
            delta = df.t_us.diff()/1e6
            positive = delta[delta > 0]
            median = float(positive.median())
            row.update(t_min_us=int(df.t_us.min()), t_max_us=int(df.t_us.max()),
                       duration_s=float((df.t_us.max()-df.t_us.min())/1e6),
                       duplicate_timestamps=int(df.t_us.duplicated().sum()),
                       backwards=int((delta < 0).sum()), median_dt_s=median,
                       p95_dt_s=float(positive.quantile(.95)),
                       max_gap_s=float(positive.max()))
            for i in df.index[(delta <= 0) | (delta > max(.5, 5*median))]:
                if i == 0:
                    continue
                anomalies.append(dict(file=name, source_line=int(df.loc[i, '_source_line']),
                                      before_us=int(df.loc[i-1, 't_us']), after_us=int(df.loc[i, 't_us']),
                                      dt_s=float(delta.loc[i])))
        audits.append(row)
    pd.DataFrame(audits).to_csv(out/'data_audit.csv', index=False)
    pd.DataFrame(anomalies, columns=['file','source_line','before_us','after_us','dt_s']).to_csv(out/'time_anomalies.csv', index=False)
    pd.DataFrame(dictionary).to_csv(out/'field_ranges.csv', index=False)


def preprocess(args):
    cfg = load_json(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = Source(args.source)
    tables, hashes = {}, {}
    try:
        for name in FILES:
            raw = src.read(name, optional=name not in FILES[:4])
            if raw is not None:
                hashes[name] = hashlib.sha256(raw).hexdigest()
                tables[name] = parse_csv(raw)
        aerial = [n for n in src.names if PurePosixPath(n.replace('\\','/')).parent.name == 'MAV Images'
                  and n.lower().endswith(('.jpg','.jpeg'))]
    finally:
        src.close()
    audit_tables(tables, out)
    rawgps = tables['OnboardGPS.csv']
    # Map reference timestamps BEFORE sorting, thinning, quality rejection, or disturbances.
    key_counts = rawgps.groupby('imgid').t_us.nunique()
    ambiguous = key_counts[key_counts > 1].index
    keymap = rawgps[~rawgps.imgid.isin(ambiguous)].drop_duplicates('imgid')[['imgid','t_us']]
    gt = tables['GroundTruthAGL.csv']
    if gt.imgid.duplicated().any():
        raise ValueError('Duplicate ground-truth image IDs; resolve before scoring')
    ref = gt.merge(keymap, on='imgid', how='left', validate='one_to_one')
    ref[ref.t_us.isna()].to_csv(out/'reference_unmatched.csv', index=False)
    ref = ref.dropna(subset=['t_us']).copy()
    ref['t_us'] = ref.t_us.astype('int64')
    gps = ordered(rawgps)
    scale = 1e-7 if float(gps.lat.abs().median()) > 90 else 1.0
    gps['lat_deg'], gps['lon_deg'] = gps.lat*scale, gps.lon*scale
    valid = (gps.fix_type >= 3) & gps.lat_deg.between(-90,90) & gps.lon_deg.between(-180,180)
    valid &= np.isfinite(gps[['lat_deg','lon_deg','vel_e_m_s','vel_n_m_s']]).all(axis=1)
    gps[~valid].to_csv(out/'gps_rejected.csv', index=False)
    gps = gps[valid].copy()
    if gps.empty:
        raise ValueError('No valid GPS fix')
    tr = Transformer.from_crs(4326,32632,always_xy=True)
    gps['east_abs'], gps['north_abs'] = tr.transform(gps.lon_deg.to_numpy(),gps.lat_deg.to_numpy())
    t0 = int(min(tables[n].t_us.min() for n in FILES[:3]))
    origin = [float(gps.east_abs.iloc[0]),float(gps.north_abs.iloc[0])]
    gps['east_m'],gps['north_m'] = gps.east_abs-origin[0],gps.north_abs-origin[1]
    gps['ve_m_s'],gps['vn_m_s'] = gps.vel_e_m_s,gps.vel_n_m_s
    # Diagnostic projection residual on the unthinned GPS stream.
    check = gt.merge(gps[['imgid','east_abs','north_abs']],on='imgid',validate='one_to_one')
    proj_err = np.hypot(check.east_abs-check.x_gps,check.north_abs-check.y_gps)
    check['projection_residual_m'] = proj_err
    check[['imgid','projection_residual_m']].to_csv(out/'projection_check.csv',index=False)
    gps['t_s'] = (gps.t_us-t0)/1e6
    # Actual duplicate receiver solutions cannot be inferred from repeated coordinates alone.
    measurement = ['lat','lon','alt','vel_n_m_s','vel_e_m_s','vel_d_m_s','fix_type','num_sat']
    repeated = int((gps[measurement] == gps[measurement].shift()).all(axis=1).sum())
    keep, last = [], -np.inf
    for i, t in enumerate(gps.t_s):
        if t-last >= cfg['gps_period_s']-1e-9:
            keep.append(i); last=t
    gps = gps.iloc[keep].copy()
    gps['t_arrival_s'] = gps.t_s
    # The archive already stores latitude/longitude in degrees and altitude in m.
    # Keep the supplied altitude datum; do not silently label it ellipsoidal height.
    gps['up_m'] = gps.alt
    gps['vu_m_s'] = -gps.vel_d_m_s
    gps[['t_us','t_s','t_arrival_s','imgid','east_m','north_m','up_m','ve_m_s','vn_m_s','vu_m_s','lat_deg','lon_deg','fix_type','eph_m','epv_m','num_sat','_source_line']].to_csv(out/'gps.csv',index=False)
    accel, gyro = ordered(tables['RawAccel.csv']), ordered(tables['RawGyro.csv'])
    gyro = gyro.rename(columns={'t_us':'gyro_t_us','_source_line':'gyro_source_line',**{c:'g'+c for c in ['x','y','z']}})
    imu = pd.merge_asof(accel,gyro[['gyro_t_us','gx','gy','gz','gyro_source_line']],
                        left_on='t_us',right_on='gyro_t_us', direction='backward',
                        tolerance=int(cfg['imu_match_tolerance_s']*1e6))
    imu['gyro_age_s'] = (imu.t_us-imu.gyro_t_us)/1e6
    imu['valid'] = np.isfinite(imu[['x','y','z','gx','gy','gz']]).all(axis=1)
    if 'Error_count' in accel:
        imu['accel_error_count'] = imu.Error_count
    imu['t_s'] = (imu.t_us-t0)/1e6
    imu = imu.rename(columns={'x':'ax','y':'ay','z':'az'})
    imu[['t_us','t_s','ax','ay','az','gx','gy','gz','gyro_t_us','gyro_age_s','valid','_source_line','gyro_source_line']].to_csv(out/'imu.csv',index=False)
    ref['t_s'] = (ref.t_us-t0)/1e6
    ref['east_m'], ref['north_m'] = ref.x_gt-origin[0], ref.y_gt-origin[1]
    ref['up_m'] = ref.z_gt
    ref[['imgid','t_us','t_s','east_m','north_m','up_m','_source_line']].sort_values('t_s').to_csv(out/'reference.csv',index=False)
    if 'OnboardPose.csv' in tables:
        pose = ordered(tables['OnboardPose.csv'])
        pose['t_s']=(pose.t_us-t0)/1e6
        pose[['t_s','Attitude_x','Attitude_y','Attitude_z','Attitude_w']].to_csv(out/'initial_attitude_candidates.csv',index=False)
    norms=np.linalg.norm(imu.loc[imu.valid,['ax','ay','az']].to_numpy(),axis=1)
    metadata = dict(source=str(Path(args.source).resolve()), archive_image_count=len(aerial),
                    archive_check='image inventory only; CRC checked for consumed members, not image payloads',
                    sha256_logs=hashes,t0_us=t0,origin_utm32n_m=origin,crs='EPSG:32632',
                    lat_lon_scale=scale,reference_total=len(gt),reference_matched=len(ref),
                    ambiguous_imgids=[int(v) for v in ambiguous],
                    projection_median_m=float(proj_err.median()), projection_max_m=float(proj_err.max()),
                    consecutive_repeated_gps_solutions=repeated,gps_output_rows=len(gps),
                    imu_valid_rows=int(imu.valid.sum()),imu_total_rows=len(imu),
                    accel_norm_median_raw=float(np.median(norms)),
                    conventions_verified=cfg['conventions_verified'],config=cfg)
    save_json(out/'metadata.json',metadata)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description="Audited AGZ numerical-log preprocessing")
    p.add_argument('command',choices=['preprocess'])
    p.add_argument('--source',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--config',default='configs/preprocess.json')
    args=p.parse_args();preprocess(args)

if __name__=='__main__': main()
