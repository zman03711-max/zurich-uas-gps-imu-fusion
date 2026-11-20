"""Validate delivered results, create the real-data demo and reading note."""
import hashlib,importlib.metadata,inspect,json,os,platform,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from filterpy.kalman import KalmanFilter
from experiment import ROOT,METHODS


def main():
    results=ROOT/'results';(ROOT/'outputs').mkdir(exist_ok=True)
    basic=json.loads((results/'validation.json').read_text())
    public=pd.read_csv(results/'public_metrics.csv').query("phase=='overall'")
    fair=pd.read_csv(results/'position_only_metrics.csv').query("phase=='overall'")
    assert len(public)==100 and len(fair)==400
    assert public.coverage.min()==1. and fair.coverage.min()==1.
    for frame in [public,fair]: assert np.isfinite(frame[['rmse_m','rmse_3d_m']]).all().all()
    validation={**basic,'core_formal_runs':basic['runs'],'public_formal_runs':len(public),
                'position_only_control_runs':len(fair),'total_formal_runs':basic['runs']+len(public)+len(fair),
                'official_demo':json.loads((results/'public_demo/validation.json').read_text()),
                'position_only_field_invariance':json.loads((results/'position_only_validation.json').read_text())}
    (results/'final_validation.json').write_text(json.dumps(validation,indent=2))
    segments=pd.read_csv(ROOT/'configs/segments.csv');segment=segments.query("purpose=='test'").iloc[0]
    demo=ROOT/'demo';demo.mkdir(exist_ok=True)
    for name in ['gps.csv','imu.csv','reference.csv']:
        frame=pd.read_csv(ROOT/'data'/name)
        frame[frame.t_s.between(segment.start_s-5,segment.end_s)].to_csv(demo/name,index=False)
    pd.DataFrame([segment]).to_csv(demo/'segment.csv',index=False)
    dist=importlib.metadata.distribution('filterpy');vendor=ROOT/'third_party/filterpy';vendor.mkdir(exist_ok=True)
    source=Path(inspect.getsourcefile(KalmanFilter));(vendor/'kalman_filter.py').write_bytes(source.read_bytes())
    for path in dist.files or []:
        if 'license' in path.name.lower():
            (vendor/path.name).write_bytes(Path(dist.locate_file(path)).read_bytes())
    (vendor/'provenance.json').write_text(json.dumps(dict(package='filterpy',version=dist.version,
        source='https://github.com/rlabbe/filterpy',kalman_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2))
    hardware=dict(platform=platform.platform(),python=sys.version,cpu=platform.processor(),
        logical_cpus=psutil.cpu_count(),memory_bytes=psutil.virtual_memory().total,
        processor_details=subprocess.run(['lscpu'],capture_output=True,text=True).stdout,
        thread_limits={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    (ROOT/'run_logs/hardware.json').write_text(json.dumps(hardware,indent=2))
    own=pd.read_csv(results/'metrics.csv').query("phase=='overall'")
    table=pd.concat([own,public]).groupby(['family','method']).rmse_m.mean().unstack()
    normal=table.loc['normal'];outlier=table.loc['outlier']
    difficult=fair[(fair.family=='noise')&(fair.strength==5)&(fair.method=='adaptive')]
    note=f'''# 实验结果说明

已在杭州服务器完成 {validation['total_formal_runs']} 次正式实验：2232 次核心方法/消融运行，100 次原版 KF-GINS 对照，400 次仅位置更新控制实验。四个独立测试路段，每段 60 秒；随机扰动五种子。有效参考时刻覆盖率均为 100%。

正常条件的平均水平 RMSE：单 GPS **{normal['gps_only']:.3f}m**，固定融合 **{normal['fixed_gain']:.3f}m**，抗异常融合 **{normal['adaptive']:.3f}m**，原版 KF-GINS **{normal['kf_gins']:.3f}m**。主表中 Python 核心方法使用 GPS 位置和速度，而原版 KF-GINS 仅更新位置；严格公平对比应查看 `results/tables/position_only_fair_comparison.csv`。

在 10/20/50m 突发偏差条件的合并平均中，单 GPS **{outlier['gps_only']:.3f}m**，固定融合 **{outlier['fixed_gain']:.3f}m**，抗异常融合 **{outlier['adaptive']:.3f}m**。抗异常融合的优势主要体现在这类异常观测抑制；正常、GPS 中断或部分组合扰动中并不普遍优于单 GPS，已保留全部失败场景，不能写成“所有条件全面提升”。

公开基线使用原版 KF-GINS 的实际编译程序。官方数据 60 秒子集的水平 RMSE 为 **{validation['official_demo']['horizontal_rmse_m']:.6f}m**；这是官方示例子集的功能性复现，不是 AGZ 精度，也不是论文全表复现。AGZ 结果与官方示例分别存放。自实现固定 KF 与 FilterPy 的最大状态差为 **{basic['max_filterpy_difference']:.2e}**。

**重要失败案例：**在严格的仅位置更新协议下，5m 位置噪声时，自适应方法的平均水平 RMSE 达到 **{difficult.rmse_m.mean():.3f}m**，最差单次达到 **{difficult.rmse_m.max():.3f}m**。这说明该方法在缺少速度观测且位置噪声较大时可能明显失稳，不应推荐用于该条件。创新降权和门控可能在惯性漂移时持续削弱有效 GPS 修正；这是需要进一步验证的机制解释，不能作为已经独立证明的原因。这个失败应在报告的结果与局限部分重点说明。

完整性检查中的“失败运行为零”仅指没有执行或非有限数值错误；参考时刻覆盖率 100% 表示均有输出，不表示高精度。明显失稳的有限轨迹同样保留在结果中。

效率表提供同机独立进程的五次测量、每事件 P50/P95 延迟、进程内存及单独跟踪分配峰值。图表误差条表示跨路段/随机条件的标准差，不代表置信区间；表内 RMSE 和 P95 的均值也不等于把所有误差点混在一起重新计算。

原始 IMU 约 10Hz，安装外参、初始机头朝向和历史固件约定未经独立标定；轨迹真值是摄影测量相机位置并使用 GPS 初值。高度基准与安装杆臂未独立校准，三维/垂向结果作为诊断保留。延迟实验表示正确测量时标下的迟到观测，未宣称解决未知时钟偏移。四段测试不能代表全航线无缝导航。

先运行 `python workflow.py --demo` 验证小样本；完整 Linux 复现入口是 `python workflow.py`。四类加分项的文件对应、算法定义和全部假设见项目 README。
'''
    (ROOT/'outputs/实验结果说明.md').write_text(note,encoding='utf-8')
    manifest={}
    for folder in ['code','configs','data','demo','scripts']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                manifest[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['workflow.py','README.md','requirements.txt']:
        manifest[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (ROOT/'run_logs/delivered_input_code_sha256.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
