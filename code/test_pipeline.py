import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from experiment import Attitude, DEFAULT, METHODS, disturbed, events_for, estimate, score
from preprocess import parse_csv


def fixture():
    times=np.arange(-5.,20.001,.1)
    imu=pd.DataFrame(dict(t_s=times,ax=0.,ay=0.,az=-9.80665,gx=0.,gy=0.,gz=0.,valid=True))
    t=np.arange(-5.,21.)
    gps=pd.DataFrame(dict(t_s=t,t_arrival_s=t,east_m=t,north_m=0.,up_m=100.,ve_m_s=1.,vn_m_s=0.,vu_m_s=0.))
    ref=pd.DataFrame(dict(t_s=np.arange(11.),imgid=np.arange(11),east_m=np.arange(11.),north_m=0.,up_m=100.))
    return gps,imu,ref,SimpleNamespace(start_s=0.,end_s=10.,segment_id='fixture')


class Pipeline(unittest.TestCase):
    def setUp(self): self.g,self.i,self.r,self.s=fixture();self.cfg=DEFAULT.copy()
    def scene(self,**kw): return dict(family='normal',strength=0,seed=0,scenario_id='normal',**kw)
    def test_gravity_three_axes(self):
        a=Attitude(self.i.head(50),self.g.head(5),self.cfg)
        np.testing.assert_allclose(a.step(self.i.iloc[0],.1),[0,0,0],atol=1e-10)
    def test_constant_velocity_all_methods(self):
        prepared=events_for(self.g,self.i,self.s,self.scene(),self.cfg)
        for method in METHODS:
            t,*_=estimate(method,prepared,self.cfg)
            np.testing.assert_allclose(t[:,1],t[:,0],atol=1e-8)
            np.testing.assert_allclose(t[:,2:4],np.tile([0.,100.],(len(t),1)),atol=1e-8)
    def test_future_cannot_change_past(self):
        p=events_for(self.g,self.i,self.s,self.scene(),self.cfg)
        a,*_=estimate('adaptive',p,self.cfg)
        changed=self.i.copy();changed.loc[changed.t_s>=5,'ax']=2.
        b,*_=estimate('adaptive',events_for(self.g,changed,self.s,self.scene(),self.cfg),self.cfg)
        np.testing.assert_allclose(a[a[:,0]<5],b[b[:,0]<5],atol=1e-10)
    def test_dropout_masks_both_gps_components(self):
        sc=dict(family='dropout',strength=3,seed=0,scenario_id='drop',window=[3,6])
        _,updates,*_=estimate('gps_only',events_for(self.g,self.i,self.s,sc,self.cfg),self.cfg)
        self.assertFalse(((updates[:,1]>=3)&(updates[:,1]<6)).any())
        self.assertTrue((updates[:,1]==6).any())
    def test_latency_compensation_uses_velocity_not_future(self):
        s=SimpleNamespace(start_s=0.,end_s=20.,segment_id='delay')
        sc=dict(family='delay',strength=1,seed=0,scenario_id='delay')
        t,*_=estimate('adaptive',events_for(self.g,self.i,s,sc,self.cfg),self.cfg)
        np.testing.assert_allclose(t[:,1],t[:,0],atol=1e-8)
    def test_position_spike_rejected_velocity_retained(self):
        self.g.loc[self.g.t_s==5,'east_m']+=100.
        _,u,*_=estimate('adaptive',events_for(self.g,self.i,self.s,self.scene(),self.cfg),self.cfg)
        spike=u[u[:,1]==5][0]
        self.assertEqual(spike[10],1)
        self.assertLess(abs(spike[6]-5),1.)
    def test_imu_missing_graceful_fallback(self):
        sc=dict(family='imu_dropout',strength=3,seed=0,scenario_id='imu_drop',window=[3,6])
        for method in METHODS:
            t,*_=estimate(method,events_for(self.g,self.i,self.s,sc,self.cfg),self.cfg)
            self.assertTrue(np.isfinite(t).all())
    def test_filterpy_independent_equivalence(self):
        self.g.east_m+=np.sin(self.g.t_s)
        p=events_for(self.g,self.i,self.s,self.scene(),self.cfg)
        a,*_=estimate('kf_fixed',p,self.cfg);b,*_=estimate('filterpy',p,self.cfg)
        np.testing.assert_allclose(a,b,atol=1e-10)
    def test_known_horizontal_and_3d_error(self):
        t=np.c_[np.arange(11),np.arange(11)+3,np.ones(11)*4,np.ones(11)*112,np.ones(11),np.zeros((11,2))]
        _,m=score(t,self.r,self.s,self.scene(),'gps_only')
        self.assertAlmostEqual(m[0]['rmse_m'],5.)
        self.assertAlmostEqual(m[0]['rmse_3d_m'],13.)
    def test_bad_csv_and_large_integer_time(self):
        df=parse_csv(b'Timpstemp,x,,\n1000000000000001,2,,\n')
        self.assertEqual(df.t_us.iloc[0],1000000000000001)
        with self.assertRaises(ValueError): parse_csv(b'Timpstemp,x\n1,2,3\n')


if __name__=='__main__': unittest.main(verbosity=2)
