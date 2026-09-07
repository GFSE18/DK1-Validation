"""Focused tests for torque budgets, physical import and installation screening."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.validation-deps'))
import numpy as np
import mujoco
from run_tests import Controller,Motion,MOTION_TIME_SCALE
import calibration
import installation_check

class Validation(unittest.TestCase):
    def model(self):
        m=mujoco.MjModel.from_xml_path(str(ROOT/'ts20_humanoid_v2.xml'))
        d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,1);mujoco.mj_forward(m,d)
        return m,d

    def test_budget_latches_without_invented_cooling(self):
        m,d=self.model();c=Controller(m,d,'peak-budget')
        self.assertTrue(np.all(c.limits()==c.peak))
        for _ in range(299):c.observe(c.peak)
        self.assertTrue(np.all(c.limits()==c.peak))
        c.observe(c.peak)
        self.assertTrue(np.all(c.limits()==c.rated))
        for _ in range(5000):c.observe(np.zeros(m.nu))
        self.assertTrue(np.all(c.limits()==c.rated))

    def test_mit_formula_and_output_limits(self):
        m,d=self.model();c=Controller(m,d,'rated')
        q=d.qpos[c.qids]+.02;v=d.qvel[c.dofs]+.03;ff=np.full(m.nu,.1)
        m.actuator_forcerange[:,0]=-c.rated;m.actuator_forcerange[:,1]=c.rated
        d.ctrl[:]=c.kp*q+c.kd*v+ff
        mujoco.mj_forward(m,d)
        expected=np.clip(c.kp*(q-d.qpos[c.qids])+c.kd*(v-d.qvel[c.dofs])+ff,-c.rated,c.rated)
        np.testing.assert_allclose(d.actuator_force,expected,atol=1e-12)
        d.ctrl[:]=100
        mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.actuator_force,c.rated)

    def test_datasheet_mapping(self):
        m,d=self.model()
        ratios=m.actuator_user[:,2]
        self.assertEqual(int((ratios==50).sum()),13)
        self.assertEqual(int((ratios==100).sum()),10)
        self.assertAlmostEqual(m.body_mass.sum(),4.11)
        self.assertEqual(m.nv,29)

    def test_motion_has_four_distinct_swings(self):
        m,d=self.model();p=Motion(m,d,'walk')
        for t,side in [(6.5*MOTION_TIME_SCALE,1),(14.5*MOTION_TIME_SCALE,0),
                       (22.5*MOTION_TIME_SCALE,1),(30.5*MOTION_TIME_SCALE,0)]:
            com,feet,support,phase=p.sample(t)
            self.assertFalse(support[side]);self.assertTrue(support[1-side])
            self.assertGreater(feet[side,2]-p.feet0[side,2],.019)

    def test_squat_target_is_deeper_than_previous_35mm_profile(self):
        m,d=self.model();p=Motion(m,d,'squat')
        standing=p.sample(2.0*MOTION_TIME_SCALE)[0][2]
        deepest=p.sample(6.0*MOTION_TIME_SCALE)[0][2]
        self.assertAlmostEqual(standing-deepest,.050,places=6)

    def test_reject_nonphysical_measured_inertia(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            path=Path(folder)/'bad.json'
            path.write_text(json.dumps({'measurement_source':'synthetic negative test, not real measurements',
                'bodies':{'pelvis':{'mass_kg':.3,'com_m':[0,0,0],'fullinertia_kg_m2':[1,1,5,0,0,0]}}}))
            with self.assertRaisesRegex(ValueError,'Nonphysical'):calibration.apply_physical(path)

    def test_blank_template_is_not_calibration(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            path=Path(folder)/'empty.json'
            path.write_text(json.dumps({'measurement_source':'test','bodies':{}}))
            with self.assertRaisesRegex(ValueError,'No measured'):calibration.apply_physical(path)

    def test_valid_measured_import_keeps_uncalibrated_status(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            temp=Path(folder)
            (temp/'ts20_humanoid_v2.xml').write_bytes((ROOT/'ts20_humanoid_v2.xml').read_bytes())
            path=temp/'measured.json'
            path.write_text(json.dumps({'measurement_source':'SYNTHETIC TEST ONLY',
                'bodies':{'pelvis':{'mass_kg':.4,'com_m':[0,0,.01],'fullinertia_kg_m2':[.0004,.0003,.0002,0,0,0]}}}))
            with patch.object(calibration,'ROOT',temp):calibration.apply_physical(path)
            m=mujoco.MjModel.from_xml_path(str(temp/'ts20_humanoid_measured.xml'))
            self.assertAlmostEqual(m.body_mass.sum(),4.16)
            status=json.loads((temp/'ts20_humanoid_measured.calibration.json').read_text())
            self.assertFalse(status['hardware_calibrated'])
            self.assertTrue(status['remaining_unmeasured_bodies'])

    def test_bench_fit_recovers_known_synthetic_rig(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            temp=Path(folder);path=temp/'synthetic.csv'
            t=np.arange(0,12,.005)
            v=.8*np.sin(2*t)+.35*np.sin(5*t)
            a=np.gradient(v,t);angle=-.4*np.cos(2*t)-.07*np.cos(5*t)
            load=.1*np.cos(angle)
            torque=.012*a+.025*v+.04*np.sign(v)+load
            np.savetxt(path,np.c_[t,angle,v,torque,load],delimiter=',',header='time_s,angle_rad,speed_rad_s,torque_Nm,external_load_Nm',comments='')
            with patch.object(calibration,'ROOT',temp):fit=calibration.fit_bench(path)
            self.assertAlmostEqual(fit['total_output_inertia_kg_m2'],.012,places=8)
            self.assertAlmostEqual(fit['viscous_Nm_per_rad_s'],.025,places=8)
            self.assertAlmostEqual(fit['coulomb_Nm'],.04,places=8)
            self.assertFalse(fit['hardware_calibrated'])

    def test_nominal_coincident_motor_conflicts_detected(self):
        report=installation_check.run()
        self.assertFalse(report['mechanical_installation_validated'])
        self.assertEqual(len(report['assumed_placements']),23)
        self.assertTrue(any({r['motor_a'],r['motor_b']}=={'left_hip_yaw','left_hip_roll'} for r in report['intersections']))

if __name__=='__main__':unittest.main()
