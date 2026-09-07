"""Create a torque-controlled v2 without overwriting the original prototype."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
TORSO_MASS_KG = 0.6

def build():
    tree = ET.parse(ROOT / 'ts20_humanoid_500mm.xml')
    root = tree.getroot()
    root.set('model', 'TS20 v2 - torque control; estimated mechanics')
    root.find('size').set('nuser_actuator', '4')
    for default in root.findall('./default/default'):
        p = default.find('position')
        peak = 2 if default.get('class') == 'ts20_50' else 4
        rated, speed, ratio = p.get('user').split()
        default.remove(p)
        ET.SubElement(default, 'general', gear='1', ctrllimited='false',
                      biastype='affine', biasprm='0 -12 -0.4', forcelimited='true',
                      forcerange=f'-{peak} {peak}', user=f'{rated} {speed} {ratio} 0.3')
    for actuator in root.findall('./actuator/position'):
        actuator.tag = 'general'
        actuator.attrib.pop('ctrlrange')
    torso = root.find(".//geom[@name='torso_shape']")
    if torso is None:
        raise ValueError('torso_shape not found in source model')
    torso.set('mass', str(TORSO_MASS_KG))
    for key in root.findall('./keyframe/key'):
        key.set('ctrl', ' '.join(['0'] * 23))
        if key.get('name') == 'shallow_squat':
            q = key.get('qpos').split()
            q[0] = str(-float(q[0]))  # Keep ankle centers at x=0 at initial crouch.
            key.set('qpos', ' '.join(q))
    # Transparent envelopes live only in a separate installation-check model.
    root.append(ET.Comment(' Run using run_tests.py. Standalone Viewer has torque sliders, no balance controller. Motor metadata: rated torque, no-load speed, reduction ratio, stated peak duration. No measured speed/thermal map. '))
    ET.indent(root, space='  ')
    tree.write(ROOT / 'ts20_humanoid_v2.xml', encoding='utf-8', xml_declaration=True)
    config = {
        'schema_version': 1,
        'source': 'KK-servo-datasheet-v0.1.pdf pages 3 and 5',
        'status': 'datasheet limits; uncalibrated dynamics and mass distribution',
        'voltage_V': 24,
        'variants': {
            '50': {'rated_Nm': .7, 'peak_Nm': 2, 'peak_duration_s': .3, 'no_load_rad_s': 31.4},
            '100': {'rated_Nm': 1.5, 'peak_Nm': 4, 'peak_duration_s': .3, 'no_load_rad_s': 15.7}},
        'mit': {'kp': 12.0, 'kd': .4, 'kp_max': 64, 'kd_max': 2},
        'unknown': ['loaded torque-speed envelope', 'efficiency', 'thermal recovery',
                    'rotor inertia', 'friction', 'control delay', 'real link inertias'],
        'peak_policy': 'Optional cumulative above-rated budget 0.3 s per run, then latch to rated. Conservative test policy, NOT manufacturer thermal logic.',
        'speed_policy': 'No-load speed is an exceedance reference, NOT an enforced loaded-speed envelope.',
    }
    (ROOT / 'motor_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Created ts20_humanoid_v2.xml and motor_config.json')

if __name__ == '__main__':
    build()
