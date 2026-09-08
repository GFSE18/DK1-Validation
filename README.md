# TS20 500 mm humanoid — initial MuJoCo model

**新版已加入动作控制与自动报告。请先阅读 [使用说明.md](使用说明.md)，双击 `Start_Tests.cmd` 启动。下面保留原版 XML 的说明；原版文件没有被覆盖。**

Open `ts20_humanoid_500mm.xml` in MuJoCo Viewer. It is standalone and needs no CAD or mesh files. With Python and MuJoCo installed, run `python -m mujoco.viewer` and drag the XML into the viewer.

Source requirements: https://chatgpt.com/share/6a9ee31b-a8f4-83e9-8a66-1067223b4da8

## Geometry

All distances in the XML are metres; box `size` values are HALF dimensions. X is forward, Y left, Z up. Angles and viewer control targets are radians.

| Dimension | Value |
|---|---:|
| Ground to ankle | 40 mm |
| Ankle to knee | 135 mm |
| Knee to hip | 120 mm |
| Hip to waist | 35 mm |
| Waist to shoulder | 70 mm |
| Shoulder to neck | 15 mm |
| Neck to head top | 85 mm |
| Standing height | 500 mm |
| Hip spacing | 80 mm |
| Shoulder spacing | 145 mm |
| Upper arm / forearm / hand | 80 / 80 / 40 mm |
| Foot length / width / thickness | 100 / 55 / 10 mm |

23 powered joints: six per leg, four per arm, waist yaw, neck yaw and pitch. The base floats freely. The two neck axes are a modeling assumption. Joint centers within each hip, shoulder, and ankle are idealized as coincident; this model does not establish that real motors fit.

## Mass assumptions

Total mass is approximately 4.438 kg with the current 0.6 kg torso and six HTDW-5036DNE replacements. All masses include allocated motor mass; do not add another 35 g motor to every link. MuJoCo estimates center of mass and inertia from the primitive shapes and masses.

| Assembly | Mass |
|---|---:|
| Pelvis | 350 g |
| Torso, including waist motor and electronics allowance | 600 g |
| Each leg | 565 g |
| Each arm | 220 g |
| Neck and head | 190 g |

These are first-pass estimates, not measurements from `TS20.SLDPRT`. The existing CAD file is not required or modified. Replace masses and mass distributions with measured/CAD values before selecting motors.

## Motors and controls

Hip yaw, arms, waist, and neck use the chat's TS20 1:50 values: 0.7 Nm rated, 2 Nm peak, 31.4 rad/s reference maximum speed. Other leg joints use 1:100: 1.5 Nm rated, 4 Nm peak, 15.7 rad/s reference maximum speed. The source datasheet was not accessible in the shared conversation, so these are transcribed assumptions.

Position actuators enforce output peak torque limits. `gear="1"` is intentional: torque specifications are already at the gearbox output. Actuator `user` fields store rated torque, reference speed, and reduction ratio. Those fields are metadata, not active limits. This is not a thermal or torque-speed motor model. Servo gains, damping, joint limits, and rotor armature are preliminary assumptions.

The six currently load-critical lower-limb joints (`left/right_hip_roll`, `left/right_knee`, and `left/right_ankle_pitch`) use the supplied HTDW-5036-02-DNE module: 323 g, 36:1, 6 Nm rated torque, 21 Nm locked-rotor torque, and 75 RPM no-load speed. The catalogue does not state a transient peak duration, so the model uses 6 Nm as the motion torque cap and does not treat 21 Nm as a safe peak limit. The module envelope is approximately 50 × 47.4 mm and still requires real mechanical packaging validation.

Load the `stand` keyframe to reset upright. `shallow_squat` is a second initial pose, not an animated squat routine. For a symmetric squat, both hip pitch and ankle pitch targets are negative; both knees are positive. For example: -0.25, +0.50, -0.25 rad. This positive knee convention differs from the earlier illustrative negative knee range in the chat.

Angle, angular-speed, and actuator-torque sensors are included for each joint, plus sole contact sensors. With unit actuator gearing, actuator force readings correspond to output torque in Nm. Mechanical power can be calculated as torque times angular velocity. No automated torque report or motion controller is included.

The floating robot has no active balance controller and can fall when targets change. Self-collision is disabled to permit an initial idealized skeleton; floor contact remains enabled for all parts. Successful loading or standing is not evidence that the motors can walk, squat continuously, or support a real robot.

MJCF attribute reference: https://mujoco.readthedocs.io/en/stable/XMLreference.html

`build_model.py` regenerates the XML if you prefer editing the construction parameters. Direct XML edits are overwritten if you run the generator again.

Current project tuning points: edit `TORSO_MASS_KG` in `build_model.py` to change the generated torso mass, and edit `MOTION_TIME_SCALE` in `run_tests.py` to change commanded motion speed (`0.5` is twice the original pace). `upgrade_model.py` preserves the configured torso mass when rebuilding v2.
