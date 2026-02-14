from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from acetele.equipment.joystick.joystick_driver import JDKFPVDriver
from acetele.simulation.px4_interface import PX4Interface
from acetele.station.base_station import make_station


class MujocoBase:
    # PX4 Quad X Standard: 1:FR(CCW), 2:BL(CCW), 3:FL(CW), 4:BR(CW)
    # MuJoCo XML: Rotor 1(FR), Rotor 2(BL), Rotor 3(FL), Rotor 4(BR)
    ROTOR_DIRECTION = np.array([1.0, 1.0, -1.0, -1.0])

    # Motor Constants
    MOTOR_CONSTANT = 8.54858e-06
    MOMENT_CONSTANT = 0.016
    ROTOR_DRAG_COEFF = 8.06428e-05
    ROLLING_MOMENT_COEFF = 1e-06

    # GPS Origin: Beihang University, Beijing
    GPS_LAT_START = 39.98329
    GPS_LON_START = 116.34745
    GPS_ALT_START = 50.0

    def __init__(self, model_path: str):
        """Initialize MuJoCo model and PX4 interface."""
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Set timestep to 0.001s (1000Hz)
        self.mj_model.opt.timestep = 0.001

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.station = make_station()
        self.joystick = JDKFPVDriver()

        # PX4 interface (TCP server at 4560)
        print("-" * 50)
        print("[PX4 SITL] Waiting for connection on TCP 4560...")
        print("[PX4 SITL] Run command: export PX4_SIM_MODEL=none_iris && make px4_sitl none")
        print("-" * 50)
        self.px4 = PX4Interface()

        self.sim_time_us = 0
        self.step_count = 0

        self.desired_rotor_velocity = np.zeros(4)
        self.rotor_velocity = np.zeros(4)
        self.rotor_angle = np.zeros(4)
        self.rotor_offsets = self._load_rotor_offsets()

        self._init_hardware_mapping()
        self._init_sensors()

        # Set initial keyframe and arm actuator ctrl
        self._reset_to_home()

    def _init_hardware_mapping(self):
        """Cache IDs for actuators, bodies, mocap, and sensors."""
        # Base link
        self.base_link_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # Arm actuators
        self.arm_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_gripper_left",
            "joint_gripper_right",
        ]
        self.arm_actuator_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.arm_joint_names
        ]

        rotor_candidates = ["rotor_1_vis", "rotor_2_vis", "rotor_3_vis", "rotor_4_vis"]
        alt_candidates = ["rotor_1", "rotor_2", "rotor_3", "rotor_4"]
        body_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in rotor_candidates]
        if any(b < 0 for b in body_ids):
            body_ids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in alt_candidates]
            self.rotor_body_names = alt_candidates
        else:
            self.rotor_body_names = rotor_candidates
        self.rotor_body_ids = body_ids
        self.rotor_mocap_ids = [self.mj_model.body_mocapid[b_id] if b_id >= 0 else -1 for b_id in self.rotor_body_ids]

        # Sensor IDs
        self.sensor_ids = {
            "pos": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "framepos"),
            "quat": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "framequat"),
            "linvel": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "framelinvel"),
            "gyro": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro"),
            "accel": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer"),
            "mag": mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "magnetometer"),
        }

    def _init_sensors(self):
        """Initialize sensor buffers with initial noisy values."""
        # Zero-order hold buffers
        self.last_accel_frd = np.zeros(3)
        self.last_gyro_frd = np.zeros(3)
        self.last_mag_frd = np.zeros(3)
        self.last_baro_alt = self.GPS_ALT_START

        # Initial noise sampling
        self.last_mag_frd = self._get_mag_with_noise()
        self.last_accel_frd = self._get_accel_with_noise()
        self.last_gyro_frd = self._get_gyro_with_noise()

    def _reset_to_home(self):
        key_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            print("Loading 'home' keyframe...")
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, key_id)
            # Initialize arm actuator control values from qpos
            home_pose = []
            for i, act_id in enumerate(self.arm_actuator_ids):
                j_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, self.arm_joint_names[i])
                if j_id >= 0 and act_id >= 0:
                    self.mj_data.ctrl[act_id] = self.mj_data.qpos[self.mj_model.jnt_qposadr[j_id]]
                    home_pose.append(self.mj_data.ctrl[act_id])
            print(f"Home Pose: [{', '.join([f'{v:.3f}' for v in home_pose])}]")
        else:
            print("No 'home' keyframe found. Using default.")

    def _load_rotor_offsets(self):
        arr = []
        for i in range(1, 5):
            site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, f"rotor_offset_{i}")
            if site_id < 0:
                return None
            arr.append(self.mj_model.site_pos[site_id].copy())
        return np.array(arr)

    def _update_arm_control_from_station(self):
        if self.step_count % 5 == 0:
            joint_pos, _, _ = self.station.act()
            for i, act_id in enumerate(self.arm_actuator_ids):
                if act_id >= 0 and i < len(joint_pos):
                    self.mj_data.ctrl[act_id] = joint_pos[i]

    def _get_sensor_raw(self, name: str):
        id = self.sensor_ids.get(name, -1)
        if id == -1:
            return np.zeros(3)
        adr = self.mj_model.sensor_adr[id]
        return self.mj_data.sensordata[adr : adr + self.mj_model.sensor_dim[id]].copy()

    def _get_accel_with_noise(self):
        val = self._get_sensor_raw("accel")
        val += np.random.normal(0, [0.00637, 0.00637, 0.00686])
        return np.array([val[0], -val[1], -val[2]])  # Body to FRD

    def _get_gyro_with_noise(self):
        val = self._get_sensor_raw("gyro")
        val += np.random.normal(0, 0.0008726646, size=3)
        return np.array([val[0], -val[1], -val[2]])  # Body to FRD

    def _get_mag_with_noise(self):
        val = self._get_sensor_raw("mag") * 10000.0  # Tesla -> Gauss
        val += np.random.normal(0, 0.003, size=3)
        return np.array([val[0], -val[1], -val[2]])  # Body to FRD

    def _update_sensors_and_send(self):
        """Update IMU/mag/baro buffers at target rates and send HIL_SENSOR."""
        # IMU (Accel/Gyro): 250Hz -> Every 2 steps (4ms)
        if self.step_count % 4 == 0:
            self.last_accel_frd = self._get_accel_with_noise()
            self.last_gyro_frd = self._get_gyro_with_noise()

        # Mag: 100Hz -> Every 5 steps (10ms)
        if self.step_count % 10 == 0:
            self.last_mag_frd = self._get_mag_with_noise()

        # Baro: 50Hz -> Every 10 steps (20ms)
        if self.step_count % 20 == 0:
            pos = self._get_sensor_raw("pos")
            self.last_baro_alt = pos[2] + self.GPS_ALT_START + np.random.normal(0, 0.25)

        # Send HIL_SENSOR at 250Hz
        if self.step_count % 4 == 0:
            self.px4.send_hil_sensor(
                self.sim_time_us, self.last_accel_frd, self.last_gyro_frd, self.last_mag_frd, self.last_baro_alt
            )

    def _update_gps_and_send(self):
        """Update GPS data and send HIL_GPS (50Hz)."""
        if self.step_count % 20 == 0:
            lat_e7, lon_e7, alt_mm = self._get_gps_pos_with_noise()
            vel_cm_s, vn_cm_s, ve_cm_s, vd_cm_s, cog_cdeg = self._get_gps_vel_with_noise()
            self.px4.send_hil_gps(
                self.sim_time_us, lat_e7, lon_e7, alt_mm, vel_cm_s, vn_cm_s, ve_cm_s, vd_cm_s, cog_cdeg
            )

    def _update_px4_controls(self):
        """Update PX4 controls (200Hz)."""
        if self.step_count % 4 == 0:
            controls = self.px4.receive_controls()
            if controls and len(controls) >= 4:
                # PX4: [FR, BL, FL, BR] -> MuJoCo: [FR, BL, FL, BR] (Direct map)
                self.desired_rotor_velocity = np.clip(np.array(controls[:4]) * 1000, 0, 1000)

    def _get_gps_pos_with_noise(self):
        """Return (lat_e7, lon_e7, alt_mm) with small position noise."""
        pos = self._get_sensor_raw("pos")
        pos_noisy = pos + np.random.normal(0, 0.01, size=3)
        lat = self.GPS_LAT_START + (pos_noisy[0] / 111319.9)
        lon = self.GPS_LON_START - (pos_noisy[1] / (111319.9 * np.cos(np.radians(self.GPS_LAT_START))))
        gps_alt = self.GPS_ALT_START + pos_noisy[2]
        return int(lat * 1e7), int(lon * 1e7), int(gps_alt * 1000)

    def _get_gps_vel_with_noise(self):
        """Return (vel, vn, ve, vd, cog) in cm/s and centidegrees with small noise."""
        vel_w = self._get_sensor_raw("linvel")
        vel_w = vel_w + np.random.normal(0, 0.1, size=3)
        vn = vel_w[0] * 100.0
        ve = -vel_w[1] * 100.0
        vd = -vel_w[2] * 100.0
        vel = float(np.linalg.norm([vn, ve, vd]))
        cog_rad = np.arctan2(ve, vn)
        cog_deg = (np.degrees(cog_rad) + 360.0) % 360.0
        return int(vel), int(vn), int(ve), int(vd), int(cog_deg * 100.0)

    def _update_rc_channels(self):
        """Update RC channels from joystick input (10Hz)."""
        if self.step_count % 10 == 0:
            data = self.joystick.act()
            if data and data["connected"] and data["mapped"]:
                mapped = data["mapped"]
                channels = np.array([-1.0] * 8)
                channels[0] = mapped.get("Roll", 0.0)
                channels[1] = mapped.get("Pitch", 0.0)
                channels[2] = mapped.get("Throttle", -1.0)
                channels[3] = mapped.get("Yaw", 0.0)
                channels[4] = mapped.get("Aux1", -1.0)
                channels[5] = mapped.get("Aux2", -1.0)
                channels[6] = mapped.get("Aux3", -1.0)
                channels[7] = mapped.get("Aux4", -1.0)
                condlist = [channels > 0.5, (channels >= -0.5) & (channels <= 0.5), channels < -0.5]
                choicelist = [2000, 1500, 1000]
                channels = np.select(condlist, choicelist)
                self.px4.send_rc_channels_to_qgc(*channels)

    # def _apply_motor_physics(self):
    #     """Apply rotor thrust, drag and reaction moments to base_link."""
    #     dt = self.mj_model.opt.timestep
    #     for i in range(4):
    #         diff = self.desired_rotor_velocity[i] - self.rotor_velocity[i]
    #         tc = 0.0125 if diff > 0 else 0.025
    #         self.rotor_velocity[i] += diff * (1.0 - np.exp(-dt / tc))

    #     base_quat = self._get_sensor_raw("quat")
    #     Rb = Rotation.from_quat(base_quat, scalar_first=True)
    #     v_com_w = self._get_sensor_raw("linvel")
    #     omega_r = self._get_sensor_raw("gyro")
    #     omega_w = Rb.apply(omega_r)

    #     f_sum_w = np.zeros(3)
    #     m_sum_w = np.zeros(3)

    #     for i in range(4):
    #         r_off_w = Rb.apply(self.rotor_offsets[i])
    #         v_point_w = v_com_w + np.cross(omega_w, r_off_w)

    #         v_point_r = Rb.inv().apply(v_point_w)
    #         v_planar_r = np.array([v_point_r[0], v_point_r[1], 0.0])

    #         omega = self.rotor_velocity[i]
    #         direction = self.ROTOR_DIRECTION[i]
    #         thrust = self.MOTOR_CONSTANT * (omega**2)
    #         torque_z_r = self.MOMENT_CONSTANT * thrust * (-direction)

    #         f_drag_r = -self.ROTOR_DRAG_COEFF * omega * v_planar_r
    #         m_rolling_r = -self.ROLLING_MOMENT_COEFF * omega * v_planar_r

    #         f_total_w = Rb.apply(np.array([0.0, 0.0, thrust]) + f_drag_r)
    #         m_react_w = Rb.apply(np.array([0.0, 0.0, torque_z_r]) + m_rolling_r)

    #         f_sum_w += f_total_w
    #         m_sum_w += np.cross(r_off_w, f_total_w) + m_react_w

    #     self.mj_data.xfrc_applied[self.base_link_id][:3] = f_sum_w
    #     self.mj_data.xfrc_applied[self.base_link_id][3:6] = m_sum_w

    def _apply_motor_physics(self):
        """Apply rotor thrust, drag and reaction moments to base_link."""
        dt = self.mj_model.opt.timestep
        for i in range(4):
            diff = self.desired_rotor_velocity[i] - self.rotor_velocity[i]
            tc = 0.0125 if diff > 0 else 0.025
            self.rotor_velocity[i] += diff * (1.0 - np.exp(-dt / tc))

        base_pos = self._get_sensor_raw("pos")
        base_quat = self._get_sensor_raw("quat")
        Rb = Rotation.from_quat(base_quat, scalar_first=True)
        v_com_w = self._get_sensor_raw("linvel")
        omega_r = self._get_sensor_raw("gyro")
        omega_w = Rb.apply(omega_r)

        self.mj_data.xfrc_applied[self.base_link_id][:] = 0.0
        self.mj_data.qfrc_applied[:] = 0.0

        for i in range(4):
            r_off_w = Rb.apply(self.rotor_offsets[i])
            v_point_w = v_com_w + np.cross(omega_w, r_off_w)

            v_point_r = Rb.inv().apply(v_point_w)
            v_planar_r = np.array([v_point_r[0], v_point_r[1], 0.0])

            omega = self.rotor_velocity[i]
            direction = self.ROTOR_DIRECTION[i]
            thrust = self.MOTOR_CONSTANT * (omega**2)
            torque_z_r = self.MOMENT_CONSTANT * thrust * (-direction)

            f_drag_r = -self.ROTOR_DRAG_COEFF * omega * v_planar_r
            m_rolling_r = -self.ROLLING_MOMENT_COEFF * omega * v_planar_r

            f_total_w = Rb.apply(np.array([0.0, 0.0, thrust]) + f_drag_r)
            m_react_w = Rb.apply(np.array([0.0, 0.0, torque_z_r]) + m_rolling_r)

            pos_w = base_pos + r_off_w
            mujoco.mj_applyFT(
                self.mj_model, self.mj_data, f_total_w, m_react_w, pos_w, self.base_link_id, self.mj_data.qfrc_applied
            )

    def _update_rotor_visuals(self):
        """Update rotor mocap visuals based on current spin and base pose."""
        base_pos = self._get_sensor_raw("pos")
        base_quat = self._get_sensor_raw("quat")
        Rb = Rotation.from_quat(base_quat, scalar_first=True)
        for i in range(4):
            mocap_id = self.rotor_mocap_ids[i]
            if mocap_id < 0:
                continue
            self.rotor_angle[i] += self.rotor_velocity[i] * self.ROTOR_DIRECTION[i] * self.mj_model.opt.timestep
            spin = Rotation.from_rotvec([0.0, 0.0, self.rotor_angle[i]])
            q_total = (Rb * spin).as_quat(scalar_first=True)
            pos_w = base_pos + Rb.apply(self.rotor_offsets[i])
            self.mj_data.mocap_pos[mocap_id] = pos_w
            self.mj_data.mocap_quat[mocap_id] = q_total

    def control(self, model, data):
        """MuJoCo control callback: exchange PX4 messages and advance simulation."""
        self.step_count += 1
        self.sim_time_us += int(model.opt.timestep * 1e6)

        if not self.px4.connected:
            self.px4.check_connection()
        else:
            self._update_sensors_and_send()
            self._update_gps_and_send()
            self._update_px4_controls()
            self._update_rc_channels()
            self._apply_motor_physics()

        self._update_arm_control_from_station()
        self._update_rotor_visuals()

    def run(self):
        mujoco.set_mjcb_control(self.control)
        mujoco.viewer.launch(self.mj_model, self.mj_data)

    def close(self):
        self.station.close()
        self.joystick.close()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_NAME = "x500_arm2x"
    MODEL_PATH = (BASE_DIR / "description" / MODEL_NAME / f"{MODEL_NAME}.xml").resolve()
    agent = MujocoBase(str(MODEL_PATH))
    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        agent.close()
