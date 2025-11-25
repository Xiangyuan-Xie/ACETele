import time

import mujoco
import mujoco.viewer
import numpy as np

from neutele.core.integrate import TeleCore


class MujocoBase:
    def __init__(self, model_path: str):
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)

        self._tele_core = TeleCore()

        pos = self._tele_core.act()
        self._data.qpos[:5] = np.append(pos, 0.0)
        self._data.qvel[:5] = 0.0
        self._data.qacc[:5] = 0.0

    def control(self):
        pos = self._tele_core.act()
        self._data.ctrl[:5] = np.append(pos, 0.0)

    def run(self):
        frame_count = 0
        last_time = time.perf_counter()
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                self.control()
                mujoco.mj_step(self._model, self._data)
                viewer.sync()
                self._tele_core.apply_torque_feedback(self._data.qfrc_constraint[:5])
                frame_count += 1
                now = time.perf_counter()
                if now - last_time >= 5.0:
                    fps = frame_count / (now - last_time)
                    print(f"[Debug] 当前帧率: {fps:.1f} Hz")
                    frame_count = 0
                    last_time = now

    def close(self):
        self._tele_core.close()


if __name__ == "__main__":
    agent = MujocoBase("/station/flying_hand/urdf/follower/flying_hand_follower.xml")
    try:
        agent.run()
    except Exception as e:
        print(e)
    finally:
        agent.close()
