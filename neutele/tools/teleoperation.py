import mujoco
import mujoco.viewer

from neutele.core.integrate import TeleCore


class MujocoBase:
    def __init__(self, model_path: str):
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_resetData(self._model, self._data)

        self._tele_core = TeleCore()

    def control(self):
        self._data.qpos[:4] = self._tele_core.act()
        self._data.qpos[-1] = 0.0
        self._data.qvel[:] = 0.0

    def run(self):
        with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                self.control()
                mujoco.mj_step(self._model, self._data)
                viewer.sync()

    def close(self):
        self._tele_core.close()


if __name__ == "__main__":
    agent = MujocoBase("G:\\NEU_Tele\\neutele\\station\\flying_hand\\urdf\\flying_hand_leader.xml")
    try:
        agent.run()
    except Exception as e:
        print(e)
    finally:
        agent.close()
