import mujoco

from neutele.core.integrate import TeleCore


class TeleoperationEnv:
    def __init__(self, model_path: str):
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._tele_core = TeleCore()

    def control(self, model: mujoco.MjModel, data: mujoco.MjData):
        pos = self._tele_core.act()[:5]
        self.mj_data.qpos = pos
        mujoco.mj_inverse(self.mj_model, self.mj_data)
        self.mj_data.ctrl = self.mj_data.qfrc_inverse

    def run(self):
        mujoco.set_mjcb_control(self.control)
        mujoco.viewer.launch(self.mj_model, self.mj_data)

    def close(self):
        self._tele_core.close()


if __name__ == "__main__":
    agent = TeleoperationEnv("G:\\NEU_Tele\\neutele\\station\\flying_hand\\urdf\\leader\\leader.xml")
    try:
        agent.run()
    except Exception as e:
        print(e)
    finally:
        agent.close()
