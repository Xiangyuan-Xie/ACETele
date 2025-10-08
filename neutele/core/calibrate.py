import numpy as np

from neutele.config.config_loader import ConfigLoader
from neutele.equipment.feetech.feetech_driver import FeeTechDriver


class Calibration:
    def __init__(self):
        self._config_loader = ConfigLoader()
        self._config = self._config_loader.config

    def calibrate(self):
        result = False
        for linker_name, linker_config in self._config["linker"].items():
            ids = linker_config["joint_ids"]
            driver = FeeTechDriver(ids, linker_config["port"])
            home_poses = (np.array(linker_config["home_poses"]) * 2048 / np.pi).astype(int)
            result = driver.calibrate(ids, home_poses)
            if result:
                pos, _ = driver.get_pos_and_vel()
                pos = np.array(list(pos.values()))
                print(f"{linker_name}臂标定完成，当前姿态：{pos}！")
            else:
                print(f"{linker_name}臂标定失败！")
            driver.close()
        return result


if __name__ == "__main__":
    calibration = Calibration()
    calibration.calibrate()
