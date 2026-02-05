import numpy as np

from acetele.config.config_loader import ConfigLoader
from acetele.equipment.feetech.feetech_driver import FeeTechDriver


class Calibration:
    def __init__(self):
        self._config_loader = ConfigLoader()
        self._config = self._config_loader.get_linker_config()

    def calibrate(self):
        result = False
        for i, linker_config in enumerate(self._config):
            ids = linker_config["joint_ids"]
            driver = FeeTechDriver(ids, linker_config["port"])
            home_poses = (np.array(linker_config["home_poses"]) * 2048.0 / np.pi).astype(int)
            result = driver.calibrate(ids, home_poses)
            if result:
                pos, _, _ = driver.get_state()
                pos = np.array(list(pos.values()))
                print(f"臂{i}标定完成，当前姿态：{pos}.")
            else:
                print(f"臂{i}标定失败！")
            driver.close()
        return result


if __name__ == "__main__":
    calibration = Calibration()
    calibration.calibrate()
