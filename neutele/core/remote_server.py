import importlib
from typing import Optional, Sequence

import msgpack_numpy as m
import zerorpc
from config.config_loader import ConfigLoader
from station.base_station import BaseStation

m.patch()


class TeleCore:
    def __init__(self, config_path: Optional[str] = None):
        self._config_loader = ConfigLoader(config_path)
        module_name, class_name = self._config_loader.get_station_info()
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self._station: BaseStation = cls(self._config_loader)

    def act(self) -> Sequence[float]:
        return self._station.act()


def run_server(host: str = "0.0.0.0", port: int = 4242):
    s = zerorpc.Server(TeleCore())
    s.bind(f"tcp://{host}:{port}")
    print(f"ZeroRPC server running on tcp://{host}:{port}")
    s.run()


if __name__ == "__main__":
    run_server()
