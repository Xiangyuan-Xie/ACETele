import time

import msgpack_numpy as m
import zerorpc

m.patch()


class TeleCore(zerorpc.Client):
    def __init__(self, host: str = "127.0.0.1", port: int = 4242):
        super().__init__(self)
        self.connect(f"tcp://{host}:{port}")


if __name__ == "__main__":
    tele_core = TeleCore()
    while True:
        print(tele_core.act())
        time.sleep(0.1)
