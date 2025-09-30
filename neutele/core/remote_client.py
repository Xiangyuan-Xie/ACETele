import time

import msgpack_numpy as m
import zerorpc

m.patch()

tele_core = zerorpc.Client()
tele_core.connect("tcp://127.0.0.1:4242")

while True:
    print(tele_core.act())
    time.sleep(0.1)
