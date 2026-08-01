# ace-robot-zmq PX4 XRCE Stack

This superbuild installs an isolated native publication stack for the ZMQ
follower. It does not inspect or overwrite `/usr/local`, where a different Agent
version may be installed.

Initialize submodules, then build the pinned stack:

```bash
git submodule update --init --recursive
cmake -S zeromq/ace_robot_zmq/xrce -B build/ace_robot_zmq-xrce \
  -DACETELE_XRCE_PREFIX="$HOME/.local/lib/acetele/xrce-2.4.2"
cmake --build build/ace_robot_zmq-xrce --parallel
```

The prefix contains:

- `MicroXRCEAgent` 2.4.2;
- Micro XRCE-DDS Client 2.4.0;
- `ace-px4-xrce-publisher` and its schema/version manifest.

The ZMQ follower validates all three versions before it opens its robot serial
ports. Set `ACETELE_XRCE_PREFIX` or pass `--xrce-prefix` when using another
isolated installation location.
