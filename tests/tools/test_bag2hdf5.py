import importlib
import sys
import types


def test_bag2hdf5_uses_ace_arm_topics_without_legacy_arm_namespace(monkeypatch):
    h5py_module = types.ModuleType("h5py")
    rclpy_module = types.ModuleType("rclpy")
    rclpy_serialization_module = types.ModuleType("rclpy.serialization")
    rclpy_serialization_module.deserialize_message = lambda *_args, **_kwargs: None
    rosidl_module = types.ModuleType("rosidl_runtime_py")
    rosidl_utilities_module = types.ModuleType("rosidl_runtime_py.utilities")
    rosidl_utilities_module.get_message = lambda *_args, **_kwargs: None

    monkeypatch.setitem(sys.modules, "h5py", h5py_module)
    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.serialization", rclpy_serialization_module)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", rosidl_module)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.utilities", rosidl_utilities_module)

    bag2hdf5 = importlib.import_module("acetele.tools.bag2hdf5")
    topics = set(bag2hdf5.TOPIC_CONFIG)

    assert "/ace_leader/arm/command" in topics
    assert "/ace_leader/gripper/command" in topics
    assert "/ace_follower/arm/state" in topics
    assert "/ace_follower/gripper/state" in topics
    assert all(not topic.startswith("/arm/") for topic in topics)

    leader_gripper_outputs = bag2hdf5.TOPIC_CONFIG["/ace_leader/gripper/command"]["outputs"]
    follower_gripper_outputs = bag2hdf5.TOPIC_CONFIG["/ace_follower/gripper/state"]["outputs"]
    assert [name for name, _extractor in leader_gripper_outputs] == ["action/gripper"]
    assert [name for name, _extractor in follower_gripper_outputs] == [
        "observation/gripper_position",
        "observation/gripper_velocity",
        "observation/gripper_effort",
    ]
