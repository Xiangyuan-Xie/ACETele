from setuptools import find_packages, setup

package_name = "leader_arm_controller_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/leader_arm_controller.launch.py"]),
        ("share/" + package_name + "/config", ["config/leader_arm_controller_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "leader_arm_controller = leader_arm_controller_ros2.leader_arm_controller:main",
        ],
    },
)
