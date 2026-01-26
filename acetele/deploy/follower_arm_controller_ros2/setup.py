from setuptools import find_packages, setup

package_name = "follower_arm_controller_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/follower_arm_controller.launch.py"]),
        ("share/" + package_name + "/config", ["config/follower_arm_controller_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "follower_arm_controller = follower_arm_controller_ros2.follower_arm_controller:main",
        ],
    },
)
