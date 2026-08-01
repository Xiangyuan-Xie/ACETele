from setuptools import find_packages, setup

package_name = "ace_robot_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/ace_robot.launch.py"]),
        (
            "share/" + package_name + "/config",
            [
                "config/ace_robot_params.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "ace_robot_node = ace_robot_ros2.ace_robot_node:main",
        ],
    },
)
