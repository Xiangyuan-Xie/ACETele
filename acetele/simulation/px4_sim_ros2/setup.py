from setuptools import find_packages, setup

package_name = "px4_sim_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/px4_sim_config.yaml"]),
    ],
    install_requires=[
        "setuptools",
        "PyYAML",
    ],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "manual_control = px4_sim_ros2.manual_control_node:main",
        ],
    },
)
