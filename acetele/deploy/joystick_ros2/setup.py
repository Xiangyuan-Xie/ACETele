from setuptools import find_packages, setup

package_name = "joystick_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
    ],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "manual_control = joystick_ros2.manual_control_node:main",
        ],
    },
)
