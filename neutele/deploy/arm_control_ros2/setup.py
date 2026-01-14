from setuptools import find_packages, setup

package_name = "arm_control_ros2"

setup(
    name=package_name,
    version="0.0.1",
    # packages=[package_name],
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/arm_control.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "arm_control = arm_control_ros2.arm_control:main",
        ],
    },
)
