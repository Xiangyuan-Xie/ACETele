from setuptools import find_packages, setup

package_name = "data_collector_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/data_collector.launch.py"]),
        ("share/" + package_name + "/config", ["config/data_collector_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "data_collector_node = data_collector_ros2.data_collector_node:main",
        ],
    },
)
