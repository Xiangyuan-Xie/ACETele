import shutil
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.build import build as _build

ROOT = Path(__file__).parent


class CleanBuild(_build):
    def run(self) -> None:
        build_lib = Path(self.build_lib)
        if build_lib.exists():
            shutil.rmtree(build_lib)
        super().run()


def read_readme() -> str:
    readme = ROOT / "README.md"
    if not readme.exists():
        return ""
    return readme.read_text(encoding="utf-8")


setup(
    name="acetele",
    version="0.2.0",
    description="A real-time remote teleoperation system for controlling robotic platforms.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Xiangyuan Xie",
    author_email="dragonboat_xxy@163.com",
    license="Apache-2.0",
    cmdclass={"build": CleanBuild},
    packages=find_namespace_packages(
        where=".",
        include=["acetele*"],
        exclude=[
            "acetele.deploy.px4_msgs*",
            "acetele.deploy.realsense-ros*",
            "build*",
            "tests*",
        ],
    ),
    include_package_data=False,
    package_data={
        "acetele.config": ["*.toml"],
        "acetele.robot.ace_follower": ["description/*.urdf", "description/meshes/*.STL"],
        "acetele.robot.ace_leader": ["description/*.urdf", "description/*.xml", "description/meshes/*.STL"],
    },
    python_requires=">=3.9",
    install_requires=[
        "h5py",
        "loguru",
        "numpy",
        "pin",
        "pygame",
        "pyserial",
        "tomli",
        "tqdm",
    ],
    keywords=["python", "robotic", "teleoperation"],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
