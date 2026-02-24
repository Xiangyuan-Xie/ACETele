from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="acetele",
    version="0.1.0",
    description="A real-time remote teleoperation system for controlling robotic platforms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Xiangyuan Xie",
    author_email="dragonboat_xxy@163.com",
    python_requires=">=3.9",
    packages=find_packages(where="."),
    install_requires=[
        "h5py",
        "loguru",
        "numpy",
        "pin",
        "pygame",
        "pyserial",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="python robotic teleoperation",
)
