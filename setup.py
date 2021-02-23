from distutils.command.bdist import bdist as bdist_command
import os
import setuptools
from setuptools.command.build_py import build_py as build_py_command
from setuptools.command.install import install as install_command
import shutil

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="prism-samplernn",
    version="0.1",
    author="RNCM PRiSM / Aalto SOPI",
    description="PRiSM implementation of SampleRNN: An Unconditional End-to-End Neural Audio Generation Model, for TensorFlow 2. (Packaged by SOPI)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SopiMlab/prism-samplernn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "tensorflow >= 2.2.0",
        "librosa >= 0.4.3",
        "natsort >= 7.1.1",
        "pydub >= 0.24.1",
        "keras-tuner >= 1.0.2"
    ],
    package_data={
        "samplernn_scripts": [
            "conf/*.config.json"
        ]
    },
    zip_safe=False
)
