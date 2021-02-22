import os
import setuptools
import shutil

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

conf_dir = "./samplernn/conf"
conf_files = [fn for fn in os.listdir(".") if fn.endswith(".config.json")]

try:
    os.mkdir(conf_dir)
    for fn in conf_files:
        shutil.copy2(fn, os.path.join(conf_dir, fn))
        
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
        packages=[
            "samplernn"
        ],
        python_requires='>=3.8',
        install_requires=[
            "tensorflow >= 2.2.0",
            "librosa >= 0.4.3",
            "natsort >= 7.1.1",
            "pydub >= 0.24.1",
            "keras-tuner >= 1.0.2"
        ],
        package_data={"samplernn": ["conf/*.config.json"]},
        zip_safe=False
    )
finally:
    for fn in conf_files:
        os.remove(os.path.join(conf_dir, fn))

    os.rmdir(conf_dir)
