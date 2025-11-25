"""Installation script for the 'nonlinear_ct' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy<2",
    "torch>=2.7",
    "onnx>=1.18.0",  # 1.16.2 throws access violation on Windows
    "prettytable==3.3.0",
    "toml",
    # reinforcement learning
    "gymnasium==1.2.1",
    # procedural-generation
    "trimesh",
    "pyglet<2",
    # image processing
    "transformers",
    "einops",  # needed for transformers, doesn't always auto-install
    "warp-lang",
    # make sure this is consistent with isaac sim version
    "pillow==11.2.1",
    # livestream
    "starlette==0.45.3",
    # testing
    "pytest",
    "pytest-mock",
    "junitparser",
    "flatdict==4.0.1",
    "flaky",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu128"]

# Installation operation
setup(
    name="nonlinear_cfg",
    author="Kyle Morgenstein",
    maintainer="Kyle Morgenstein",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="MIT",
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    packages=["nonlinear_ct"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
