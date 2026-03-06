import pathlib

import setuptools


def parse_reqs(filename):
    """解析 requirements 文件。"""
    requirements = pathlib.Path(filename).read_text(encoding="utf-8").splitlines()
    cleaned = [x.lstrip("\ufeff").strip() for x in requirements]
    return [x for x in cleaned if x and not x.startswith("#")]


setuptools.setup(
    name="dreamerv3-torch-lite",
    version="0.1.0",
    author="DreamerV3 Torch Lite",
    description="精简版 PyTorch DreamerV3 实现",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    # 仅打包精简后的 dreamerv3 核心代码，避免把旧版大模块一起安装。
    packages=["dreamerv3"],
    include_package_data=True,
    install_requires=parse_reqs("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
