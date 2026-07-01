"""Setup script for Adam-TCA optimizer."""

import os
from setuptools import setup, find_packages


def read_readme() -> str:
    """Read the README file for use as the long description."""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="adam-tca",
    version="0.2.0",
    description=(
        "Adam-TCA: Curvature-Aware Adam Optimizer with "
        "Cosine-Similarity and Variance-Based Learning Rate Modulation"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Ali Zafar",
    author_email="alizafar780@example.com",
    url="https://github.com/AliZafar780/Adam-TCA",
    license="Apache 2.0",
    packages=find_packages(include=["adam_tca", "adam_tca.*"]),
    py_modules=["adam_tca"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "torch>=1.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=(
        "pytorch optimizer adam curvature cosine-similarity "
        "gradient-variance deep-learning transformer"
    ),
)
