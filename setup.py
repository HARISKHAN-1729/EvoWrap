from setuptools import setup, find_packages

setup(
    name="evowrap",
    version="0.2.0",
    description="Agent-agnostic self-evolving framework for continual learning",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
        "faiss": ["faiss-cpu>=1.7.0"],
    },
)
