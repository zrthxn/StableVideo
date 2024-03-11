from setuptools import setup, find_packages
from pathlib import Path

def read_requirements():
    return list(Path('requirements.txt').read_text().splitlines())

setup(
    name='StableVideo',
    version='1.0.0',
    packages=find_packages(),
    install_requires=read_requirements()
)