from setuptools import setup, find_packages
setup(
    name='CrossSiameseNet',
    version='0.1.0',
    install_requires=["scikit-fingerprints"],
    packages=find_packages(include=['CrossSiameseNet', 'CrossSiameseNet.*'])
)