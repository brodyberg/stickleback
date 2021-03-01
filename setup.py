import os
from setuptools import setup, find_packages


setup(
    name="stickleback",
    description="Detect point behaviors in longitudinal sensor data",
    author="Max Czapanskiy",
    packages=find_packages(exclude=['data', 'figures', 'output', 'notebooks'])
)
