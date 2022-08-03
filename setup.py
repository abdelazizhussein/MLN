from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLN",
    author="Abdelaziz Hussein,Ouail Kitouni, Niklas Nolte'",
    author_email="abdelh@mit.edu,kitouni@mit.edu, nnolte@mit.edu",
    description="Tools for Monotonic Lipschitz Network development",
    packages=find_packages(),
    install_requires=requirements,
    scripts=[],
)
