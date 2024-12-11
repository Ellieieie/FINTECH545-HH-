from setuptools import setup, find_packages

setup(
    name="MyLibrary",
    version="0.0.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

