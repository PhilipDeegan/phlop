from setuptools import find_packages, setup

setup(
    name="phlop",
    version="0.0.13",
    cmdclass={},
    classifiers=[],
    include_package_data=True,
    packages=find_packages(exclude=["lib/", "tests/"]),
)
