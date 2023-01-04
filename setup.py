from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="lpsnav",
    version="1.0.0",
    description="A navigation framework capable of reasoning on its legibility and predictability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Jean-Luc Bastarache",
    author_email="jbastarache@uwaterloo.ca",
    url="https://github.com/jlbas/LPSNav",
    packages=find_packages(),
    install_requires=[
        "matplotlib"
        "numpy",
        "pandas",
        "seaborn",
        "sympy",
        "scipy",
        "tabulate",
        "tensorflow",
        "tomli",
        "tqdm",
    ],
)
