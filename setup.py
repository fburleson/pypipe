from setuptools import setup, find_packages

setup(
    name="pypipe",
    description="A small python library for automated machine learning and creating pipelines.",
    long_description=open("README.md").read(),
    packages=find_packages(),
    version="0.1.0",
    install_requires=["numpy", "seaborn", "pandas", "scikit-learn", "pytest"],
    author="Joel Burleson",
    url="https://github.com/fburleson/pypipe",
    python_requires=">=3.10",
)
