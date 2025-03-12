from setuptools import setup, find_packages

setup(
    name="pypipe",
    description="A simple, easy to use library for automated machine learning and highly composable pipelines. Made with scikit-learn and pandas.",
    long_description=open("README.md").read(),
    packages=find_packages(),
    version="0.1.0",
    install_requires=["numpy", "seaborn", "pandas", "scikit-learn", "pytest"],
    author="Joel Burleson",
    url="https://github.com/fburleson/pypipe",
    python_requires=">=3.11",
)
