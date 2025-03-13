# pypipe :gear:
A simple, easy to use library for automated machine learning and highly composable pipelines. Made with scikit-learn and pandas.

## Introduction :book:
This project was made to showcase my understanding of the machine learning pipeline/workflow and scikit-learn. It allows for quick development by automating the machine learning pipeline, from data preprocessing to model evaluation. Made to solve classification and regression problems whilst being being modular and easy to use.

## Features :sparkles:
- Building highly composable, custom machine learning pipelines, from preprocessing to model evaluation
- Builtin compatibility with numpy, pandas and scikit-learn
- Builtin compatibility with custom functions and transformers
- Saving and loading pipelines

## Requirements :clipboard:
- Python >= 3.11

## Installation :gear:
### Linux
```bash
git clone --depth 1 git@github.com:fburleson/pypipe.git pypipe
cd pypipe
python3 -m venv .venv
source .venv/bin/activate 
pip install -e .
pytest
```
### Windows
```bash
git clone --depth 1 https://github.com/fburleson/pypipe.git pypipe
cd pypipe
python -m venv .venv
.\.venv\Scripts\activate 
pip install -e .
pytest
```

## Usage :computer:
```bash
python run.py
```
```bash
python3 run.py
```

