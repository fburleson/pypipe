import dill
from pypipe.compose import Pipeline


def save_pipeline(pipeline: Pipeline, file_name: str):
    with open(file_name, "wb") as file:
        dill.dump(pipeline, file)
    print(f"saved pipeline as {file_name}")


def load_pipeline(file_path: Pipeline):
    with open(file_path, "rb") as file:
        loaded: Pipeline = dill.load(file)
    print(f"loaded pipeline {file_path}")
    return loaded
