import subprocess


def main():
    models: dict = {
        "iris": [
            ["examples/iris.py"],
            """A simple multi classifier using logistic regression and the classic iris data set""",
        ],
    }
    is_exit = "n"
    while is_exit != "y":
        args: list[str] = input(
            f'choose a script to run (type "help" to see options) {tuple(models.keys())}: '
        ).split()
        try:
            if len(args) == 0:
                continue
            if "help" in args:
                print("-d\tto see the description of a model")
                continue
            if "-d" in args:
                print(models[args[0]][1])
                continue
            cmd: list[str] = ["python3", *models[args[0]][0]]
            subprocess.run(cmd)
        except KeyError:
            print(f"{args[0]} is invalid input")
        is_exit: str = input("exit? (y/n): ")


if __name__ == "__main__":
    main()
