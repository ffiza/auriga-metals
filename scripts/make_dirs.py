import os
import argparse

from auriga.settings import Settings
from auriga.paths import Paths


def new_dir(path: str):
    if not os.path.isdir(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        type=str,
                        required=True,
                        help="The directory in which to create folder.")
    args = parser.parse_args()
    settings = Settings()
    for galaxy in settings.galaxies:
        new_dir(path=f"{args.dir}/au{galaxy}_or_l4/")
        if galaxy in settings.reruns:
            new_dir(path=f"{args.dir}/au{galaxy}_re_l4/")


if __name__ == "__main__":
    main()
