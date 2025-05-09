import os
import re
import imageio
import argparse


def get_user_input() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--fps", type=int, required=True)
    args = parser.parse_args()
    return args


def create_movie(inpath: str, outpath: str, fps: int) -> None:
    def extract_number(fname):
        match = re.search(r'snapshot_(\d+)', fname)
        return int(match.group(1)) if match else -1

    # Get and sort PNG files by the number in the filename
    images = sorted(
        [f for f in os.listdir(inpath) if f.startswith('snapshot_') \
            and f.endswith('.png')],
        key=extract_number
    )

    # Create the video
    with imageio.get_writer(outpath, fps=fps) as writer:
        for filename in images:
            image = imageio.imread(inpath + filename)
            writer.append_data(image)


if __name__ == "__main__":
    args = get_user_input()
    create_movie(args.inpath, args.outpath, args.fps)
