import numpy as np
import pandas as pd
import argparse
from multiprocessing import Pool
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from auriga.support import make_snapshot_number
from auriga.snapshot import Snapshot
from auriga.parser import parse
from auriga.paths import Paths
from auriga.support import timer


def read_data(simulation: str) -> pd.DataFrame:
    s = Snapshot(simulation=simulation, loadonlytype=[0, 1, 2, 3, 4, 5])
    s.add_extra_coordinates()
    
    is_real_star = (s.type == 4) & (s.stellar_formation_time > 0)

    props = {
        "Mass_Msun": s.mass[is_real_star],
        "SphericalRadius_ckpc": s.r[is_real_star],
        "Halo": s.halo[is_real_star],
        "Subhalo": s.subhalo[is_real_star],}

    df = pd.DataFrame(props)
    df.halo_idx = s.halo_idx
    df.subhalo_idx = s.subhalo_idx
    df.time = s.time

    return df


def calculate_virial_mass_in_snapshot(simulation: str) -> list:
    if parse(simulation)[3] < 35:
        return [parse(simulation)[3]] + [np.nan] * 4

    df = read_data(simulation)
    r200_ckpc = pd.read_csv(
        f"results/{'_'.join(simulation.split('_')[:-1])}/temporal_data.csv").\
            at[parse(simulation)[3], "VirialRadius_ckpc"]
    res = [
        parse(simulation)[3],  # Snapshot number
        df.time,  # Cosmic time in Gyr
        df["Mass_Msun"][df["SphericalRadius_ckpc"] <= r200_ckpc].sum() \
            - df["Mass_Msun"][
                (df["SphericalRadius_ckpc"] <= r200_ckpc) \
                    & (df["Halo"] == df.halo_idx) \
                        & (df["Subhalo"] == df.subhalo_idx)
            ].sum(),
        df["Mass_Msun"][df["SphericalRadius_ckpc"] <= 0.5 * r200_ckpc].sum() \
            - df["Mass_Msun"][
                (df["SphericalRadius_ckpc"] <= 0.5 * r200_ckpc) \
                    & (df["Halo"] == df.halo_idx) \
                        & (df["Subhalo"] == df.subhalo_idx)
            ].sum(),
        df["Mass_Msun"][df["SphericalRadius_ckpc"] <= 0.3 * r200_ckpc].sum() \
            - df["Mass_Msun"][
                (df["SphericalRadius_ckpc"] <= 0.3 * r200_ckpc) \
                    & (df["Halo"] == df.halo_idx) \
                        & (df["Subhalo"] == df.subhalo_idx)
            ].sum(),
    ]
    return res


@timer
def analyze_simulation(simulation: str) -> None:
    paths = Paths(
        parse(simulation)[0], parse(simulation)[1], parse(simulation)[2])
    n_snapshots = make_snapshot_number(
        parse(simulation)[1], parse(simulation)[2])
    data = np.array(
        Pool(4).map(
            calculate_virial_mass_in_snapshot,
            [f"{simulation}_s{i}" for i in range(n_snapshots)]))
    df = pd.DataFrame(data, columns=[
        "SnapshotNumber", "Time_Gyr", "StellarMassInR200_Msun",
        "StellarMassIn0.5R200_Msun", "StellarMassIn0.3R200_Msun"])
    df["SnapshotNumber"] = df["SnapshotNumber"].astype("int")
    df["Time_Gyr"] = df["Time_Gyr"].round(3)
    df.to_csv(f"{paths.results}/mass_in_virial_radius.csv", index=False)


def get_user_input() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=str, required=False)
    args = parser.parse_args()
    return vars(args)


def main() -> None:
    BLUE = "\033[94m"

    args = get_user_input()

    s = ("Au" + str(parse(args['simulation'])[0])).rjust(4)
    print(f"{BLUE}Processing {s}... {BLUE}",
          end="", flush=True)
    analyze_simulation(args["simulation"])
    print(f"{BLUE}Done.{BLUE}", flush=True)


if __name__ == "__main__":
    main()
