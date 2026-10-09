"""Download the CTU-UHB Intrapartum Cardiotocography Database v1.0.0 from PhysioNet.

The data are NOT redistributed in this repository. They are openly available from PhysioNet
(https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) under the terms stated there. Cite the
database and PhysioNet as PhysioNet requires.

    python scripts/download_data.py            # -> data/raw/ctu-uhb-ctgdb/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import wfdb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default="data/raw/ctu-uhb-ctgdb")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("ctu-uhb-ctgdb", dl_dir=str(dest))
    n_hea = len(list(dest.glob("*.hea")))
    n_dat = len(list(dest.glob("*.dat")))
    print(f"downloaded to {dest}: {n_hea} .hea, {n_dat} .dat, RECORDS present: {(dest / 'RECORDS').exists()}")
    if n_hea != 552 or n_dat != 552:
        raise SystemExit("expected 552 records; the download looks incomplete")


if __name__ == "__main__":
    main()
