
# scripts/01_prepare_targets.py
from pathlib import Path
import glob
import time as tm
import pandas as pd
import sys



SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.target import Target
from stages.dataprep import DataPrep
from utils.queue import enqueue


def prepare_one_target(target_dir: Path) -> None:
    ticid, gaia_id = Target.discover_ids_from_dirname(target_dir)
    dummy = pd.Series(dtype=object)
    t = Target(ticid=ticid, gaia_id=gaia_id, root_dir=target_dir, catalog_row=dummy)
    t.load_state()  # safe if re-running

    dp = DataPrep(target=t, flavour="TGLC")
    total_file = dp.prepare()
    enqueue("02", t.ticid)
    
    print(f"Prepared TIC {t.ticid}: {total_file}")
    

if __name__ == "__main__":
    t0 = tm.time()
    target_dirs = sorted(glob.glob("../../toi_data/target_*"))  # adjust your path
    for td in target_dirs:
        try:
            prepare_one_target(Path(td))
        except Exception as e:
            print(f"[WARN] {td}: {e}")
    t1 = tm.time()
    print("num files", len(target_dirs))
    print("time it took:", (t1 - t0)/60, "minutes")
